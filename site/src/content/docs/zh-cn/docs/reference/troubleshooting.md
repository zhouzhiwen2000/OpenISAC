---
title: 故障排查
description: 按现象排查 OpenISAC 编译、配置、USRP、通信、感知和前端问题。
---

排查时先保留首个明确错误，不要只截取最后一行日志。记录启动命令、当前工作目录、使用的 `BS.yaml` / `UE.yaml` 和 USRP 型号，通常能快速区分配置、环境和链路问题。

## 先做这四项检查

1. 确认进程的当前工作目录中存在正确的 `BS.yaml` 或 `UE.yaml`。
2. 确认 `radio.radio_backend` 与当前场景一致：真实 USRP 使用 `uhd`，仿真使用 `sim`。
3. 对比 BS/UE 的采样率、FFT/CP、帧长、导频、频率、双工方式和资源映射。
4. 将相关 `logging.modules` 模块临时调到 `info` 或 `debug`，复现一次后从第一个 Warn/Error 开始查看。

## 编译失败

先使用标准构建命令复现：

```bash
cmake -S . -B build
cmake --build build -j$(nproc)
```

- CMake 配置阶段失败：查看第一个缺失的包，重点检查 UHD、Boost、FFTW3、yaml-cpp、ZeroMQ、OpenMP 和 Aff3ct。
- 编译阶段失败：保留第一个 compiler error；后面的连锁报错通常只是结果。
- 修改过编译器、CUDA 或依赖路径后仍出现旧配置：使用新的空构建目录重新配置，避免继续复用旧 CMake cache。

## 运行时找不到 YAML

`BS` 和 `UE` 只从当前工作目录读取固定文件名，不会自动读取 `config/` 中的模板。例如：

```bash
cp config/BS_X310.yaml build/BS.yaml
cp config/UE_X310.yaml build/UE.yaml
cd build
./BS
```

如果通过 `sudo` 或启动脚本运行，仍要确认启动时的当前目录是 `build/`。修改 YAML 后必须重启后端。

## 找不到 USRP 或无法初始化

1. 运行 `uhd_find_devices` 确认 UHD 能发现设备。
2. 检查 `usrp_device.device_args` 以及 TX/RX 专用 `device_args` 是否指向正确设备。
3. X310 检查网卡 IP、路由和 MTU；B210 检查 USB 3 连接、线缆和供电。
4. 使用外部时钟或 PPS 时，确认实际已接入对应源；否则先用 `internal` 验证设备基本连接。

## UHD overflow、underflow 或 late packet

- X310：检查网卡链路速率、MTU、包丢失和是否与其他高流量任务共用网卡。
- B210：确认使用 USB 3，避免通过集线器或与高带宽设备共用控制器。
- 两者都应检查 CPU 是否持续过载、关键线程是否按 YAML 绑定，并先降低 `rf_sampling.sample_rate` 验证问题是否与吞吐有关。

## UE 无法完成下行解码

按信号链路顺序检查，不要直接从 LDPC 参数开始调整：

1. UE 是否发现同步峰并进入稳定跟踪。
2. CFO/SFO 是否收敛，是否反复触发重新同步。
3. BS/UE 的 `ofdm_frame.*`、导频位置、帧内导频和资源块是否一致。
4. 信道估计和星座图是否合理，再查看 LLR 与 LDPC 诊断。
5. 在真实射频链路上，检查中心频率、收发增益和天线端口，避免输入过弱或饱和。

## 上行不工作

确认 BS 和 UE 两侧的 `uplink.enabled` 均为 `true`，且 `uplink.duplex_mode`、TDD 符号窗口或 FDD 中心频率相同。使用信道仿真器时，还需要启用 `simulation.enable_uplink`。如果能接收但无法解码，再按定时窗口、信道估计、均衡和 LDPC 的顺序检查。

## 感知输出不稳定或结果异常

- 先确认通信/同步链路稳定；感知处理不能补救持续错帧或失锁。
- 核对 `alignment`、RF 链路时延和通道幅相校准值，并确认 FFT 尺寸与前端解析参数一致。
- 检查后端是否报告帧配对队列、感知输出队列或 ZMQ 高水位丢帧。
- 双站感知还需要确认 `sensing.sensing_delay_correction_mode` 与当前 LoS tracking / eRTM 配置一致。

## 可视化窗口不更新

1. 确认后端已启用相应的 `network_output.*_enabled` 开关。
2. 确认可视化脚本连接的 IP/端口与 YAML 一致，远程主机之间的防火墙允许该 TCP 端口。
3. 在后端主机上用 `ss -ltnp` 检查预期端口是否正在监听。
4. 查看后端日志，确认它在生成数据，且没有因队列满或无订阅者而持续丢帧。

如果上述检查仍不能定位问题，保留完整 YAML、启动命令、从进程启动到首次失败的日志，以及 USRP 连接方式和主机环境信息。
