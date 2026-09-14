---
title: 校准流程
description: 校准系统时延和感知通道响应，减少硬件误差对测量结果的影响。
---

开始感知实验前，建议先校准系统时延，再校准感知通道的系统响应。前者用于对齐感知 RX 帧与直达路径定时参考，后者用于补偿收发链路的幅相响应。按此顺序操作，可以避免把硬件引入的时延或频率响应误认为传播特征。

## 校准系统时延

直连前请先降低 `build/BS.yaml` 中的 `downlink.tx_gain`，并选择合适的衰减器，避免感知 RX 饱和。然后用射频线缆将发射 RF 输出连接到要校准的感知 RX 输入，并在整个校准过程中保持连接稳定。

只为当前要测试的感知通道开启系统时延估计：

```yaml
sensing:
  rx_channels:
    - usrp_channel: 1
      alignment: 63
      enable_system_delay_estimation: true
```

从 `build/` 目录启动 BS 后端。该模式下，被选中的感知通道会关闭常规感知处理链，并周期性执行基于 ZC 相关的系统时延测试。观察 BS 控制台输出：CPU 版本会打印 `[SYSDLY CH <n>]`，CUDA 版本会打印 `[CUDA SYSDLY CH <n>]`。CPU 日志中的建议值字段是 `alignment_suggest=<value>`；CUDA 日志中的建议值字段是 `suggest=<value>`。

当建议值稳定后，停止后端，把该值写回同一通道的 `alignment` 字段，并关闭估计模式：

```yaml
sensing:
  rx_channels:
    - usrp_channel: 1
      alignment: <suggested value>
      enable_system_delay_estimation: false
```

每个感知通道都需要单独执行一次射频直连测量。多通道系统中，请把直连线切换到下一个感知 RX 路径，并更新该通道自己的 `alignment`；不要把一个通道的结果直接复用到其他 RF 路径。

## 校准感知通道系统响应

启动后端和连接射频回环之前，设置合适的增益并选择足够的外部衰减，使 RX 保持在线性区工作。然后连接发射 RF 输出与待校准感知 RX 输入，并保持路径稳定。衰减器在信号带宽内应尽量平坦，否则其带内起伏也会被写入校准结果。

单站感知校准时，先启动 BS 后端和单站感知前端，在前端中切换到需要校准的感知通道，然后点击系统响应校准。如果是单站多通道配置，请分别对需要校准的通道执行一次。

双站感知校准时，先启动 BS 和 UE 两端后端，打开双站感知前端，并在射频直连状态下点击双站前端里的系统响应校准。

等待后端日志提示校准完成并保存校准文件。当前运行会立即使用新的系统响应校准结果；之后重新启动后端时，程序会自动加载匹配的校准文件。如果没有找到匹配文件，程序会继续运行并打印提示，感知处理会跳过系统响应补偿。

校准完成后，先停止相关后端，再恢复正常的天线连接或实验连接，然后继续进行空口测量。如果增益或其他射频设置发生变化，应重新评估校准是否适用。


## 命令行执行与验证 Hsys 校准

独立工具调用现有后端校准功能，无需启动 viewer：

```bash
# BS 单站逻辑感知通道 0，不一定是 USRP 物理通道 0
python3 scripts/calibrate_hsys.py capture --backend-dir build --channel 0

# UE 双站路径，从 build/UE.yaml 读取控制端口
python3 scripts/calibrate_hsys.py capture --backend-dir build --role bistatic --channel 0

# 离线检查现有文件，不连接后端
python3 scripts/calibrate_hsys.py inspect build/sensing_system_response_monostatic_ch0.bin
```

1. 先完成系统时延对齐并关闭 `enable_system_delay_estimation`，使用 dense 或覆盖全带宽的规则 compact mask。
2. **启动后端和连接带衰减的射频回环之前**设置合适的 TX/RX 增益。优先使用预期工作增益配合足够的外部衰减，保证接收机在线性区工作；改变增益可能改变射频响应。增益设为 0 dB **不等于关闭发射**。
3. 后端从实际工作目录读取 YAML，`--backend-dir` 必须指向同一目录。远程后端可通过 `--host` 配合其工作目录的共享挂载访问；仅在本机复制一份 YAML 无法验证保存完成。
4. 校准期间关闭其他控制客户端。现有 `ALCH` 是后端全局选择，其他 viewer 同时操作可能改变请求目标。脚本会明确选择一个逻辑感知通道，并保留该选择。
5. 执行 `capture` 并确认准备完成。`--yes` 只跳过交互确认。`--symbols` 默认 1000，`--timeout` 默认 60 秒，`--connect-timeout` 默认 5 秒；较大的 sensing stride 或较慢采集可能需要延长超时。
6. 等待 **Saved file verified**，然后停止后端，再改变射频连接或恢复增益。各条射频路径分别校准。

工具要求 V9 参数响应，并检查后端角色、FFT 和射频参数是否与 YAML 一致。请求新校准之前，已有文件会备份为唯一命名的 `.bin.<id>.bak`。只有新写入且完整的 Hsys 文件，其角色、通道、FFT、射频参数和采集符号数均匹配，才判定保存完成；旧文件或“命令已发出”不能作为完成证据。JSON 摘要包含 SHA-256、幅度分位数和弱 bin 数量，后端对低于逆响应阈值的 bin 不执行均衡。

脚本不会改变增益、改写 YAML 或自动恢复文件。超时、中断或校验失败时返回非零状态；后端采集可能仍在进行，因此改变接线或增益之前应先停止后端。新校准确认有效之前保留备份。二进制格式使用本机字节序，检查文件的主机应与后端字节序一致。

文件校验只能证明结构和数值一致，不能证明没有削顶、信噪比足够或射频接线正确。目前文件头没有记录设备序列号、增益、天线端口或线缆／衰减器标识；这些射频条件变化后，即使后端仍接受文件，也应重新校准。Hsys 包含回环线缆及衰减器的响应，不是绝对距离或天线校准。

命令行流程借鉴了 [Henry Teall 的校准工具](https://github.com/philippaPA/OpenISAC/commit/9b630f3744a2261b234d22f95bfe6e07b727d4d8)。不包含 TX-off DC 基线采集。

## 分析系统时延日志

保存一次校准运行的后端输出，分析最近的估计结果：

```bash
python3 scripts/calibrate_system_delay.py capture/system-delay.log --channel 0
# 也支持从标准输入读取完整日志
cat capture/system-delay.log | python3 scripts/calibrate_system_delay.py -
```

工具识别 CPU 和 CUDA/ROCm 的 `SYSDLY` 日志格式，按通道分别统计。默认检查最近 20 次观测，至少需要 10 次，建议值极差不超过 2 个采样点，并要求每次相关峰／均值比不低于 10。这些是可配置的筛选阈值（`--window`、`--min-samples`、`--max-spread`、`--min-ratio`），不是统计精度保证。后端序列重启或 alignment 改变时重新统计；重复记录不增加样本数。结果不稳定或样本不足时，不给出 alignment 建议并返回非零状态。

检查输出的整数 `alignment` 后，停止后端，更新对应逻辑通道的 YAML 并关闭估计模式。工具不会修改配置或发送硬件命令。请使用同一次后端运行的日志；建议值包含校准路径的影响，并不代表绝对传播时延。
