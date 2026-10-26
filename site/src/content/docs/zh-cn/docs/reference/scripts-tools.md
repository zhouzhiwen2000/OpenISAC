---
title: 脚本与工具
description: 按用途查找 OpenISAC 常用辅助脚本、启动命令和使用条件。
---

本页只列出用户需要直接启动的工具。如无特别说明，命令都在仓库根目录执行，并需先安装 `requirements.txt` 中的 Python 依赖。

## 感知可视化

两个界面的用途、截图、按钮功能和实验流程见[感知界面操作指南](/zh-cn/docs/tools-workflows/sensing-viewers/)。

BS 和 UE 后端已经运行且开启对应感知输出后，启动查看器：

```bash
# BS 多通道单站感知
python3 scripts/plot_sensing_fast.py

# UE 双站感知
python3 scripts/plot_bi_sensing_fast.py
```

如果窗口能打开但没有数据，先检查 YAML 中的感知输出开关、IP/端口和后端日志，而不是重复启动查看器。

## OCXO 调试

```bash
python3 scripts/plot_ocxo_debug.py --host 127.0.0.1 --port 12347
python3 scripts/plot_ocxo_debug.py --channels 6,7,8,13,15 --record capture/ocxo.csv
```

查看器按单位分组显示曲线，保留可配置的滚动历史，并可暂停绘图而不中断接收和 CSV 记录。
通道勾选只影响曲线显示，CSV 始终保存完整的 16 通道数据包。

## 上行定时控制

`uplink_timing_control.py` 通过 BS 和 UE 的 ZMQ 控制端口读取、调整 `DUTI` 和 `TADV` 定时值：

```bash
python3 scripts/uplink_timing_control.py \
  --bs-host 127.0.0.1 --bs-port 9999 \
  --ue-host 127.0.0.1 --ue-port 10001
```

远程运行时，将 host 换成 BS/UE 的实际 IP，并确认 `network_output.control_port` 与命令一致。

## CPU 隔离与后端启动

`isolate_cpus.py` 从 `BS.yaml` / `UE.yaml` 读取关键 CPU 绑定。先应用隔离，再通过 `run` 启动后端：

```bash
cd build
sudo ../scripts/isolate_cpus.py
sudo ../scripts/isolate_cpus.py run ./BS
```

默认命令会询问本机角色是 BS、UE 还是 BS+UE。只要存在内核隔离 CPU 就会显示网卡 IRQ 二级菜单；至少有两个可用的隔离物理核心时才显示双队列 RX/TX 分流。`run` 允许 YAML 关键线程使用指定 CPU，同时会从进程中排除专用网卡 IRQ 核。

## 网页配置台

```bash
python3 scripts/config_web_editor.py --host 127.0.0.1 --port 8765
```

浏览器打开 `http://127.0.0.1:8765/`。该工具编辑当前 `build/BS.yaml` 和 `build/UE.yaml`，并提供后端启停与 CPU 隔离操作。具体使用方法见[网页配置台](/zh-cn/docs/tools-workflows/web-config-console/)。

`config_web_editor_schema.yaml`、`config_web_editor.js` 和相关 HTML/CSS 是网页配置台的内部组成文件，不需要单独执行。

## 文档站

```bash
# 本地预览
cd site
npm run dev -- --host 0.0.0.0

# 构建并发布到仓库根目录 docs/
npm run build
```

`npm run build` 会先生成 `site/dist/`，再调用 `scripts/publish_docs_site.py` 更新 `docs/`。不要直接修改 `docs/` 下的生成 HTML。

## 感知物理坐标

fast viewer 和 backend sensing viewer 的 Doppler 坐标统一显示 **Hz**。单站频谱显示**距离（m）**，双站频谱显示**相对时延（ns）**；微多普勒时间轴显示秒。检测器参数和 bin 选择控件仍以 bin 索引配置。

后端与 viewer 必须一同更新：仅接收 **V10 `PARM` 参数包**，旧版本或未知格式将被拒绝。后端提供原始 OFDM FFT 长度、CP 长度、实际感知符号步长及均匀子载波步长；viewer 使用实际处理 FFT 长度（包括补零）换算坐标。显示裁剪或降采样不会改变物理标定。非均匀 compact 子载波集合不存在统一的 FFT-bin 时延比例，此时明确显示物理时延不可用。

设采样率为 $F_s$，OFDM FFT 长度为 $N$，CP 长度为 $N_{CP}$，符号步长为 $S_t$，子载波步长为 $S_f$，处理 FFT 长度为 $N_R,N_D$：

$$
f_{slow}=\frac{F_s}{(N+N_{CP})S_t},\quad \Delta f_D=\frac{f_{slow}}{N_D},\quad \Delta\tau=\frac{N}{F_s S_f N_R}.
$$

单站按 $R=c\tau/2$ 换算距离；双站按 $10^9\tau$ 显示 ns，**零点相对于当前同步／对齐参考**，不能当作发射机或接收机到目标的绝对距离。单站零距离也依赖系统时延对齐标定。改变 FFT 长度会改变坐标的 bin 间隔，不会改变波形本身的物理分辨率。


## 校准工具

`calibrate_hsys.py capture` 请求校准并验证新保存的响应，`calibrate_hsys.py inspect` 离线检查文件。`calibrate_system_delay.py` 按通道分析后端日志中最近的 alignment 建议。准备步骤、命令和验证范围见[校准流程](/zh-cn/docs/tools-workflows/calibration-workflows/)。

### 单站可选径向速度显示

在 `plot_sensing_fast.py` 的 **Display & View** 中，将实际发射的射频载频填入 **Velocity RF Carrier (GHz)**，点击 **Set** 或按回车。主谱图横轴、微多普勒图纵轴和检测目标列表随即显示径向速度，单位 **m/s**。清空输入框并应用后，恢复多普勒频率 **Hz**。已应用的值会自动保存，下次启动继续使用。

换算关系为 $v_r = f_D c / (2 f_c)$，正负号沿用图中的多普勒频率。使用外接毫米波变频器时，应填写变频后的实际射频载频，而非 USRP 中频；例如只有实际射频为 77 GHz 时才填 `77`。该设置仅用于显示换算，不改变设备调谐、FFT 运算、检测门限或 bin 坐标。正负速度对应的实际运动方向需结合变频器的频谱方向和实验约定确认。双站界面保持多普勒频率显示，因为不能直接套用单站公式换算径向速度。

主谱图右上角以黄色文字单独显示最强检测回波的距离及速度，保留一位小数；载频留空时显示多普勒频率。信息框位置与尺寸固定，没有检出目标时数值显示 **N/A**。

单站快速上位机在丢帧或帧抽取时保留微多普勒滚动历史，按缺失的慢时间位置补零。这样可保留真实时间间隔，但不能恢复缺失回波，空档仍可能降低谱峰或引入旁瓣。改变距离单元、采样几何或时间戳回退时，历史重新建立。
