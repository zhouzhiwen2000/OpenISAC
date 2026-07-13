---
title: UE 运行流水线
description: UE 同步、解调和解码的主要阶段。
---

UE 运行时把连续下行样本转为解码后的 payload 数据。

![UE 软件架构](/images/SoftArchUE.png)

## 主要阶段

- USRP receive 捕获下行流。
- 同步模块搜索已知结构并确定帧起点。
- CFO/SFO 跟踪在运行中保持解调器对齐。
- 信道估计和均衡为数据符号解映射做准备。
- LDPC 解码恢复 payload 比特。
- 可选前端发布状态和感知数据。

## 运行行为

多数 OTA 测试应先启动 UE 再启动 BS。调试时先确认同步稳定，再检查解码结果，最后再调 UDP 或视频等上层工作流。
