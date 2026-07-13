---
title: BS 运行流水线
description: BS 的主要处理阶段和线程职责。
---

BS 运行时是分阶段流水线，将 payload 准备、OFDM 波形生成、USRP 发送、USRP 接收和感知处理分开。

![BS 软件架构](/images/SoftArchBS.png)

## 主要阶段

- Bit processing 接收 UDP payload、填充帧并准备编码比特。
- OFDM 发射处理完成符号映射、pilot 插入、IFFT 和循环前缀添加。
- USRP transmit 将基带样本流送入无线电。
- USRP receive 捕获感知样本。
- Sensing processing 解调接收资源并生成时延-多普勒输出。

## 设计意图

流水线避免把硬件编排和可复用 DSP 混在一起。共享信号处理逻辑应放在 `include/OFDMCore.hpp` 等头文件中，运行时所有权、队列和 UHD 控制靠近可执行入口。

热路径应避免无界分配和阻塞式诊断。
