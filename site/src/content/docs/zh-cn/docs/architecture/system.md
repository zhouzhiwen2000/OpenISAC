---
title: 整体架构
description: OpenISAC 系统中的硬件和软件角色。
---

OpenISAC 围绕 BS 节点和 UE 节点构建。每个节点包含主机 PC、USRP、运行时 YAML 配置和 C++ 处理流水线。

![OpenISAC 系统架构](/images/SysArch.png)

## BS 节点

BS 生成下行 OFDM 波形，将其流式发送到 USRP，接收感知回波，并计算单站感知结果。它还负责 payload 输入和前端数据发布。

## UE 节点

UE 接收下行波形，估计同步和信道状态，解码通信 payload，并可发布双站感知数据。运行时的时序信息也会被 OTA/eRTM 工作流使用。

## 前端工具

Python 工具通过 ZeroMQ 接收感知或状态流，进行绘图，并提供控制/配置工作流。前端与硬实时无线链路刻意解耦。
