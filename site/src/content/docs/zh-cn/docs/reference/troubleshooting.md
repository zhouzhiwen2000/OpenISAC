---
title: 故障排查
description: 常见失败模式和优先检查项。
---

## 编译失败

先检查缺失的开发包：UHD、Boost、FFTW3、yaml-cpp、ZeroMQ、OpenMP 和 Aff3ct。如果 CMake 缓存可疑，使用干净重编译。

## 运行时找不到 YAML

从包含 `BS.yaml` 或 `UE.yaml` 的目录启动。后端读取当前工作目录，不读取 `config/`。

## UHD 流错误

降低采样率，检查网络或 USB 吞吐，确认设备地址和 MTU，避免共用高吞吐控制器。

## UE 无法解码

按顺序检查时序锁定、CFO/SFO、信道估计和 LDPC 诊断。

## 感知输出不稳定

检查同步、校准值、sensing stride、前端速率以及主机是否过载。双站感知还应检查时序偏移假设。

## 前端不更新

检查 ZeroMQ 端点、防火墙、进程启动顺序，以及后端是否确实发布了预期数据流。
