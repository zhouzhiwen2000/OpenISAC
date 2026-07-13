---
title: 参考手册
description: OpenISAC 运行配置、工具脚本和故障排查入口。
---

本栏目用于查找参数、命令和故障处理方法。如果你是第一次部署 OpenISAC，建议先完成[快速开始](/zh-cn/docs/getting-started/hardware/)；需要修改某个 YAML 参数、启动辅助工具或定位运行问题时，再回到这里。

## 运行配置

- [BS YAML 参考](./bs-yaml/)：查询 BS 的射频、下行、上行、感知、网络输出和日志参数。
- [UE YAML 参考](./ue-yaml/)：查询 UE 的接收、同步跟踪、上行、双站感知和输出参数。

BS 和 UE 都从当前工作目录读取 YAML，且运行中不会自动重载。修改配置后，需要重启对应后端才能生效。

## 工具与问题定位

- [脚本与工具](./scripts-tools/)：按用途查找感知可视化、定时控制、CPU 隔离、网页配置和文档站命令。
- [故障排查](./troubleshooting/)：按现象检查编译、YAML、USRP 流、通信解码、感知输出和前端连接。

## 查找参数的建议方式

1. 先确认正在修改 `BS.yaml` 还是 `UE.yaml`。
2. 使用完整路径搜索参数，例如 `uplink.enabled` 或 `network_output.control_port`。
3. 将“典型值”作为参考，实际值以选用的 `config/*.yaml` 模板和当前运行场景为准。
4. 修改帧结构、双工方式、频率或资源映射时，同时核对 BS 和 UE 两侧的对应参数。
