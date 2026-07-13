---
title: 视频与 UDP 工作流
description: UDP 流量和视频演示的 payload 工作流。
---

OpenISAC 可通过 OFDM 通信链路承载 UDP payload。视频演示通常把压缩后的 packet 输入同一 payload 路径。

## 启动顺序

先用低速率流量验证 PHY。然后提高 UDP payload 速率。确认链路稳定后，再加入视频编码或解码工具。

## 调试

应区分 PHY 失败和应用失败：

- 同步不稳定时，先修无线链路。
- LDPC 解码失败时，检查信道和偏移估计。
- 解码稳定但 UDP 丢包时，检查队列、datagram 大小和应用 pacing。
- UDP 到达但视频卡顿时，检查 codec 和播放器行为。

这个顺序可以避免把前端或应用症状误认为无线问题。
