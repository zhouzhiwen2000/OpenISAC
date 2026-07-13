---
title: 信号处理总览
description: 从双向 OFDM 传输到多通道单站感知和双站感知的统一处理链。
---

OpenISAC 以同一套 OFDM 参数连接四类处理：BS 到 UE 的下行通信、UE 到 BS 的上行通信、BS 侧多通道单站感知，以及 UE 侧双站感知。各部分共享帧索引、子载波索引和参考信号定义，因此通信估计得到的定时、频偏与信道信息可以自然延伸到感知处理。

## 完整处理链

1. BS 和 UE 分别在下行与上行资源栅格中映射同步序列、导频和编码后的 QPSK 符号。
2. 各资源栅格经过 IFFT 与循环前缀插入，形成连续 OFDM 波形；TDD 在同一帧内划分下行、保护和上行符号，FDD 在两个载波上同时传输。
3. 接收端利用全带宽 Zadoff–Chu（ZC）符号完成帧定时和初始信道估计，再利用导频跟踪残余载波频偏（CFO）与采样频偏（SFO）。
4. 通信链路对每个数据资源进行信道均衡、QPSK 软解映射和 LDPC 解码，分别恢复下行与上行信息。
5. BS 将各感知通道的接收栅格除以已知下行发送栅格，得到多通道时频信道张量，并进一步形成距离–多普勒–角度结果或微多普勒谱。
6. UE 先从下行通信判决中重构未知数据符号，再形成双站信道栅格；连续时偏与 SFO 补偿使感知时延轴在长时间观测中保持稳定。
7. 当上下行同时可用时，eRTM 将两个方向的信道时延关系与固定校准项结合，用于分离 BS 与 UE 两侧的定时偏移。

## 全章符号

| 符号 | 含义 |
|---|---|
| $x\in\{\mathrm{DL},\mathrm{UL}\}$ | 下行或上行链路 |
| $\gamma$ | 连续 OFDM 帧索引 |
| $m$ | 帧内 OFDM 符号索引 |
| $n$ | FFT 存储索引；$\kappa_n$ 为对应的有符号子载波索引 |
| $r=0,\ldots,R-1$ | BS 感知阵列通道索引 |
| $\boldsymbol B_\gamma^x$ | 链路 $x$ 的发送资源栅格 |
| $\boldsymbol Y_\gamma^x$ | FFT 后的接收资源栅格 |
| $\boldsymbol H_\gamma^x$ | 通信信道频率响应 |
| $\boldsymbol F_\gamma$ | 去除调制后的感知信道符号 |

斜体小写、粗体小写和粗体大写分别表示标量、向量和矩阵；$(\cdot)^T$、$(\cdot)^H$ 与 $(\cdot)^*$ 分别表示转置、共轭转置和复共轭。

## 阅读顺序

建议按以下顺序阅读：

- [信号模型](/zh-cn/docs/signal-processing/signal-model/)统一定义上下行、多通道单站与双站传播模型。
- [OFDM 资源](/zh-cn/docs/signal-processing/ofdm-resources/)定义连续波形、TDD/FDD 资源集合和参考信号。
- [同步、CFO 与 SFO](/zh-cn/docs/signal-processing/sync-cfo-sfo/)给出两条通信链路共用的获取与跟踪方法。
- [下行通信](/zh-cn/docs/signal-processing/ue-reception/)和[上行通信](/zh-cn/docs/signal-processing/uplink-communication/)完成双向信息恢复。
- [单站感知](/zh-cn/docs/signal-processing/monostatic-sensing/)和[双站感知](/zh-cn/docs/signal-processing/bistatic-sensing/)从同一资源栅格推导感知结果。
- [OTA 与 eRTM 定时](/zh-cn/docs/signal-processing/ota-ertm-timing/)说明相对时延、绝对时偏与长期稳定性之间的关系。

以下推导默认信道最大时延扩展不超过循环前缀，且单个 OFDM 符号内的多普勒远小于子载波间隔。此时符号间干扰与载波间干扰可以忽略，接收栅格可采用逐资源元素模型。
