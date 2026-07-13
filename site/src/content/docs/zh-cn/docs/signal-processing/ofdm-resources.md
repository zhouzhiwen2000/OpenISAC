---
title: OFDM 资源
description: 帧结构、pilot、同步符号和资源映射。
---

OpenISAC 使用可配置 OFDM 资源栅格，而不是标准 Wi-Fi/NR 帧结构。这样可以保持 PHY 简洁，并让感知资源显式可控。

![OFDM 帧结构示例](/images/OpenISAC_FrameStructure.png)

## 连续帧

OpenISAC 采用连续 OFDM 传输，而不是由业务驱动的 packet burst。

![分组无线电与连续波形时序](/images/PacketVSCW.png)

分组无线电的帧间隔不规则，会造成慢时间采样不均匀，从而恶化多普勒或微多普勒处理。连续帧提供确定的 OFDM 符号间隔，因此感知符号可以进行更长的相干积累。

## 符号集合

令 $m_\mathrm{sync}$ 为每帧必选的全带宽 Zadoff-Chu 同步符号。可选 acquisition 字段可以在 $m_\mathrm{sync}-1$ 增加第二个 ZC 符号，并在 $m_\mathrm{sync}+1$ 增加重复 CFO training 字段。

定义：

$$
\mathcal{S}_\mathrm{ZC}=\{m_\mathrm{sync}\}\cup\mathcal{S}_\mathrm{sec},
\quad
\mathcal{S}_\mathrm{CFO}=\{m_\mathrm{sync}+1\}\ \text{if enabled}
$$

令 $\mathcal{P}$ 为 pilot 子载波集合，$\mathcal{D}=\{0,\ldots,N-1\}\setminus\mathcal{P}$ 为数据子载波集合。

频域资源映射为：

$$
b_{n,m,\gamma}=
\begin{cases}
z_n, & m\in\mathcal{S}_\mathrm{ZC},\\
c_n^\mathrm{CFO}, & m\in\mathcal{S}_\mathrm{CFO},\\
z_n, & m\notin\mathcal{S}_\mathrm{ZC}\cup\mathcal{S}_\mathrm{CFO},\ n\in\mathcal{P},\\
d_{n,m,\gamma}, & m\notin\mathcal{S}_\mathrm{ZC}\cup\mathcal{S}_\mathrm{CFO},\ n\in\mathcal{D}.
\end{cases}
$$

其中 $z_n$ 是 ZC 参考序列，$c_n^\mathrm{CFO}$ 是可选 CFO training 符号，$d_{n,m,\gamma}$ 是编码和加扰后的 QPSK 数据符号。

## 资源角色

- 全带宽 ZC 符号提供时序、粗 CFO 支持和初始信道获取。
- Pilot 子载波跟踪残余 CFO/SFO，并支持帧内信道传播。
- 数据子载波承载通信 payload；没有 payload 时填充随机 QPSK。
- 后端保留完整资源栅格，使感知链路能够去除调制并估计信道矩阵。

## 设计权衡

增大 $N$ 或带宽可提高时延分辨率，但会增加计算和 I/O 负载。增加感知帧长度可提高多普勒分辨率，但会增加延迟。启用额外 acquisition 符号可提升初始鲁棒性，但会占用原本可用于感知的资源。
