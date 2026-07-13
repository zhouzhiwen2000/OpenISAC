---
title: 下行通信
description: BS→UE 的信道估计、CFO/SFO 跟踪、均衡、软解映射与 LDPC 解码。
---

下行处理从 UE 已完成帧定时并取得第 $\gamma$ 帧的 FFT 栅格 $\boldsymbol Y_\gamma^\mathrm{DL}$ 开始。完整链路依次为

$$
\text{ZC 信道估计}
\rightarrow
\text{导频相位跟踪}
\rightarrow
\text{逐符号信道重建}
\rightarrow
\text{均衡}
\rightarrow
\text{QPSK LLR}
\rightarrow
\text{LDPC 解码}.
$$

TDD 只处理 $m\in\mathcal S_\mathrm{DL}$ 的资源，FDD 则处理独立下行载波上的完整帧。

## 频域接收模型

在循环前缀覆盖信道时延扩展、单符号内多普勒足够小的条件下，

$$
Y_{n,m,\gamma}^\mathrm{DL}
=b_{n,m,\gamma}^\mathrm{DL}
H_{n,m,\gamma}^\mathrm{DL}
+Z_{n,m,\gamma}^\mathrm{DL},
$$

其中

$$
H_{n,m,\gamma}^\mathrm{DL}
=\sum_{l=1}^{L_\mathrm{DL}}
\alpha_l^\mathrm{DL}
e^{j2\pi\left[
(f_{D,l}^\mathrm{DL}+\Delta\bar f_{c,\gamma}^\mathrm{DL})
(m+\gamma M)T_O
-\kappa_n\Delta f
(\tau_l^\mathrm{DL}+\bar\tau_{d,\gamma,m}^\mathrm{DL})
\right]}.
$$

$\Delta\bar f_{c,\gamma}^\mathrm{DL}$ 和 $\bar\tau_{d,\gamma,m}^\mathrm{DL}$ 表示当前帧补偿后仍残留的频偏与时偏。

## 信道参考与帧内重建

全带宽 ZC 给出初始 LS 估计

$$
\hat H_{n,m_\mathrm{sync},\gamma}^\mathrm{LS}
=\frac{Y_{n,m_\mathrm{sync},\gamma}^\mathrm{DL}}
{z_n^\mathrm{DL}}.
$$

该估计在时延域限制到循环前缀支撑区并进行 Wiener 平滑后，得到噪声更低的信道锚点。若帧中还包含全带宽信道参考符号，则每个参考位置都可形成锚点 $\hat H_{n,m_a,\gamma}$。位于相邻锚点 $m_a<m<m_b$ 之间的信道可按

$$
\hat H_{n,m,\gamma}^{\mathrm{base}}
=(1-\xi_m)\hat H_{n,m_a,\gamma}
+\xi_m\hat H_{n,m_b,\gamma},
\qquad
\xi_m=\frac{m-m_a}{m_b-m_a}
$$

插值。随后叠加由导频估计得到的残余 CFO/SFO 相位，使每个数据符号都使用与其时刻对应的 $\hat H_{n,m,\gamma}^\mathrm{DL}$。没有额外锚点时，该过程退化为从同步符号向整帧传播。

## ZF 与 MMSE 均衡

零迫（ZF）均衡系数为

$$
G_{n,m}^{\mathrm{ZF}}
=\frac{(\hat H_{n,m}^\mathrm{DL})^*}
{\max(|\hat H_{n,m}^\mathrm{DL}|^2,\epsilon)},
$$

正则化 MMSE 系数为

$$
G_{n,m}^{\mathrm{MMSE}}
=\frac{(\hat H_{n,m}^\mathrm{DL})^*}
{|\hat H_{n,m}^\mathrm{DL}|^2+\hat\sigma_Z^2/E_s}.
$$

ZF 在高信噪比下保持星座幅度直接，MMSE 在深衰落处限制噪声放大。数据资源上的均衡符号统一写为

$$
\hat d_{n,m,\gamma}^\mathrm{DL}
=G_{n,m}Y_{n,m,\gamma}^\mathrm{DL},
\qquad(n,m)\in\Omega_\mathrm{data}^\mathrm{DL}.
$$

均衡后的导频误差给出等效噪声方差

$$
\hat\sigma_\mathrm{eq}^2
=\frac{1}{|\Omega_\mathrm{ref}^\mathrm{DL}|}
\sum_{(n,m)\in\Omega_\mathrm{ref}^\mathrm{DL}}
|\hat p_{n,m}^\mathrm{DL}-p_{n,m}^\mathrm{DL}|^2.
$$

## QPSK 软信息与 LDPC

对每个均衡符号 $\hat d$，第 $i$ 个比特的 max-log LLR 为

$$
L_i(\hat d)
\approx
\frac{
\min\limits_{a\in\mathcal A_i^{(1)}}|\hat d-a|^2
-\min\limits_{a\in\mathcal A_i^{(0)}}|\hat d-a|^2
}{\hat\sigma_\mathrm{eq}^2},
$$

其中 $\mathcal A_i^{(b)}$ 是第 $i$ 位标记为 $b$ 的 QPSK 子集。保留 LLR 幅度比先做硬判决能向 LDPC 解码器提供可靠度信息。

发送端的逻辑顺序为

$$
\text{信息比特}
\rightarrow\text{LDPC 编码}
\rightarrow\text{加扰}
\rightarrow\text{交织}
\rightarrow\text{QPSK 映射},
$$

UE 按相反顺序执行软解交织、软解扰和 LDPC 解码，最终恢复下行信息比特。与此同时，$\hat H_{n,m,\gamma}^\mathrm{DL}$ 与均衡判决还会继续供[双站感知](/zh-cn/docs/signal-processing/bistatic-sensing/)使用。
