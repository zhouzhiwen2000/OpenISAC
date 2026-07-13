---
title: UE 双站感知
description: 下行符号重构，以及 OTA LoS tracking 或 eRTM 二选一的连续定时补偿。
---

UE 双站感知与下行通信观察同一个 $h_\mathrm{DL}(t,\tau)$，但目标不同：通信希望消除信道，感知则希望保留信道的频域和慢时间域结构。相对于 BS 单站感知，UE 还需解决两点：数据符号事先未知，以及 BS–UE 定时漂移会直接移动感知时延轴。

## 1. 下行符号重构

ZC、导频和全带宽信道参考都是已知符号。对数据资源，为降低感知重构的复杂度和时延，UE 直接对均衡后的 QPSK 符号做硬判决，不经过 LDPC 重新编码和星座映射：

$$
\tilde b_{n,m,\gamma}^\mathrm{DL}
=\frac{1}{\sqrt2}
\left[
\operatorname{sgn}(\operatorname{Re}\{\hat d_{n,m,\gamma}^\mathrm{DL}\})
+j\operatorname{sgn}(\operatorname{Im}\{\hat d_{n,m,\gamma}^\mathrm{DL}\})
\right].
$$

统一的重构网格为

$$
\tilde b_{n,m,\gamma}=
\begin{cases}
b_{n,m,\gamma}^\mathrm{DL},&(n,m)\in\Omega_\mathrm{ref}^\mathrm{DL}
\text{ 或 }m\in\mathcal S_\mathrm{ZC}^\mathrm{DL},\\
\tilde b_{n,m,\gamma}^\mathrm{DL},&(n,m)\in\Omega_\mathrm{data}^\mathrm{DL}.
\end{cases}
$$

由此去除通信调制：

$$
F_{n,m,\gamma}^\mathrm{UE}
=\frac{Y_{n,m,\gamma}^\mathrm{DL}}
{\tilde b_{n,m,\gamma}}.
$$

若数据判决正确，$F_{n,m,\gamma}^\mathrm{UE}$ 就是 BS→UE 信道的时频采样；判决错误会形成稀疏异常值，因此低信噪比时可只选高置信度数据或已知参考资源进行感知。

## 2. 为什么通信定时不足以支持感知

通信只要求总时延扩展留在循环前缀内。亚采样级时偏可被吸收到 $\hat H_{n,m,\gamma}^\mathrm{DL}$ 的相位中，只有累计漂移接近阈值时才需要移动下行解调边界。

双站感知中，时延本身就是测量量。若直接沿用这些离散的整数采样修正，目标时延轨迹会呈阶梯状，并在时延–多普勒或微多普勒结果中产生人工跳变。因此感知链路需要连续的时偏估计，并将通信帧起点的整数跳变计入同一个连续坐标。

OpenISAC 提供两种可选的双站定时方法：仅使用下行观测的 **OTA LoS tracking**，以及同时使用上下行信道的 **eRTM**。两者是二选一的感知时偏来源，不会同时驱动同一帧的时偏校正。

## 双站定时补偿的两种选项

### OTA LoS tracking 选项

OTA LoS tracking 只使用 UE 的下行信道估计，并持续跟踪 LoS 路径相对于当前下行解调边界的观测坐标。根据[信号模型](/zh-cn/docs/signal-processing/signal-model/#双向通信的时延组成)，该坐标为

$$
\tau_\mathrm{LoS}^\mathrm{UE}(t)
=\tau_{\mathrm{LoS,prop}}(t)
+\tau_\mathrm{TO}^\mathrm{UE}(t).
$$

下行解调窗口位置由 $\tau_d^\mathrm{UE}$ 表示，并包含在 UE 端时偏 $\tau_\mathrm{TO}^\mathrm{UE}$ 中，其关系为

$$
\tau_\mathrm{TO}^\mathrm{UE}
=\tau_\mathrm{DL}^\mathrm{RF}-\tau_d^\mathrm{UE}.
$$

由同步 ZC 的信道估计计算复时延谱

$$
p_\gamma[k]
=\frac{1}{N}\sum_{n=0}^{N-1}
\hat H_{n,m_\mathrm{sync},\gamma}^\mathrm{DL}
e^{j2\pi nk/N}.
$$

设整数峰值为 $k_{\max,\gamma}$，并定义相邻 bin 与主峰的复比值

$$
r_\gamma[q]
=\frac{p_\gamma[k_{\max,\gamma}+q]}
{p_\gamma[k_{\max,\gamma}]},
\qquad q\in\{-1,1\}.
$$

Quinn 型分数估计给出两个候选：

$$
\hat\delta_{\tau,+}
=\frac{r_\gamma[1]}{r_\gamma[1]-1},
\qquad
\hat\delta_{\tau,-}
=\frac{r_\gamma[-1]}{1-r_\gamma[-1]}.
$$

按两个候选实部的符号一致性选择 $\hat\delta_{\tau,\gamma}$ 后，当前 LoS 观测坐标估计为

$$
\hat k_{\tau,\gamma}
=k_{\max,\gamma}+\hat\delta_{\tau,\gamma},
\qquad
\hat\tau_{o,\gamma}=\frac{\hat k_{\tau,\gamma}}{B}.
$$

在从 $\gamma_w$ 开始、长度为 $\Gamma_W$ 的窗口内，令 $\hat k_\mathrm{TO,\gamma}$ 表示通信链路已经施加的整数定时修正。窗口内累计修正为

$$
A_{\gamma_w+\ell}
=\sum_{i=0}^{\ell-1}\hat k_\mathrm{TO,\gamma_w+i},
\qquad \ell=0,\ldots,\Gamma_W-1.
$$

将整数修正加回观测，重构连续轨迹

$$
\tilde k_{\tau,\gamma_w+\ell}
=\hat k_{\tau,\gamma_w+\ell}+A_{\gamma_w+\ell}.
$$

采样时钟误差在短窗口内变化缓慢，因此用线性模型

$$
\tilde k_{\tau,\gamma_w+\ell}
\approx
\epsilon_\mathrm{SIO,w}\ell+k_{\tau,\gamma_w}
$$

做最小二乘拟合。斜率

$$
\epsilon_\mathrm{SIO,w}
=MN_sB\,\Delta T_{as,w}
$$

表示每帧由采样间隔偏移引起的时延漂移。感知时偏递归更新为

$$
\hat k_{\tau,\gamma}^\mathrm{sens}
=\hat k_{\tau,\gamma-1}^\mathrm{sens}
+\hat\epsilon_\mathrm{SIO,w-1}
-\hat k_\mathrm{TO,\gamma-1}
+\mu_\gamma e_\gamma,
$$

$$
e_\gamma
=\hat k_{\tau,\gamma}
-\hat k_{\tau,\gamma-1}^\mathrm{sens}.
$$

$\hat\epsilon_\mathrm{SIO,w-1}$ 预测下一帧漂移，$-\hat k_\mathrm{TO,\gamma-1}$ 抵消通信链路移动帧起点造成的坐标变化，$\mu_\gamma e_\gamma$ 防止长期预测误差积累。对应的频域补偿为

$$
\tilde F_{n,m,\gamma}^\mathrm{UE}
=F_{n,m,\gamma}^\mathrm{UE}
\exp\!\left\{
j2\pi\kappa_n\Delta f
\left(
\frac{\hat k_{\tau,\gamma}^\mathrm{sens}}{B}
+mN_s\Delta\hat T_{as,w-1}
\right)
\right\}.
$$

该选项同时去除 LoS 真实传播时延与 UE 端 TO，输出时延以 LoS 路径为零点。该方法依赖 LoS 路径持续可见；若 LoS 消失或主峰发生路径切换，跟踪到的路径坐标将不再对应同一条物理路径。

### eRTM 双向定时选项 [1]

#### 时延与时偏关系

![eRTM 上下行 OFDM 定时关系](/images/ofdm-timing-diagram.svg)

图中给出了下行参考信号、对应上行参考信号、传播时延、定时提前量与两端观测时延之间的关系。

对同一组下行/上行参考边界，假设下行参考信号和对应的上行参考信号在 OFDM 网格中的理论时间差为 $T$，$t_\mathrm{DL-UL}^\mathrm{BS}$ 为 BS 下行发射与上行接收参考边界之差，$t_\mathrm{TA}^\mathrm{UE}$ 为 UE 上行定时提前量。以 UE 的下行参考边界为零点，第 $l$ 条下行路径在 $\tau_l^\mathrm{UE}$ 时刻到达，上行在 $T-t_\mathrm{TA}^\mathrm{UE}$ 时刻发送，因此

$$
t_{\mathrm{rx-tx},l}^\mathrm{UE}
=T-\tau_l^\mathrm{UE}-t_\mathrm{TA}^\mathrm{UE}.
$$

以 BS 的下行参考边界为零点，BS 的上行接收参考边界位于 $T+t_\mathrm{DL-UL}^\mathrm{BS}$，第 $l$ 条上行路径相对该边界延迟 $\tau_l^\mathrm{BS}$，因此

$$
t_{\mathrm{tx-rx},l}^\mathrm{BS}
=T+\tau_l^\mathrm{BS}+t_\mathrm{DL-UL}^\mathrm{BS}.
$$

减去 UE 内部的接收到发送等待时间后，剩余量等于上下行总链路时延：

$$
t_{\mathrm{tx-rx},l}^\mathrm{BS}
-t_{\mathrm{rx-tx},l}^\mathrm{UE}
=\tau_l^\mathrm{DL}+\tau_l^\mathrm{UL}.
$$

将上述时间间隔以及
$\tau_l^\mathrm{DL}=\tau_{l,\mathrm{prop}}+\tau_\mathrm{DL}^\mathrm{RF}$、
$\tau_l^\mathrm{UL}=\tau_{l,\mathrm{prop}}+\tau_\mathrm{UL}^\mathrm{RF}$ 代入，得到

$$
\tau_l^\mathrm{BS}+\tau_l^\mathrm{UE}
=2\tau_{l,\mathrm{prop}}+\tau_\mathrm{DL}^\mathrm{RF}
+\tau_\mathrm{UL}^\mathrm{RF}
-t_\mathrm{DL-UL}^\mathrm{BS}
-t_\mathrm{TA}^\mathrm{UE}.
$$

将与路径索引 $l$ 无关的量定义为

$$
\tau_c
=\tau_\mathrm{DL}^\mathrm{RF}
+\tau_\mathrm{UL}^\mathrm{RF}
-t_\mathrm{DL-UL}^\mathrm{BS}
-t_\mathrm{TA}^\mathrm{UE},
$$

于是，两端观测到的路径时延满足

$$
\boxed{
\tau_l^\mathrm{BS}+\tau_l^\mathrm{UE}
=2\tau_{l,\mathrm{prop}}+\tau_c
}.
$$

再代入
$\tau_l^\mathrm{UE}=\tau_{l,\mathrm{prop}}+\tau_\mathrm{TO}^\mathrm{UE}$ 与
$\tau_l^\mathrm{BS}=\tau_{l,\mathrm{prop}}+\tau_\mathrm{TO}^\mathrm{BS}$，可得

$$
\boxed{
\tau_\mathrm{TO}^\mathrm{BS}
+\tau_\mathrm{TO}^\mathrm{UE}
=\tau_c
}.
$$

$\tau_c$ 可以由系统校准参数和已知运行时参数直接计算得到。具体地，先将校准得到的下行和上行 RF 群时延相加，再减去 BS 运行时下行发射参考边界与上行接收参考边界之间的偏移，以及 UE 的上行定时提前量。$\tau_l^\mathrm{BS}$ 和 $\tau_l^\mathrm{UE}$ 则分别由 BS 与 UE 端观测。


eRTM 同时使用 BS 端上行信道估计 $\hat H_{\mathrm{BS}}[n]$ 和 UE 端下行信道估计 $\hat H_{\mathrm{UE}}[n]$。两份信道分别取自时刻相近的上行和下行参考符号；由于 TO 变化缓慢，同一测量对中的 $\tau_\mathrm{TO}^\mathrm{BS}(t_{\mathrm{UL}})$ 和 $\tau_\mathrm{TO}^\mathrm{UE}(t_{\mathrm{DL}})$ 可分别简记为 $\tau_\mathrm{TO}^\mathrm{BS}$ 和 $\tau_\mathrm{TO}^\mathrm{UE}$。在 TDD 互易条件下，[信号模型](/zh-cn/docs/signal-processing/signal-model/#tdd-上下行信道关系)给出

$$
H_{\mathrm{BS}}[n]
\approx H_{\mathrm{UE}}[n]
e^{-j2\pi\kappa_n\Delta f
\tau_\mathrm{TO}^{\mathrm{BS-UE}}}.
$$

FDD 的上下行位于不同载波；若两个载波的可见路径集合或路径散射系数差异过大，eRTM 算法的可靠性会下降。eRTM 的第一步是估计差分 TO $\tau_\mathrm{TO}^{\mathrm{BS-UE}}$。可以采用频域最大似然指标或者时延幅度谱指标。

#### 最大似然指标

令 UE 端当前时延轴上的共同未知信道为

$$
H_{0,\gamma}[n]
=\sum_{l=0}^{L-1}\alpha_l
e^{-j2\pi\kappa_n\Delta f
[\tau_{l,\mathrm{prop}}+\tau_\mathrm{TO}^\mathrm{UE}]}
$$

待估计量 $\tau$ 对应 $\tau_\mathrm{TO}^{\mathrm{BS-UE}}$，则观测模型写为

$$
\hat H_{\mathrm{BS}}[n]
=H_{0,\gamma}[n]
e^{-j2\pi\kappa_n\Delta f\tau}
+V_{\mathrm{BS},\gamma}[n],
$$

$$
\hat H_{\mathrm{UE}}[n]
=H_{0,\gamma}[n]+V_{\mathrm{UE},\gamma}[n],
$$

$$
V_{\mathrm{BS},\gamma}[n]
\sim\mathcal{CN}(0,\sigma_{\mathrm{BS},n}^2),
\qquad
V_{\mathrm{UE},\gamma}[n]
\sim\mathcal{CN}(0,\sigma_{\mathrm{UE},n}^2),
$$

并假设两端噪声相互独立。消去共同信道后，加权 ML 估计为

$$
\hat\tau_\mathrm{TO}^{\mathrm{BS-UE},\mathrm{ML}}
=\arg\min_\tau
\sum_{n=0}^{N-1}
\frac{
\left|
\hat H_{\mathrm{BS}}[n]
e^{j2\pi\kappa_n\Delta f\tau}
-\hat H_{\mathrm{UE}}[n]
\right|^2
}{
\sigma_{\mathrm{BS},n}^2+\sigma_{\mathrm{UE},n}^2
}.
$$

若两端公共相位已校准，该式等价于

$$
\hat\tau_\mathrm{TO}^{\mathrm{BS-UE},\mathrm{ML}}
=\arg\max_\tau
\operatorname{Re}\!\left\{
\sum_{n=0}^{N-1}
\frac{
\hat H_{\mathrm{BS}}[n]
\hat H_{\mathrm{UE}}^{*}[n]
}{
\sigma_{\mathrm{BS},n}^2+\sigma_{\mathrm{UE},n}^2
}
e^{j2\pi\kappa_n\Delta f\tau}
\right\}.
$$

若两端还存在未知的公共相位差，则对相关结果取幅度：

$$
\hat\tau_\mathrm{TO}^{\mathrm{BS-UE},\mathrm{ML}}
=\arg\max_\tau
\left|
\sum_{n=0}^{N-1}
\frac{
\hat H_{\mathrm{BS}}[n]
\hat H_{\mathrm{UE}}^{*}[n]
}{
\sigma_{\mathrm{BS},n}^2+\sigma_{\mathrm{UE},n}^2
}
e^{j2\pi\kappa_n\Delta f\tau}
\right|.
$$

对白噪声，分母为与 $n$ 无关的常数，可以省略：

$$
\hat\tau_\mathrm{TO}^{\mathrm{BS-UE},\mathrm{ML}}
=\arg\max_\tau
\left|
\sum_{n=0}^{N-1}
\hat H_{\mathrm{BS}}[n]
\hat H_{\mathrm{UE}}^{*}[n]
e^{j2\pi\kappa_n\Delta f\tau}
\right|.
$$

在长度为 $P$ 的离散搜索网格上，定义

$$
q_\gamma[p]
=\operatorname{IFFT}_{P}\!\left\{
\frac{
\hat H_{\mathrm{BS}}[n]
\hat H_{\mathrm{UE}}^{*}[n]
}{
\sigma_{\mathrm{BS},n}^2+\sigma_{\mathrm{UE},n}^2
}
\right\},
$$

则未知公共相位下的离散 ML 峰值为

$$
\hat p=\arg\max_p|q_\gamma[p]|.
$$

将循环 IFFT 峰值索引映射为有符号时延 bin：

$$
\hat p_\mathrm{s}
=
\begin{cases}
\hat p,
&\hat p\le \dfrac{P}{2},\\
\hat p-P,
&\hat p>\dfrac{P}{2}.
\end{cases}
$$

令 $Q[p]=|q_\gamma[p]|$。采用峰值及其两个循环相邻点进行三点抛物线插值，分数 bin 修正为

$$
\hat\delta_p
=\frac{1}{2}
\frac{
Q[(\hat p-1)\bmod P]-Q[(\hat p+1)\bmod P]
}{
Q[(\hat p-1)\bmod P]-2Q[\hat p]+Q[(\hat p+1)\bmod P]
}.
$$

于是

$$
\hat\tau_\mathrm{TO}^{\mathrm{BS-UE},\mathrm{ML}}
=\frac{\hat p_\mathrm{s}+\hat\delta_p}{P\Delta f}.
$$

#### 时延幅度谱指标

当上下行信道因参考信号间隔、收发系统响应差异等原因表现出较差的相位一致性时，可以对两端的时延幅度谱进行互相关，以降低相位互异对差分 TO 估计的影响并提高鲁棒性。

令 $P=L_\mathrm{os}N$ 为补零后的 IFFT 长度，其余 $P-N$ 个频域系数取零。直接使用子载波索引 $\kappa_n$，过采样时延响应及其幅度写为

$$
\tilde h_{q,\gamma}[p]
=\frac{1}{P}
\sum_{n=0}^{N-1}
\hat H_{q,\gamma}[\kappa_n]
e^{j2\pi\kappa_n p/P}.
$$

对应的时延幅度谱为

$$
a_{q,\gamma}[p]
=|\tilde h_{q,\gamma}[p]|,
\qquad
p=0,\ldots,P-1,
\qquad
q\in\{\mathrm{BS},\mathrm{UE}\}.
$$

对两个时延幅度谱做循环关联

$$
C_\mathrm{amp}[d]
=\sum_{p=0}^{P-1}
a_{\mathrm{BS},\gamma}[p]
a_{\mathrm{UE},\gamma}^{\vphantom{*}}[(p-d)\bmod P].
$$

令循环相关峰值索引为

$$
\hat d=\arg\max_d C_\mathrm{amp}[d].
$$

将其映射为有符号时延 bin：

$$
\hat d_\mathrm{s}
=
\begin{cases}
\hat d,
&\hat d\le \dfrac{P}{2},\\
\hat d-P,
&\hat d>\dfrac{P}{2}.
\end{cases}
$$

三点抛物线插值得到分数 bin 修正

$$
\hat\delta_d
=\frac{1}{2}
\frac{
C_\mathrm{amp}[(\hat d-1)\bmod P]
-C_\mathrm{amp}[(\hat d+1)\bmod P]
}{
C_\mathrm{amp}[(\hat d-1)\bmod P]
-2C_\mathrm{amp}[\hat d]
+C_\mathrm{amp}[(\hat d+1)\bmod P]
}.
$$

因此

$$
\hat\tau_\mathrm{TO}^{\mathrm{BS-UE},\mathrm{amp}}
=\frac{\hat d_\mathrm{s}+\hat\delta_d}{P\Delta f}.
$$

该指标利用完整多径时延结构而不是只比较两个最大峰。

#### 分离 BS 与 UE 端时偏

由[时延与时偏关系](#时延与时偏关系)可以得到：

$$
\left\{
\begin{aligned}
\tau_\mathrm{TO}^\mathrm{BS}
+\tau_\mathrm{TO}^\mathrm{UE}
&=\tau_c,\\
\tau_\mathrm{TO}^\mathrm{BS}
-\tau_\mathrm{TO}^\mathrm{UE}
&=\tau_\mathrm{TO}^{\mathrm{BS-UE}}.
\end{aligned}
\right.
$$

进而解得

$$
\left\{
\begin{aligned}
\tau_\mathrm{TO}^\mathrm{UE}
&=\frac{\tau_c-\tau_\mathrm{TO}^{\mathrm{BS-UE}}}{2},\\
\tau_\mathrm{TO}^\mathrm{BS}
&=\frac{\tau_c+\tau_\mathrm{TO}^{\mathrm{BS-UE}}}{2}.
\end{aligned}
\right.
$$

对 UE 双站信道去除 TO 时，频域补偿为

$$
\tilde F_{n,m,\gamma}^\mathrm{UE}
=F_{n,m,\gamma}^\mathrm{UE}
e^{j2\pi\kappa_n\Delta f
\hat\tau_\mathrm{TO}^\mathrm{UE}}.
$$

正时延产生负相位斜率，因此补偿使用正指数。

## 双站结果与分辨率

UE 未做感知时偏补偿时，第 $l$ 条下行路径在当前解调边界下的坐标为

$$
\tau_l^\mathrm{UE}
=\tau_{l,\mathrm{prop}}
+\tau_\mathrm{TO}^\mathrm{UE}.
$$

OTA LoS tracking 补偿的是 LoS 观测坐标 $\tau_{\mathrm{LoS,prop}}+\tau_\mathrm{TO}^\mathrm{UE}$，因此输出为相对 LoS 的传播时延

$$
\tilde\tau_{l,\mathrm{OTA}}
=\tau_{l,\mathrm{prop}}
-\tau_{\mathrm{LoS,prop}}.
$$

eRTM 估计并补偿 $\tau_\mathrm{TO}^\mathrm{UE}$，因此输出保留真实传播时延

$$
\tilde\tau_{l,\mathrm{eRTM}}
=\tau_l^\mathrm{UE}
-\hat\tau_\mathrm{TO}^\mathrm{UE}
\approx\tau_{l,\mathrm{prop}}.
$$

补偿后的 $\tilde F_{n,m,\gamma}^\mathrm{UE}$ 沿连续慢时间拼接，并进行杂波抑制、Delay–Doppler 2D FFT 或微多普勒处理。使用连续带宽 $B=N\Delta f$ 时，基本时延分辨率为

$$
\Delta\tau=\frac{1}{B},
$$

对应双站总路径长度差分辨率 $\Delta d_\mathrm{bi}=c/B$。频域等间隔采样形成的循环时延无模糊周期为 $1/\Delta f$，而不产生符号间干扰的有效时延扩展仍应限制在 $T_\mathrm{CP}$ 内。若慢时间间隔为 $T_\mathrm{slow}$、相干处理长度为 $M_s$，则多普勒分辨率为 $1/(M_sT_\mathrm{slow})$，两侧无模糊范围为 $\pm1/(2T_\mathrm{slow})$。


## 使用边界

- **OTA LoS tracking** 要求 LoS 路径持续可见；LoS 消失或参考峰切换会把传播结构变化误当作时钟漂移。
- **eRTM 不需要 LoS 路径**，但需要上下行同时启用、固定 RF 群时延已校准，并且上下行信道具有足够相似的传播结构。
- TDD 在相干时间内最符合互易模型。FDD 的复路径系数不一定互易，只能在主要路径时延仍可对应时近似使用相关指标；若两个载波的可见路径集合或者路径散射系数差异过大，eRTM 算法的可靠性会下降。

## 参考文献

[1] S. Ding et al., “A Synchronization Solution for Bistatic ISAC Under NLOS With Rich Multipaths,” *IEEE Internet of Things Journal*, vol. 13, no. 13, pp. 29185–29199, Jul. 1, 2026, doi: [10.1109/JIOT.2026.3686456](https://doi.org/10.1109/JIOT.2026.3686456).
