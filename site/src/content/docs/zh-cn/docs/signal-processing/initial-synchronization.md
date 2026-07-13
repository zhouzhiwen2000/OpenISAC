---
title: 初始同步
description: UE 利用下行 ZC、可选第二同步符号与 CFO 训练字段完成初始时偏和频偏估计及校正。
---

初始同步以 BS 为时间和频率参考，由 UE 利用下行链路完成。它负责在 `SYNC_SEARCH` 状态中找到完整帧边界、估计初始时偏和频偏并施加校正；进入 `NORMAL` 状态后，下行和上行再分别利用各自的信道参考与导频估计残余 CFO/SFO。上行 ZC 因此用于 BS 侧信道估计和残余对齐，而不是建立一套与下行无关的初始同步过程。

令 $s_\mathrm{ZC}[k]$ 为包含循环前缀、长度为 $N_s$ 的主 ZC 时域符号，$m_\mathrm{sync}$ 为其帧内位置。OpenISAC 支持紧凑的单 ZC 捕获，以及由第二同步符号和 CFO 训练字段增强的捕获路径。

## 1. 主 ZC 定时检测

UE 对候选起点 $u$ 计算归一化相关能量

$$
\Lambda_\mathrm{ZC}[u]
=\frac{
\left|\sum_{k=0}^{N_s-1}y_\mathrm{UE}[u+k]s_\mathrm{ZC}^{*}[k]\right|^2
}{
\left(\sum_{k=0}^{N_s-1}|y_\mathrm{UE}[u+k]|^2\right)
\left(\sum_{k=0}^{N_s-1}|s_\mathrm{ZC}[k]|^2\right)
}.
$$

记搜索区为 $\mathcal U$，峰值位置为 $\hat k_\mathrm{peak}=\arg\max_{u\in\mathcal U}\Lambda_\mathrm{ZC}[u]$。实现以峰均比

$$
\rho_\mathrm{ZC}
=\frac{\Lambda_\mathrm{ZC}[\hat k_\mathrm{peak}]}
{\frac{1}{|\mathcal U|}\sum_{u\in\mathcal U}\Lambda_\mathrm{ZC}[u]}
$$

判断检测是否可靠。检测成立后，帧起点的整数时偏估计为

$$
\hat k_\mathrm{TO}
=\hat k_\mathrm{peak}-m_\mathrm{sync}N_s-N_\mathrm{lag}.
$$

$N_\mathrm{lag}<N_\mathrm{CP}$ 在最强相关峰之前保留多径余量，避免较早到达的路径在校正后越出循环前缀。单 ZC 模式训练开销最低，但较大的初始 CFO 会在相关窗口内产生明显相位旋转并削弱 ZC 峰值，因此还可启用下述增强字段。

## 2. 可选第二同步符号

启用第二同步符号时，$m_\mathrm{sync}-1$ 与 $m_\mathrm{sync}$ 连续发送相同的 ZC OFDM 符号。对试探起点 $u$，UE 计算两个有用部分之间的 Schmidl–Cox 型度量

$$
P_\mathrm{SC}[u]
=\sum_{k=0}^{N-1}
y_\mathrm{UE}^{*}[u+N_\mathrm{CP}+k]
y_\mathrm{UE}[u+N_s+N_\mathrm{CP}+k],
$$

$$
R_\mathrm{SC}[u]
=\sum_{k=0}^{N-1}
\left|y_\mathrm{UE}[u+N_s+N_\mathrm{CP}+k]\right|^2,
\qquad
\Lambda_\mathrm{SC}[u]
=\frac{|P_\mathrm{SC}[u]|^2}{R_\mathrm{SC}^2[u]}.
$$

由此得到粗起点和模糊 CFO：

$$
\hat u=\arg\max_u\Lambda_\mathrm{SC}[u],
\qquad
\hat f_{o,\mathrm{mod}}
=\frac{\angle P_\mathrm{SC}[\hat u]}{2\pi N_sT_s}.
$$

该相位估计的无模糊范围为 $\pm1/(2N_sT_s)$。在配置的 CFO 搜索范围内，候选频偏为

$$
f_a=\hat f_{o,\mathrm{mod}}+\frac{a}{N_sT_s},
\qquad a\in\mathbb Z.
$$

UE 分别用 $f_a$ 对接收样本去旋转，并在预期主 ZC 附近计算局部 ZC 相关；若未启用 CFO 训练字段，选择相关峰最强的候选。最终仍由主 ZC 按上一节公式细化时偏，第二同步符号不替代主 ZC 的精细定时作用。

## 3. 可选 CFO 训练字段

CFO 训练字段位于 $m_\mathrm{sync}+1$，其有用部分满足

$$
s_\mathrm{CFO}[k+N_\mathrm{CFO}]
=s_\mathrm{CFO}[k].
$$

令 $u_\mathrm{CFO}$ 为该字段起点，则独立 CFO 参考为

$$
\hat f_\mathrm{CFO,tr}
=\frac{1}{2\pi N_\mathrm{CFO}T_s}
\angle\!\left(
\sum_{k=0}^{N-N_\mathrm{CFO}-1}
y_\mathrm{UE}^{*}[u_\mathrm{CFO}+N_\mathrm{CP}+k]
y_\mathrm{UE}[u_\mathrm{CFO}+N_\mathrm{CP}+k+N_\mathrm{CFO}]
\right).
$$

其无模糊范围为 $\pm1/(2N_\mathrm{CFO}T_s)$。该字段只用于从候选集合中选择与 $\hat f_\mathrm{CFO,tr}$ 最接近的 CFO，不取代主 ZC 定时，也不取代随后基于循环前缀尾部的频偏细化。第二同步符号和 CFO 训练字段可以分别启用：前者提高大 CFO 下的粗捕获能力，后者提高 CFO 模糊消除的可靠性。

## 4. CP-tail 频偏估计与 CFO 模糊消除

未使用或未检测到第二同步符号时，UE 先由主 ZC 找到帧位置，再对当前块内完整 OFDM 符号的循环前缀和有用部分尾部进行相干累积。若 $\mathcal M_\mathrm{CP}$ 为可用符号集合，则

$$
r_\mathrm{CP}
=\sum_{m\in\mathcal M_\mathrm{CP}}
\sum_{i=0}^{N_\mathrm{CP}-1}
y_\mathrm{UE}^{*}[u_m+i]
y_\mathrm{UE}[u_m+N+i],
$$

$$
\hat f_{o,\mathrm{CP,mod}}
=\frac{\angle r_\mathrm{CP}}{2\pi NT_s}.
$$

$\hat f_{o,\mathrm{CP,mod}}$ 只给出 CFO 对 $\Delta f=1/(NT_s)$ 取模后的结果，不能单独确定 CFO。若配置的 CFO 搜索范围为 $\mathcal F_\mathrm{search}$，则候选 CFO 集合为

$$
\mathcal F_\mathrm{CFO}
=\left\{
\hat f_{o,\mathrm{CP,mod}}+a\Delta f
\;\middle|\;
a\in\mathbb Z,
\ \hat f_{o,\mathrm{CP,mod}}+a\Delta f
\in\mathcal F_\mathrm{search}
\right\}.
$$

UE 分别用每个候选 CFO 对接收样本去旋转，再在主 ZC 的预期位置附近计算局部相关：未启用 CFO 训练字段时，选择 ZC 相关峰最强的候选 CFO；启用 CFO 训练字段时，选择与 $\hat f_\mathrm{CFO,tr}$ 最接近的候选 CFO。完成模糊消除后，所选值就是初始 CFO 估计。

使用第二同步符号时，模糊 CFO 由相邻 ZC 的相位差得到，候选 CFO 间隔为 $1/(N_sT_s)$；UE 采用同样的局部 ZC 相关或 CFO 训练字段选择其中一个候选 CFO，再用 CP-tail 结果估计该值附近的残余频偏。因此，第二同步符号和仅使用主 ZC 时生成候选 CFO 的方式不同，但也通过 CP-tail 相关提高频偏估计精度。

## 5. 校正与状态切换

确定 $\hat k_\mathrm{TO}$ 后，UE 通过调整下一接收块的取样和丢弃位置更新下行解调/FFT 窗口，使主导路径落在配置的目标峰位置。更新后的窗口直接作为 $\tau_d^\mathrm{UE}$ 的当前参考。确定初始 CFO 后，UE 通过数字频率校正或参考时钟调节去除相应频偏。时偏和频偏校正成功后，接收机进入 `NORMAL` 状态，并把后续信道估计、残余 CFO/SFO 跟踪和补偿交给[下行通信](/zh-cn/docs/signal-processing/ue-reception/)处理。若检测置信度下降或残差超出可靠范围，则返回 `SYNC_SEARCH` 重新捕获。
