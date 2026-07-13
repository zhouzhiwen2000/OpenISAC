---
title: 信号模型
description: 从传播、RF 群时延和本地解调边界出发，统一描述下行、上行与感知信道。
---

当前 OpenISAC 系统包含三条链路：BS→UE 下行通信链路、UE→BS 上行通信链路，以及 BS 单站感知链路。BS 的下行波形同时承担通信信号和单站感知照射信号；UE 对同一下行波形的接收既用于通信解调，也构成双站感知观测。

## 双向通信的时延组成

令 $x\in\{\mathrm{DL},\mathrm{UL}\}$ 表示下行或上行，并令 $q_x$ 表示链路 $x$ 的接收端，即 $q_\mathrm{DL}=\mathrm{UE}$、$q_\mathrm{UL}=\mathrm{BS}$。第 $l$ 条路径的时延分为三个层次。

真实无线传播时延记为

$$
\tau_{l,\mathrm{prop}}^x(t),
$$

它只描述电磁波在空间和散射环境中的传播。链路 $x$ 的固定 RF 群时延记为 $\tau_x^\mathrm{RF}$，包含对应发射链和接收链的合成延迟。因此物理总链路时延为

$$
\tau_{l,\mathrm{link}}^x(t)
=\tau_{l,\mathrm{prop}}^x(t)+\tau_x^\mathrm{RF}.
$$

接收机以当前本地解调窗口切分 OFDM 符号。令 $\tau_d^{q_x}(t)$ 表示接收端 $q_x$ 当前解调窗口相对于链路 $x$ 发射端帧边界的时变偏移，即解调窗口起始时刻减去发射端帧边界时刻；正值表示解调窗口位于发射端帧边界之后。采样频率偏差会使解调窗口缓慢偏移，且初始同步或后续整数采样校正会直接更新当前解调窗口，因此该偏移是时变的。

定义本地延迟轴上的 TO 为真实传播时延到本地观测路径时延的公共偏移。对下行和上行分别有

$$
\tau_\mathrm{TO}^\mathrm{UE}(t)
\triangleq\tau_\mathrm{DL}^\mathrm{RF}-\tau_d^\mathrm{UE}(t),
\qquad
\tau_\mathrm{TO}^\mathrm{BS}(t)
\triangleq\tau_\mathrm{UL}^\mathrm{RF}-\tau_d^\mathrm{BS}(t).
$$

因此两端在自己的本地延迟轴上观测到的第 $l$ 条路径时延分别为

$$
\tau_l^\mathrm{UE}(t)
=\tau_{l,\mathrm{prop}}(t)
+\tau_\mathrm{TO}^\mathrm{UE}(t),
$$

$$
\tau_l^\mathrm{BS}(t)
=\tau_{l,\mathrm{prop}}(t)
+\tau_\mathrm{TO}^\mathrm{BS}(t).
$$

这里的 $\tau_l^\mathrm{DL}$ 与 $\tau_l^\mathrm{UL}$ 是包含 RF 群时延的物理总链路时延，$\tau_l^\mathrm{UE}$ 与 $\tau_l^\mathrm{BS}$ 则是两个接收机在本地延迟轴上的直接观测。后者与真实传播时延之差即为本文的 TO。

## 连续时间接收模型

接收端 $q_x$ 观察到的等效时变基带冲激响应写为

$$
h_x(t,\tau)=
\sum_{l=0}^{L_x-1}
\alpha_l^x(t)
e^{j2\pi(f_{D,l}^x+\Delta f_c^x)t}
\delta\!\left(
\tau-\tau_{l,\mathrm{prop}}^x(t)
-\tau_x^\mathrm{RF}
+\tau_d^{q_x}(t)
\right).
$$

其中，$L_x$ 是可分辨路径数；$\alpha_l^x(t)$、$\tau_{l,\mathrm{prop}}^x(t)$ 和 $f_{D,l}^x$ 分别是第 $l$ 条路径的复散射系数、真实传播时延和多普勒频移；$\Delta f_c^x$ 是链路残余载波频偏。下行与上行接收信号分别满足

$$
y_\mathrm{UE}^\mathrm{DL}(t)
=\int h_\mathrm{DL}(t,\tau)s_\mathrm{DL}(t-\tau)\,d\tau
+z_\mathrm{UE}(t),
$$

$$
y_\mathrm{BS}^\mathrm{UL}(t)
=\int h_\mathrm{UL}(t,\tau)s_\mathrm{UL}(t-\tau)\,d\tau
+z_\mathrm{BS}^\mathrm{UL}(t).
$$

## TDD 上下行信道关系

令 $t_{\mathrm{DL}}$ 和 $t_{\mathrm{UL}}$ 分别表示参与 eRTM 的下行与上行参考 OFDM 符号时刻。TDD 中若二者间隔远小于信道相干时间，则

$$
\tau_{l,\mathrm{prop}}(t_{\mathrm{DL}})
\approx
\tau_{l,\mathrm{prop}}(t_{\mathrm{UL}})
\approx
\tau_{l,\mathrm{prop}},
$$

$$
\alpha_l^\mathrm{DL}(t_{\mathrm{DL}})
\approx
\alpha_l^\mathrm{UL}(t_{\mathrm{UL}})
\approx
\alpha_l,
$$

TO 主要随两端时钟缓慢漂移，因此对于时间上足够接近的下行与上行参考符号，同一测量对还满足

$$
\tau_\mathrm{TO}^\mathrm{UE}(t_{\mathrm{DL}})
\approx\tau_\mathrm{TO}^\mathrm{UE},
\qquad
\tau_\mathrm{TO}^\mathrm{BS}(t_{\mathrm{UL}})
\approx\tau_\mathrm{TO}^\mathrm{BS}.
$$

这里将同一测量对中两个相近时刻的 TO 分别简记为 $\tau_\mathrm{TO}^\mathrm{UE}$ 和 $\tau_\mathrm{TO}^\mathrm{BS}$，并不假设二者相等。UE 端观测到的下行频域信道和 BS 端观测到的上行频域信道分别为

$$
H_{\mathrm{UE}}[n]
=\sum_{l=0}^{L-1}\alpha_l
e^{-j2\pi\kappa_n\Delta f[
\tau_{l,\mathrm{prop}}
+\tau_\mathrm{TO}^\mathrm{UE}]},
$$

$$
H_{\mathrm{BS}}[n]
=\sum_{l=0}^{L-1}\alpha_l
e^{-j2\pi\kappa_n\Delta f[
\tau_{l,\mathrm{prop}}
+\tau_\mathrm{TO}^\mathrm{BS}]}.
$$

定义两端 TO 之差

$$
\tau_\mathrm{TO}^{\mathrm{BS-UE}}
\triangleq
\tau_\mathrm{TO}^\mathrm{BS}
-\tau_\mathrm{TO}^\mathrm{UE}
=\tau_l^\mathrm{BS}-\tau_l^\mathrm{UE},
$$

则有

$$
\boxed{
H_{\mathrm{BS}}[n]
\approx H_{\mathrm{UE}}[n]
e^{-j2\pi\kappa_n\Delta f
\tau_\mathrm{TO}^{\mathrm{BS-UE}}}
}.
$$

## BS 多通道单站信道

假设 BS 端配备 $R$ 阵元的均匀线阵（ULA）。阵元间距为 $d_a$，下行载波波长为 $\lambda=c/f_c$，入射角 $\theta$ 从阵列法向（broadside）计量。阵列导向矢量定义为

$$
\boldsymbol a(\theta)=
\begin{bmatrix}
1 & e^{j\mu(\theta)} & \cdots & e^{j(R-1)\mu(\theta)}
\end{bmatrix}^{T},
\qquad
\mu(\theta)=\frac{2\pi d_a}{\lambda}\sin\theta.
$$

设场景中共有 $Q=P+C$ 个可分辨反射分量，其中 $P$ 个为运动目标、$C$ 个为静态或近静态杂波。令 $\tau_{s,p}^\mathrm{prop}$ 为第 $p$ 个分量的真实往返传播时延，$\tau_\mathrm{sens}^\mathrm{RF}$ 为单站收发链的固定群时延，则

$$
\tau_{s,p}^\mathrm{link}
=\tau_{s,p}^\mathrm{prop}+\tau_\mathrm{sens}^\mathrm{RF},
$$

$$
\boldsymbol h_\mathrm{BS}^\mathrm{sens}(t,\tau)
=\sum_{p=1}^{Q}
\beta_p\boldsymbol a(\theta_p)
\delta(\tau-\tau_{s,p}^\mathrm{link})
e^{j2\pi f_{D,s,p}t}.
$$

因此多通道接收向量为

$$
\boldsymbol y_\mathrm{BS}^\mathrm{sens}(t)
=\int
\boldsymbol h_\mathrm{BS}^\mathrm{sens}(t,\tau)
s_\mathrm{DL}(t-\tau)\,d\tau
+\boldsymbol z_\mathrm{BS}^\mathrm{sens}(t),
$$

其中 $\boldsymbol z_\mathrm{BS}^\mathrm{sens}(t)\sim\mathcal{CN}(\boldsymbol 0,\sigma^2\boldsymbol I_R)$。对距离 $r_p$、径向速度 $v_p$ 和雷达散射截面 $\sigma_{\mathrm{RCS},p}$ 的点目标，窄带远场模型给出

$$
\beta_p=
\sqrt{\frac{c^2\sigma_{\mathrm{RCS},p}}
{(4\pi)^3r_p^4f_c^2}}e^{j\phi_p},
\qquad
\tau_{s,p}^\mathrm{prop}=\frac{2r_p}{c},
\qquad
f_{D,s,p}=\frac{2v_pf_c}{c}.
$$

校准并去除 $\tau_\mathrm{sens}^\mathrm{RF}$ 后，单站距离和速度分别由 $r=c\tau^\mathrm{prop}/2$ 与 $v=cf_D/(2f_c)$ 得到；这里约定 $v_p>0$ 表示目标沿径向接近。

## 采样时钟偏差

名义采样间隔为 $T_s=1/B$。若接收端 $q_x$ 的实际采样间隔为 $T_{s,q_x}=T_s-\Delta T_{s,q_x}$，则第 $l$ 条通信路径的离散样本可近似写为

$$
y_{q_x,l}[k]\approx
\alpha_l^x
s_x\!\left(
kT_s
-\tau_{l,\mathrm{prop}}^x
-\tau_x^\mathrm{RF}
+\tau_d^{q_x}[0]
-k\Delta T_{s,q_x}
\right)
e^{j2\pi(f_{D,l}^x+\Delta f_c^x)kT_s}.
$$

该式忽略了 $(f_{D,l}^x+\Delta f_c^x)\Delta T_{s,q_x}$ 的二阶小量。固定 RF 群时延形成 TO 的静态部分，当前解调窗口相对于发射端帧边界的偏移形成其时变部分，并以负号计入 TO。CFO 产生随时间累积的公共相位，SFO 则使 TO 随时间缓慢漂移并形成随子载波变化的相位斜率。
