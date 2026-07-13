---
title: 信号模型
description: 双向 OFDM 通信、多通道 ULA 单站感知和 UE 双站感知的统一模型。
---

当前 OpenISAC 系统包含一条 BS→UE 下行、一条 UE→BS 上行、$R$ 个 BS 单站感知接收通道，以及一个 UE 双站感知观测。BS 的下行波形同时承担通信和照射信号；UE 接收同一波形并同时用于通信解调与双站感知。

## 双向通信信道

令 $x\in\{\mathrm{DL},\mathrm{UL}\}$ 表示下行或上行。链路 $x$ 的时变多径信道写为

$$
h_x(t,\tau)=
\sum_{l=1}^{L_x}
\alpha_l^x
\delta\!\left(\tau-\tau_l^x-\tau_d^x(t)\right)
e^{j2\pi(f_{D,l}^x+\Delta f_c^x)t}.
$$

其中，$L_x$ 是可分辨路径数，$\alpha_l^x$、$\tau_l^x$ 和 $f_{D,l}^x$ 分别是第 $l$ 条路径的复系数、传播时延和多普勒频移；$\tau_d^x(t)$ 是两端参考时间不一致造成的链路定时偏移，$\Delta f_c^x$ 是载波频偏。

下行与上行分别满足

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

这里不要求 $h_\mathrm{DL}=h_\mathrm{UL}$。TDD 在信道相干时间内可以近似利用传播互易性，但两端定时、频率偏移和收发响应仍需分别估计；FDD 的两个载波则直接视为两条独立频率响应。

## BS 多通道单站信道

将 $R$ 个 BS 感知通道视为均匀线阵（ULA）。阵元间距为 $d_a$，下行载波波长为 $\lambda=c/f_c$，入射角 $\theta$ 从阵列法向（broadside）计量。阵列导向矢量定义为

$$
\boldsymbol a(\theta)=
\begin{bmatrix}
1 & e^{j\mu(\theta)} & \cdots & e^{j(R-1)\mu(\theta)}
\end{bmatrix}^{T},
\qquad
\mu(\theta)=\frac{2\pi d_a}{\lambda}\sin\theta.
$$

设场景中共有 $Q=P+C$ 个可分辨反射分量，其中 $P$ 个为运动目标、$C$ 个为静态或近静态杂波。阵列信道为

$$
\boldsymbol h_\mathrm{BS}^{\mathrm{sens}}(t,\tau)
=\sum_{p=1}^{Q}
\beta_p\boldsymbol a(\theta_p)
\delta(\tau-\tau_{s,p})
e^{j2\pi f_{D,s,p}t}.
$$

因此多通道接收向量为

$$
\boldsymbol y_\mathrm{BS}^{\mathrm{sens}}(t)
=\int
\boldsymbol h_\mathrm{BS}^{\mathrm{sens}}(t,\tau)
s_\mathrm{DL}(t-\tau)\,d\tau
+\boldsymbol z_\mathrm{BS}^{\mathrm{sens}}(t),
$$

其中 $\boldsymbol z_\mathrm{BS}^{\mathrm{sens}}(t)\sim\mathcal{CN}(\boldsymbol 0,\sigma^2\boldsymbol I_R)$。对距离 $r_p$、径向速度 $v_p$ 和雷达散射截面 $\sigma_{\mathrm{RCS},p}$ 的点目标，窄带远场模型给出

$$
\beta_p=
\sqrt{\frac{c^2\sigma_{\mathrm{RCS},p}}
{(4\pi)^3r_p^4f_c^2}}e^{j\phi_p},
\qquad
\tau_{s,p}=\frac{2r_p}{c},
\qquad
f_{D,s,p}=\frac{2v_pf_c}{c}.
$$

这里约定 $v_p>0$ 表示目标沿径向接近。单站距离和速度分别由 $r=c\tau/2$ 与 $v=cf_D/(2f_c)$ 得到。

## UE 双站几何

对经过第 $l$ 个散射体的 BS→UE 路径，设 BS 到散射体、散射体到 UE、BS 到 UE 的距离分别为 $d_{B,l}$、$d_{l,U}$ 与 $d_{B,U}$，则

$$
\tau_l^\mathrm{bi}=\frac{d_{B,l}+d_{l,U}}{c},
\qquad
\tau_\mathrm{LoS}=\frac{d_{B,U}}{c}.
$$

以直达径为定时参考后，双站感知观测的是超额时延

$$
\Delta\tau_l^\mathrm{bi}
=\tau_l^\mathrm{bi}-\tau_\mathrm{LoS},
\qquad
\Delta d_l^\mathrm{bi}=c\Delta\tau_l^\mathrm{bi}.
$$

因此一个固定双站时延对应以 BS 和 UE 为两个焦点的椭圆，而不是单站模型中的圆。低多普勒分量通常归入杂波，动态多径则作为双站散射体。

## 采样时钟偏差

名义采样间隔为 $T_s=1/B$。若接收端的实际采样间隔为 $T_{s,q}=T_s-\Delta T_{s,q}$，则第 $l$ 条通信路径的离散样本可近似写为

$$
y_{q,l}[k]\approx
\alpha_l
s\!\left(kT_s-\tau_l-\tau_d[0]-k\Delta T_{s,q}\right)
e^{j2\pi(f_{D,l}+\Delta f_c)kT_s}.
$$

该式忽略了 $(f_{D,l}+\Delta f_c)\Delta T_{s,q}$ 的二阶小量。它揭示了三种可分离的影响：固定定时偏移移动帧起点，CFO 产生随时间累积的公共相位，SFO 则同时造成随时间增长的时偏和与子载波频率相关的相位斜率。
