---
title: 多通道单站感知
description: BS 侧多通道去调制、杂波抑制、距离–多普勒–角度与微多普勒处理。
---

BS 以已知下行 OFDM 波形照射场景，并在 $R$ 个同步感知通道上接收回波。多通道处理保留同一目标在阵列上的复相位关系，因此结果不仅包含距离和多普勒，还可以在 ULA 假设下估计到达角。

## 1. 多通道时频栅格

定义第 $(n,m,\gamma)$ 个资源上的接收向量

$$
\boldsymbol Y_{n,m,\gamma}^\mathrm{sens}
=
\begin{bmatrix}
Y_{n,m,\gamma}^{(0)}&\cdots&Y_{n,m,\gamma}^{(R-1)}
\end{bmatrix}^{T}.
$$

在最大回波时延不超过循环前缀、单符号内多普勒远小于 $\Delta f$ 时，

$$
\boldsymbol Y_{n,m,\gamma}^\mathrm{sens}
=b_{n,m,\gamma}^\mathrm{DL}
\sum_{p=1}^{P+C}
\beta_p\boldsymbol a(\theta_p)
e^{j2\pi\left(f_{D,s,p}t_{m,\gamma}
-\kappa_n\Delta f\tau_{s,p}^\mathrm{link}\right)}
+\boldsymbol Z_{n,m,\gamma},
$$

其中 $t_{m,\gamma}=(\gamma M+m)T_O$，$\tau_{s,p}^\mathrm{link}=\tau_{s,p}^\mathrm{prop}+\tau_\mathrm{sens}^\mathrm{RF}$。由于 $b_{n,m,\gamma}^\mathrm{DL}$ 已知，可以逐元素去除调制：

$$
\boldsymbol F_{n,m,\gamma}
=\frac{\boldsymbol Y_{n,m,\gamma}^\mathrm{sens}}
{b_{n,m,\gamma}^\mathrm{DL}}
=\sum_{p=1}^{P+C}
\beta_p\boldsymbol a(\theta_p)
e^{j2\pi\left(f_{D,s,p}t_{m,\gamma}
-\kappa_n\Delta f\tau_{s,p}^\mathrm{link}\right)}
+\tilde{\boldsymbol Z}_{n,m,\gamma}.
$$

将所有通道保留下来可形成三阶信道张量，其中 $M_\mathrm{sens}$ 是每帧感知符号数：

$$
\mathcal F_\gamma
\in\mathbb C^{N\times M_\mathrm{sens}\times R}.
$$

## 2. 阵列通道校准

真实多通道观测通常还包含各通道固定的复增益。令

$$
\boldsymbol C=\operatorname{diag}(g_0,g_1,\ldots,g_{R-1})
$$

表示相对于参考通道的幅相响应，则校准后的信道向量为

$$
\boldsymbol F_{n,m,\gamma}^{\mathrm{cal}}
=\boldsymbol C^{-1}\boldsymbol F_{n,m,\gamma}.
$$

后续距离和多普勒处理可以逐通道进行，但角度估计必须使用校准后的跨通道相位。公共复增益只改变整体幅度与相位，不改变角度；通道间的相对相位偏差则会直接形成角度偏差。

## 3. 慢时间序列与杂波抑制

从连续帧中选取等间隔感知符号，并用慢时间索引 $q$ 表示：

$$
\boldsymbol F[n,q]
=\boldsymbol F_{n,m_q,\gamma_q}^{\mathrm{cal}},
\qquad
t_q=qT_\mathrm{slow}.
$$

若每隔 $M_D$ 个 OFDM 符号取一次，则 $T_\mathrm{slow}=M_DT_O$。TDD 中只选择下行有效资源；用于普通 Doppler FFT 的样本应保持等间隔，否则应使用实际 $t_q$ 进行非均匀谱估计。

静态与近静态反射集中在零多普勒附近。对每个子载波和阵列通道沿 $q$ 方向施加高通 MTI：

$$
\tilde{\boldsymbol F}[n,q]
=\frac{1}{a_0}
\left(
\sum_{i=0}^{I}b_i\boldsymbol F[n,q-i]
-\sum_{j=1}^{J}a_j\tilde{\boldsymbol F}[n,q-j]
\right).
$$

$\{b_i\}$ 与 $\{a_j\}$ 分别是前向和反馈系数。该滤波在零多普勒附近形成阻带，抑制固定泄漏和静态杂波，同时保留阻带外的运动目标。若目标本身接近零速度，则应缩窄阻带或直接使用未做 MTI 的信道张量。

## 4. 距离–多普勒处理

将连续慢时间流分成长度为 $M_s$ 的相干处理区间。令 $N_\mathrm{Per}$ 和 $M_\mathrm{Per}$ 分别为时延 IFFT 与 Doppler FFT 的长度。对二维窗 $w[n,q]$，每个距离–多普勒单元的阵列复向量为

$$
\boldsymbol z_\gamma[k_\tau,k_f]
=\sum_{q=0}^{M_s-1}\sum_{n=0}^{N-1}
\tilde{\boldsymbol F}_\gamma[n,q]w[n,q]
e^{j2\pi\kappa_n k_\tau/N_\mathrm{Per}}
e^{-j2\pi qk_f/M_\mathrm{Per}}.
$$

不区分角度时，可使用非相干阵列功率

$$
P_\mathrm{RD}[k_\tau,k_f]
=\frac{1}{NM_s}
\left\|\boldsymbol z_\gamma[k_\tau,k_f]\right\|_2^2.
$$

经 FFT 移位后的 Doppler 索引按有符号 $k_f$ 解释：

$$
\hat\tau^\mathrm{link}=\frac{\hat k_\tau}{N_\mathrm{Per}\Delta f},
\qquad
\hat f_D=\frac{\hat k_f}{M_\mathrm{Per}T_\mathrm{slow}},
$$

$$
\hat\tau^\mathrm{prop}
=\hat\tau^\mathrm{link}-\tau_\mathrm{sens}^\mathrm{RF},
\qquad
\hat r=\frac{c\hat\tau^\mathrm{prop}}{2},
\qquad
\hat v=\frac{c\hat f_D}{2f_c}.
$$

零填充增加显示采样密度，但基本距离分辨率仍由 $B$ 决定，多普勒分辨率仍由相干观测时长 $M_sT_\mathrm{slow}$ 决定。

## 分辨率与无模糊范围

使用连续带宽 $B=N\Delta f$ 时，基本时延分辨率为 $\Delta\tau=1/B$，对应单站距离分辨率

$$
\Delta r_\mathrm{mono}=\frac{c}{2B}.
$$

等间隔子载波形成的循环时延无模糊周期为 $1/\Delta f$，对应单站距离周期 $c/(2\Delta f)$；实际无符号间干扰的回波时延仍应落在循环前缀 $T_\mathrm{CP}$ 内。慢时间采样间隔为 $T_\mathrm{slow}$、相干处理长度为 $M_s$ 时，多普勒分辨率和两侧无模糊范围分别为

$$
\Delta f_D=\frac{1}{M_sT_\mathrm{slow}},
\qquad
|f_D|<\frac{1}{2T_\mathrm{slow}}.
$$

对应速度分辨率为 $c\Delta f_D/(2f_c)$。零填充只增加显示采样密度，不改变这些基本分辨率或无模糊范围。

## 5. ULA 角度处理

对固定的 $(k_\tau,k_f)$ 单元，用 ULA 导向矢量扫描角度：

$$
P_\mathrm{RDA}[k_\tau,k_f,\theta]
=\frac{
\left|\boldsymbol a^H(\theta)
\boldsymbol z_\gamma[k_\tau,k_f]\right|^2
}{R^2NM_s}.
$$

其峰值给出

$$
\hat\theta
=\arg\max_{\theta}P_\mathrm{RDA}[k_\tau,k_f,\theta].
$$

对单个主导目标，校准后的阵列相位近似满足

$$
\angle z_r\approx\phi_0+r\mu,
\qquad
\mu=\frac{2\pi d_a}{\lambda}\sin\theta.
$$

因此也可先对通道相位解缠并拟合斜率 $\hat\mu$，再计算

$$
\hat\theta
=\arcsin\!\left(\frac{\lambda\hat\mu}{2\pi d_a}\right).
$$

$d_a\le\lambda/2$ 可避免可见角域内的空间混叠。多个目标落入同一距离–多普勒单元时，简单相位斜率不再对应单一角度，应改用波束扫描、空间 FFT 或更高分辨率的阵列估计方法。

## 6. 多通道微多普勒

先对频率轴做 IFFT，得到各通道的距离–慢时间序列

$$
\boldsymbol r[k_\tau,q]
=\frac{1}{N}\sum_{n=0}^{N-1}
\tilde{\boldsymbol F}[n,q]e^{j2\pi nk_\tau/N}.
$$

选择目标距离单元 $k_\tau^\star$ 后，可以直接对各通道功率合成，也可以先对关注角度 $\theta_0$ 波束形成：

$$
r_{\theta_0}[q]
=\frac{1}{R}\boldsymbol a^H(\theta_0)
\boldsymbol r[k_\tau^\star,q].
$$

长度为 $M_w$、步长为 $M_H$ 的短时傅里叶变换为

$$
G[u,k_f]
=\sum_{\ell=0}^{M_w-1}
r_{\theta_0}[uM_H+\ell]
w_\mathrm{md}[\ell]
e^{-j2\pi k_f\ell/M_\mathrm{md}}.
$$

$|G[u,k_f]|^2$ 描述目标细微运动随时间的 Doppler 展开；先做角度波束形成可以抑制同一距离单元内来自其他方向的干扰。
