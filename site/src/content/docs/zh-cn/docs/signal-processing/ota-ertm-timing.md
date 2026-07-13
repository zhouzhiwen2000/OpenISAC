---
title: OTA 与 eRTM 时序
description: OTA 时序补偿、eRTM 时序项和 MTI suppression ratio 解释。
---

OTA 时序工作流用于估计和校正影响双站感知的相对时序项。这里区分两个相关层次：

- 论文层面的 OTA 感知补偿，用于保持 UE 侧感知时延连续。
- 运行时 eRTM 时序变量，用于暴露实现中的语义化 timing-offset 项。

## eRTM 时序项

eRTM 时序模型将配置中的 RF 延迟与运行时实时测得的时序值分开：

$$
\tau_c
=
\tau_\mathrm{DL,RF}+\tau_\mathrm{UL,RF}-t_\mathrm{DL-UL,BS}-t_\mathrm{TA,UE}
$$

双站时序关系可解释为：

$$
\tau_\mathrm{TO,UE}
=
\frac{\tau_c-\tau_\mathrm{TO,BS-UE}}{2}
$$

$$
\tau_\mathrm{TO,BS}
=
\frac{\tau_c+\tau_\mathrm{TO,BS-UE}}{2}
$$

实现上，RF 链路项应来自 YAML，而 DUTI/TADV 类运行时项应来自实时控制或时序状态。原始显示值、语义 timing-offset 变量和实际 correction 值应保持分离。

## 为什么 OTA 补偿会改善 MTI

在静态双站场景中，残余时序漂移会导致静态 clutter 无法被干净抵消。OpenISAC 使用 MTI suppression ratio（MSR）作为稳定性代理指标：

$$
\mathrm{MSR}
=
\frac{
\sum_{m=m_\mathrm{start}}^{m_\mathrm{start}+M_\mathrm{avg}}
\sum_{n=0}^{N-1}
|(\grave{\boldsymbol{F}}_\mathrm{UE})_{n,m}|^2
}{
\sum_{m=m_\mathrm{start}}^{m_\mathrm{start}+M_\mathrm{avg}}
\sum_{n=0}^{N-1}
|(\tilde{\boldsymbol{F}}_\mathrm{UE})_{n,m}|^2
}
$$

其中 $\grave{\boldsymbol{F}}_\mathrm{UE}$ 是 MTI 前的双站 TF-domain stream，$\tilde{\boldsymbol{F}}_\mathrm{UE}$ 是 MTI 后的 stream。

考虑单个静态分量且 $f_{D,1}=0$，MTI 前样本可近似为：

$$
(\grave{\boldsymbol{F}}_\mathrm{UE})_{n,m}
\approx
\alpha_1
e^{-j2\pi n\Delta f(\tau_1+\bar{\tau}_{d,\gamma,mN_s})}
$$

直观起见，若使用 two-pulse MTI，则：

$$
\begin{aligned}
(\tilde{\boldsymbol{F}}_\mathrm{UE})_{n,m}
&\approx
\alpha_1 e^{-j2\pi n\Delta f\tau_1}
\\
&\quad\cdot
\left(
e^{-j2\pi n\Delta f\bar{\tau}_{d,\gamma,mN_s}}
-
e^{-j2\pi n\Delta f\bar{\tau}_{d,\gamma,(m-1)N_s}}
\right)
\end{aligned}
$$

取平方幅度后公共相位被抵消：

$$
|(\tilde{\boldsymbol{F}}_\mathrm{UE})_{n,m}|^2
=
4|\alpha_1|^2
\sin^2\!\left(
\pi n\Delta f
(\bar{\tau}_{d,\gamma,mN_s}
-
\bar{\tau}_{d,\gamma,(m-1)N_s})
\right)
$$

因此，更大的帧间时序波动会增加 MTI 后残余能量，降低可达到的 MSR。实际运行时使用的是 IIR MTI 而不是 two-pulse canceller，但趋势相同。

## 实测稳定性影响

![OTA 同步带来的 MSR 改善](/images/OpenISAC_MSRImprovement.png)

论文实验在两个中心频率 $f_c\in\{2.4,3.1\}$ GHz 和两个带宽 $B\in\{50,100\}$ MHz 下测量 MSR 改善。零时钟误差附近提升接近 0 dB，$\pm0.25$ ppm 附近约 10--14 dB，$\pm0.5$ ppm 附近约 17--22 dB。

操作层面的含义很直接：当 UE 时钟本来就对齐良好时，OTA 补偿可修正的内容很少；随着时钟误差增大，连续时序补偿可以阻止静态场景的时延漂移泄漏到 MTI 输出中。
