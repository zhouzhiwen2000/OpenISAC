---
title: UE 双站感知
description: 下行符号重构、连续时偏与 SFO 补偿，以及双站时延–多普勒解释。
---

UE 双站感知与下行通信观察同一个 $h_\mathrm{DL}(t,\tau)$，但目标不同：通信希望消除信道，感知则希望保留信道随时延和慢时间的结构。相对于 BS 单站感知，UE 还需解决两点：数据符号事先未知，以及 BS–UE 定时漂移会直接移动感知时延轴。

## 1. 下行符号重构

ZC、导频和全带宽信道参考都是已知符号。对数据资源，UE 从均衡后的 QPSK 符号做硬判决：

$$
\tilde b_{n,m,\gamma}^\mathrm{DL}
=\frac{1}{\sqrt2}
\left[
\operatorname{sgn}(\operatorname{Re}\{\hat d_{n,m,\gamma}^\mathrm{DL}\})
+j\operatorname{sgn}(\operatorname{Im}\{\hat d_{n,m,\gamma}^\mathrm{DL}\})
\right].
$$

统一的重构栅格为

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

通信只要求总时延扩展留在循环前缀内。亚采样级时偏可被吸收到 $\hat H_{n,m,\gamma}^\mathrm{DL}$ 的相位中，只有累计漂移接近一个采样时才需要移动帧起点。

双站感知中，时延本身就是测量量。若直接沿用这些离散的整数采样修正，目标时延轨迹会呈阶梯状，并在时延–多普勒或微多普勒结果中产生人工跳变。因此感知链路需要重建连续时偏，而不是只记录通信帧起点的变化。

## 3. 分数时偏估计

由同步 ZC 的信道估计计算时延谱

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

按两个候选实部的符号一致性选择 $\hat\delta_{\tau,\gamma}$ 后，当前总时偏估计为

$$
\hat k_{\tau,\gamma}
=k_{\max,\gamma}+\hat\delta_{\tau,\gamma},
\qquad
\hat\tau_{o,\gamma}=\frac{\hat k_{\tau,\gamma}}{B}.
$$

这一估计避免了高倍频域零填充，但单帧结果仍受噪声和多径峰形影响，因而还需利用 SFO 的长期结构进行平滑。

## 4. SFO 窗口拟合

在从 $\gamma_w$ 开始、长度为 $\Gamma_W$ 的窗口内，令 $\hat k_\mathrm{TO,\gamma}$ 表示通信链路已经施加的整数定时修正。窗口内累计修正为

$$
A_{\gamma_w+\ell}
=\sum_{i=0}^{\ell-1}
\hat k_\mathrm{TO,\gamma_w+i},
\qquad
\ell=0,\ldots,\Gamma_W-1.
$$

将整数修正加回观测，可重构连续轨迹

$$
\tilde k_{\tau,\gamma_w+\ell}
=\hat k_{\tau,\gamma_w+\ell}
+A_{\gamma_w+\ell}.
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

表示每帧由采样间隔偏移（SIO）引起的时延漂移，$\Delta T_{as,w}$ 为该窗口的平均采样间隔偏差。

## 5. 连续感知时偏

感知时偏以“预测 + 已施加修正 + 观测反馈”的形式递归更新：

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

$\hat\epsilon_\mathrm{SIO,w-1}$ 预测下一帧漂移，$-\hat k_\mathrm{TO,\gamma-1}$ 抵消通信链路移动帧起点造成的坐标变化，$\mu_\gamma e_\gamma$ 则防止长期预测误差积累。稳定时使用较小 $\mu_\gamma$ 以抑制噪声；持续偏差出现时可提高反馈权重以重新收敛。

## 6. 时偏与 SFO 相位补偿

$\hat k_{\tau,\gamma}^\mathrm{sens}$ 的单位是采样，因此对应的秒数为 $\hat k_{\tau,\gamma}^\mathrm{sens}/B$。维度一致的补偿式为

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

正时延在频域产生负相位斜率，因此补偿使用正指数。补偿后的 $\tilde F_{n,m,\gamma}^\mathrm{UE}$ 沿连续慢时间拼接，随后采用与单站相同的杂波抑制、时延–多普勒和微多普勒处理，但 UE 的单通道观测不包含 ULA 角度维。

## 7. 双站结果解释

以 LoS 为参考后，峰值时延对应

$$
\Delta\tau_l
=\frac{d_{B,l}+d_{l,U}-d_{B,U}}{c},
\qquad
\Delta d_l=c\Delta\tau_l.
$$

因此结果是相对 LoS 的超额路径，而不是单站距离 $c\tau/2$。若要恢复绝对传播时延，需要已知 BS–UE 基线距离，或使用双向定时关系进一步分离绝对时偏。

该 OTA 方法在 LoS 或另一个稳定主导路径持续可见时最可靠。丰富 NLoS 场景中，主峰在多条路径之间切换会改变参考点，需要更长时间平均、路径连续性约束，或使用 [eRTM 双向定时](/zh-cn/docs/signal-processing/ota-ertm-timing/)提供额外约束。
