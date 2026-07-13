# eRTM 时间关系与算法说明

本文整理 bi-static ISAC 场景下 eRTM（enhanced round-trip measurement）的时间关系、delay spectrum folding 关系，以及基于 FFT/IFFT 的循环互相关 TO 估计算法。本文不使用原论文中的公式编号，而是按照工程实现逻辑重新组织。

> 记号说明：本文中的 `≜` / `\triangleq` 表示“在 delay spectrum folding 意义下等价”。更严格地说，它可以理解为模最大无模糊延迟 `τ_max` 的等价关系。

---

## 1. 场景与目标

考虑 BS-UE bi-static sensing。基本 round-trip measurement 时序为：

```text
BS 发出下行 sensing signal
        ↓
经过第 l 条路径传播到 UE
        ↓
UE 接收下行信号
        ↓
UE 根据本地 uplink timing 发出上行 sensing signal
        ↓
经过近似互易的同一条第 l 条路径传播回 BS
        ↓
BS 接收上行信号
```

系统要解决的问题是：BS 和 UE 有各自的本地时钟，无法天然共享绝对时间轴，因此两侧 delay spectrum 都带有 timing offset，简称 TO。

传统 RTT 往往依赖 LOS 或最强路径。一旦 NLOS rich multipath 下最强路径不稳定，BS 和 UE 两侧可能选到不同路径，产生 path mismatch。eRTM 的核心思想是：

$$
\boxed{\text{不依赖单条路径，而是利用整个 delay spectrum 的循环平移关系估计 TO}}
$$

---

## 2. 下行、上行与测量延迟

第 `l` 条路径的真实无线传播延迟记为：

$$
\tau_l
$$

下行总延迟包含无线传播延迟和下行 RF 链路固定延迟：

$$
\tau_l^{DL}
=
\tau_l
+
\tau_{DL}^{RF}
$$

上行总延迟包含无线传播延迟和上行 RF 链路固定延迟：

$$
\tau_l^{UL}
=
\tau_l
+
\tau_{UL}^{RF}
$$

其中：

- `τ_DL^RF`：BS 发射链路与 UE 接收链路引入的下行固定延迟；
- `τ_UL^RF`：UE 发射链路与 BS 接收链路引入的上行固定延迟。

这两个 RF 延迟由硬件和射频链路决定，工程上应通过校准得到，然后放入配置文件中。当前实现中，UE YAML 使用 `uplink.ertm_dl_rf_delay_samples` 和 `uplink.ertm_ul_rf_delay_samples`，单位为 samples，支持小数样本。

UE 侧根据自身本地 timing 测得的路径延迟为：

$$
\tau_l^{UE}
=
\tau_l
+
\tau_{TO}^{UE}
$$

BS 侧根据自身本地 timing 测得的路径延迟为：

$$
\tau_l^{BS}
=
\tau_l
+
\tau_{TO}^{BS}
$$

因此，`τ_l^UE` 和 `τ_l^BS` 都不是纯传播延迟，而是分别带有 UE 侧和 BS 侧的 TO。

---

## 3. UE 侧 rx-to-tx gap

UE 侧定义的时间间隔为：

$$
t_{rx-tx,l}^{UE}
=
T
-
\tau_l^{UE}
-
t_{TA}^{UE}
$$

其中：

- `t_{rx-tx,l}^UE`：UE 从接收到第 `l` 条下行路径，到发出上行信号之间的时间间隔；
- `T`：一个 OFDM symbol 的时间长度，包括 CP；
- `τ_l^UE`：UE 侧测得的第 `l` 条下行路径延迟；
- `t_TA^UE`：UE timing advance。

推导方式如下。以 UE 的本地下行 timing 为参考，令当前参考 OFDM symbol 起点为 `0`。第 `l` 条下行路径到达 UE 的时刻为：

$$
\tau_l^{UE}
$$

如果没有 timing advance，UE 大约在下一个 symbol 边界 `T` 附近发送上行信号。由于 timing advance 存在，UE 会提前发送，上行发送时刻变为：

$$
T - t_{TA}^{UE}
$$

于是 UE 从收到下行到发出上行之间的 gap 为：

$$
\begin{aligned}
t_{rx-tx,l}^{UE}
&=
\text{UE 上行发送时刻}
-
\text{UE 下行接收时刻}
\\
&=
(T - t_{TA}^{UE}) - \tau_l^{UE}
\\
&=
T - \tau_l^{UE} - t_{TA}^{UE}
\end{aligned}
$$

这里两个负号都有直观意义：

- 路径越晚到 UE，UE 剩余等待时间越短，因此 `τ_l^UE` 是负号；
- timing advance 越大，UE 发上行越早，因此 `t_TA^UE` 也是负号。

---

## 4. BS 侧 tx-to-rx gap

BS 侧定义的时间间隔为：

$$
t_{tx-rx,l}^{BS}
=
T
+
\tau_l^{BS}
+
t_{DL-UL}^{BS}
$$

其中：

- `t_{tx-rx,l}^BS`：BS 从发出下行信号，到接收到第 `l` 条上行路径之间的时间间隔；
- `T`：一个 OFDM symbol 的时间长度，包括 CP；
- `τ_l^BS`：BS 侧测得的第 `l` 条上行路径延迟；
- `t_DL-UL^BS`：BS 侧 uplink timing 与 downlink timing 之间的时间差。

推导方式如下。以 BS 下行发射 timing 为参考，令 BS 发出下行信号的参考时刻为 `0`。若 uplink timing 与 downlink timing 完全一致，对应的 uplink 接收参考边界约为 `T`。实际系统中二者存在差异：

$$
t_{DL-UL}^{BS}
$$

因此 BS 的 uplink 接收参考边界位于：

$$
T + t_{DL-UL}^{BS}
$$

第 `l` 条上行路径相对于 BS uplink timing 的测得延迟为：

$$
\tau_l^{BS}
$$

所以 BS 实际收到该路径的时刻为：

$$
T + t_{DL-UL}^{BS} + \tau_l^{BS}
$$

从 BS 发下行到收上行的 gap 为：

$$
\begin{aligned}
t_{tx-rx,l}^{BS}
&=
\text{BS 上行接收时刻}
-
\text{BS 下行发射时刻}
\\
&=
T + t_{DL-UL}^{BS} + \tau_l^{BS}
\\
&=
T + \tau_l^{BS} + t_{DL-UL}^{BS}
\end{aligned}
$$

这里 `τ_l^BS` 是正号，因为上行路径越晚到 BS，BS 从发下行到收上行的总等待时间越长。

---

## 5. Round-trip 闭环关系

BS 侧观测到的总 tx-to-rx gap 减去 UE 内部的 rx-to-tx gap，剩下的就是信号真正经历的下行总延迟与上行总延迟：

$$
t_{tx-rx,l}^{BS}
-
t_{rx-tx,l}^{UE}
=
\tau_l^{DL}
+
\tau_l^{UL}
$$

将前面的两个 gap 代入，有：

$$
(T + \tau_l^{BS} + t_{DL-UL}^{BS})
-
(T - \tau_l^{UE} - t_{TA}^{UE})
=
\tau_l^{DL}
+
\tau_l^{UL}
$$

左边展开后，`T` 被消掉：

$$
\tau_l^{BS}
+
\tau_l^{UE}
+
t_{DL-UL}^{BS}
+
t_{TA}^{UE}
=
\tau_l^{DL}
+
\tau_l^{UL}
$$

右边代入下行与上行总延迟：

$$
\tau_l^{DL}
+
\tau_l^{UL}
=
2\tau_l
+
\tau_{DL}^{RF}
+
\tau_{UL}^{RF}
$$

于是得到：

$$
\tau_l^{BS}
+
\tau_l^{UE}
=
2\tau_l
+
\tau_{DL}^{RF}
+
\tau_{UL}^{RF}
-
t_{DL-UL}^{BS}
-
t_{TA}^{UE}
$$

定义常数项：

$$
\tau_c
=
\tau_{DL}^{RF}
+
\tau_{UL}^{RF}
-
t_{DL-UL}^{BS}
-
t_{TA}^{UE}
$$

则上式可简写为：

$$
\tau_l^{BS}
+
\tau_l^{UE}
=
2\tau_l
+
\tau_c
$$

工程上：

- `τ_DL^RF`、`τ_UL^RF` 来自配置文件；
- `t_DL-UL^BS` 可由运行时 DUTI 读取或换算得到；
- `t_TA^UE` 可由运行时 TADV 读取或换算得到。

---

## 6. BS-UE 测得延迟差

由两侧测得延迟定义：

$$
\tau_l^{UE}
=
\tau_l
+
\tau_{TO}^{UE}
$$

$$
\tau_l^{BS}
=
\tau_l
+
\tau_{TO}^{BS}
$$

两式相减可得：

$$
\tau_l^{BS}
-
\tau_l^{UE}
=
\tau_{TO}^{BS}
-
\tau_{TO}^{UE}
$$

定义两侧 TO 差值：

$$
\tau_{TO}^{BS-UE}
=
\tau_{TO}^{BS}
-
\tau_{TO}^{UE}
$$

则：

$$
\tau_l^{BS}
=
\tau_l^{UE}
+
\tau_{TO}^{BS-UE}
$$

这个关系对所有路径都成立。因此，在近似信道互易的情况下，BS 和 UE 看到的是同一组多径，只是在 delay axis 上存在共同偏移。

---

## 7. Delay spectrum folding 与循环平移

OFDM sensing 的 delay spectrum 有最大无模糊延迟：

$$
\tau_{max}
=
\frac{1}{\eta_f \Delta f}
$$

对应的 delay resolution 为：

$$
\epsilon
=
\frac{1}{N\eta_f\Delta f}
$$

其中：

- `N`：sensing subcarrier 数量；
- `η_f`：相邻 sensing subcarrier 的子载波间隔倍数；
- `Δf`：子载波间隔。

当延迟值超出 `[0, τ_max]` 时，delay spectrum 会发生 folding。也就是说，超出最大无模糊延迟的路径会绕回到 delay spectrum 前端；小于 0 的路径会从后端绕回。

UE 侧 folding 后的路径位置满足：

$$
\tau_{rx,l}^{UE}
\triangleq
\tau_l^{UE}
$$

也可以写成：

$$
\tau_{rx,l}^{UE}
=
\tau_l^{UE}
+
\kappa_l^{UE}\tau_{max}
$$

BS 侧 folding 后的路径位置满足：

$$
\tau_{rx,l}^{BS}
\triangleq
\tau_l^{BS}
$$

也可以写成：

$$
\tau_{rx,l}^{BS}
=
\tau_l^{BS}
+
\kappa_l^{BS}\tau_{max}
$$

这里的 `κ` 是整数，用来把延迟折叠回可观测区间。更规范的理解是：

$$
\tau_{rx,l}^{UE}
\equiv
\tau_l^{UE}
\pmod{\tau_{max}}
$$

$$
\tau_{rx,l}^{BS}
\equiv
\tau_l^{BS}
\pmod{\tau_{max}}
$$

考虑 folding 后，延迟和关系变成：

$$
\tau_{rx,l}^{BS}
+
\tau_{rx,l}^{UE}
\triangleq
2\tau_l
+
\tau_c
$$

BS-UE 延迟差关系变成：

$$
\tau_{rx,l}^{BS}
\triangleq
\tau_{rx,l}^{UE}
+
\tau_{TO}^{BS-UE}
$$

这就是 eRTM 的理论核心：

$$
\boxed{
\text{BS 侧 delay spectrum 是 UE 侧 delay spectrum 的 cyclic shift 版本}
}
$$

---

## 8. 由 BS-UE shift 得到 UE 和 BS 的 TO

将 BS-UE 的 folding 平移关系代入延迟和关系：

$$
(\tau_{rx,l}^{UE} + \tau_{TO}^{BS-UE})
+
\tau_{rx,l}^{UE}
\triangleq
2\tau_l
+
\tau_c
$$

整理得到：

$$
\tau_{rx,l}^{UE}
\triangleq
\tau_l
+
\frac{\tau_c}{2}
-
\frac{\tau_{TO}^{BS-UE}}{2}
$$

因此 UE 侧测得 delay spectrum 相对于真实传播 delay profile 的整体偏移为：

$$
\tau_{TO}^{UE}
=
\frac{\tau_c}{2}
-
\frac{\tau_{TO}^{BS-UE}}{2}
$$

再由：

$$
\tau_{TO}^{BS-UE}
=
\tau_{TO}^{BS}
-
\tau_{TO}^{UE}
$$

可得：

$$
\tau_{TO}^{BS}
=
\frac{\tau_c}{2}
+
\frac{\tau_{TO}^{BS-UE}}{2}
$$

工程含义是：只要能从两侧 delay spectrum 估计出 `τ_TO^{BS-UE}`，再结合 `τ_c`，就能得到 UE 和 BS 两侧各自的 TO。

---

## 9. 从 CSI 生成 delay spectrum

下行测量中，UE 获得一个 CSI 矩阵：

$$
\mathbf{H}^{UE}
=
[\mathbf{h}_0^{UE},\mathbf{h}_1^{UE},\cdots,\mathbf{h}_{M-1}^{UE}]
\in \mathbb{C}^{N\times M}
$$

其中 `N` 是 sensing subcarrier 数量，`M` 是一个 CPI 内 sensing OFDM symbol 数量。

上行测量中，BS 至少获得一个 CSI 向量：

$$
\mathbf{h}^{BS}\in\mathbb{C}^{N\times 1}
$$

对频域 CSI 做 `P` 点 IDFT，即可得到 delay spectrum。工程实现中可以用 IFFT 实现：

$$
\mathbf{r}^{BS}
=
\operatorname{IDFT}_P\{\mathbf{h}^{BS}\}
$$

$$
\mathbf{r}_m^{UE}
=
\operatorname{IDFT}_P\{\mathbf{h}_m^{UE}\}
$$

所有 UE OFDM symbol 的 delay spectrum 组成：

$$
\mathbf{R}^{UE}
=
[\mathbf{r}_0^{UE},\mathbf{r}_1^{UE},\cdots,\mathbf{r}_{M-1}^{UE}]
\in\mathbb{C}^{P\times M}
$$

其中：

$$
P = \beta N
$$

`β` 是 delay-domain oversampling factor。

默认情况下，`P=N` 表示没有 oversampling；如果 `P>N`，则 delay-domain 采样间隔变小，有助于提高 TO 估计精度。

Delay-domain 采样间隔为：

$$
\Delta\tau_{bin}
=
\frac{\epsilon}{\beta}
$$

如果直接用 `P` 点 IFFT 描述，也可以写成：

$$
\Delta\tau_{bin}
=
\frac{\tau_{max}}{P}
$$

二者等价，因为：

$$
\tau_{max}=N\epsilon
$$

---

## 10. 为什么互相关用幅度谱而不是复数谱

理想情况下，BS 和 UE 的复数 delay spectrum 之间不仅有 delay shift，也可能存在：

- CFO 引入的相位旋转；
- random phase；
- TD 引起的额外相位变化；
- 上下行 RF 链路不完全一致引起的复增益变化。

如果直接对复数 delay spectrum 做 cross-correlation，峰值可能被相位差破坏，导致错误的 shift 估计。

因此 eRTM 对归一化后的**幅度 delay spectrum**做循环互相关：

$$
|\bar{\mathbf{r}}^{BS}|
\quad \text{and} \quad
|\bar{\mathbf{r}}_0^{UE}|
$$

幅度谱更接近 multipath power profile，受 CFO 和 random phase 影响更小。

---

## 11. Delay spectrum 归一化

由于上下行覆盖、发射功率、接收链路增益不同，UE 和 BS 两侧 delay spectrum 的总功率一般不同。为了让互相关只关注形状相似性，需要先归一化。

BS 侧归一化：

$$
\bar{\mathbf{r}}^{BS}
=
\frac{\mathbf{r}^{BS}}
{\sqrt{\sum_{p=0}^{P-1}|r^{BS}(p)|^2}}
$$

UE 侧选择一个参考 OFDM symbol，通常选一个 CPI 内的第一个 sensing OFDM symbol：

$$
\bar{\mathbf{r}}_0^{UE}
=
\frac{\mathbf{r}_0^{UE}}
{\sqrt{\sum_{p=0}^{P-1}|r_0^{UE}(p)|^2}}
$$

选择参考 OFDM symbol 的原则：

- 与上行测量 symbol 尽量处在同一个 timing adjustment period；
- 与上行测量 symbol 的时间间隔尽量小；
- 这样可以减小 TD 对 round-trip measurement 的影响。

---

## 12. 用 FFT/IFFT 实现循环互相关

直接在 delay domain 做循环互相关的复杂度较高。工程上可以利用 FFT/IFFT：

$$
\mathbf{x}
=
\operatorname{IDFT}_P
\left\{
\operatorname{DFT}_P\{|\bar{\mathbf{r}}^{BS}|\}
\odot
\left(
\operatorname{DFT}_P\{|\bar{\mathbf{r}}_0^{UE}|\}
\right)^*
\right\}
$$

其中：

- `⊙` 表示逐点相乘；
- `*` 表示复共轭；
- `x[p]` 是两个幅度 delay spectrum 在循环 shift 为 `p` 时的相关结果。

这一式子对应循环互相关的 FFT 实现。直观上，它在寻找：UE 侧 delay spectrum 经过多大循环平移后，最像 BS 侧 delay spectrum。

---

## 13. 由互相关峰值估计循环 shift

找到互相关峰值位置：

$$
\delta_{max}
=
\arg\max_{p\in[0,P-1]} |x[p]|
$$

由于 `δ_max` 是 modulo `P` 的循环索引，需要转换为有符号 shift：

$$
\delta_{BS-UE}
=
\begin{cases}
\delta_{max}, & \delta_{max}\le P/2 \\
\delta_{max}-P, & \delta_{max}>P/2
\end{cases}
$$

于是 BS-UE 的 TO 差估计为：

$$
\hat{\tau}_{TO}^{BS-UE}
=
\delta_{BS-UE}\cdot\Delta\tau_{bin}
$$

也就是：

$$
\hat{\tau}_{TO}^{BS-UE}
=
\delta_{BS-UE}\cdot\frac{\epsilon}{\beta}
$$

或者等价地：

$$
\hat{\tau}_{TO}^{BS-UE}
=
\delta_{BS-UE}\cdot\frac{\tau_{max}}{P}
$$

---

## 14. 由 shift 估计 UE/BS TO

先计算常数项：

$$
\tau_c
=
\tau_{DL}^{RF}
+
\tau_{UL}^{RF}
-
t_{DL-UL}^{BS}
-
t_{TA}^{UE}
$$

其中工程输入建议如下：

| 项 | 来源 | 说明 |
|---|---|---|
| `τ_DL^RF` | `uplink.ertm_dl_rf_delay_samples` | 下行 RF 链路校准延迟，当前实现单位为 samples，支持小数 |
| `τ_UL^RF` | `uplink.ertm_ul_rf_delay_samples` | 上行 RF 链路校准延迟，当前实现单位为 samples，支持小数 |
| `t_DL-UL^BS` | 运行时 DUTI / `uplink.bs_dl_ul_timing_diff` | BS 侧 DL-UL timing difference，当前实现单位为 samples |
| `t_TA^UE` | 运行时 TADV / `uplink.ue_timing_advance` | UE timing advance，当前实现单位为 samples |

UE 侧 TO 估计：

$$
\hat{\tau}_{TO}^{UE}
=
\frac{\tau_c}{2}
-
\frac{\hat{\tau}_{TO}^{BS-UE}}{2}
$$

BS 侧 TO 估计：

$$
\hat{\tau}_{TO}^{BS}
=
\frac{\tau_c}{2}
+
\frac{\hat{\tau}_{TO}^{BS-UE}}{2}
$$

---

## 15. TO 抑制：delay spectrum 对齐

如果要校正 UE 侧 delay spectrum，需要把 UE 侧 delay spectrum 沿 delay axis 反向循环平移。

需要平移的 delay bin 数量为：

$$
\delta_{TO}
=
\operatorname{round}\left(
\frac{\hat{\tau}_{TO}^{UE}}{\Delta\tau_{bin}}
\right)
$$

也可以写成：

$$
\delta_{TO}
=
\operatorname{round}\left(
\frac{\hat{\tau}_{TO}^{UE}}{\epsilon/\beta}
\right)
$$

对 UE 侧每个 sensing OFDM symbol 的 delay spectrum 做：

$$
\mathbf{r}_{m,aligned}^{UE}
=
\operatorname{roll}(\mathbf{r}_m^{UE}, -\delta_{TO})
$$

对于整个矩阵：

$$
\mathbf{R}_{aligned}^{UE}
=
\operatorname{roll}(\mathbf{R}^{UE}, -\delta_{TO}, \text{axis}=\text{delay})
$$

注意：

- TO 是一个 CPI 内的初始 timing offset，所以同一 CPI 内所有 sensing OFDM symbols 使用相同的 `δ_TO`；
- TD、CFO、random phase 可以在后续 ADD 或类似处理链中继续补偿；
- 如果 TO 抑制作用于 delay-Doppler spectrum，也应沿 delay axis 做同样的循环平移。

---

## 16. 完整算法流程

### 输入

- UE 侧下行 CSI 矩阵：`H_UE ∈ C^{N×M}`；
- BS 侧上行 CSI 向量：`h_BS ∈ C^{N×1}`；
- sensing 参数：`N`、`P`、`η_f`、`Δf`；
- RF 校准参数：`τ_DL^RF`、`τ_UL^RF`；
- 运行时 timing 参数：`DUTI`、`TADV`，换算得到 `t_DL-UL^BS`、`t_TA^UE`。

### 输出

- BS-UE TO 差值估计：`τ_hat_TO_BS_UE`；
- UE 侧 TO 估计：`τ_hat_TO_UE`；
- BS 侧 TO 估计：`τ_hat_TO_BS`；
- 可选：TO 对齐后的 UE delay spectrum 或 delay-Doppler spectrum。

### 步骤

1. 根据 sensing 参数计算：

   $$
   \tau_{max}=\frac{1}{\eta_f\Delta f}
   $$

   $$
   \epsilon=\frac{1}{N\eta_f\Delta f}
   $$

   $$
   \Delta\tau_{bin}=\frac{\tau_{max}}{P}
   $$

2. 对 BS 侧上行 CSI 做 `P` 点 IFFT，得到：

   $$
   \mathbf{r}^{BS}
   $$

3. 对 UE 侧下行 CSI 的参考 OFDM symbol 做 `P` 点 IFFT，得到：

   $$
   \mathbf{r}_0^{UE}
   $$

4. 对两侧 delay spectrum 做功率归一化。

5. 取幅度谱：

   $$
   |\bar{\mathbf{r}}^{BS}|,
   \quad
   |\bar{\mathbf{r}}_0^{UE}|
   $$

6. 用 FFT/IFFT 计算循环互相关：

   $$
   \mathbf{x}
   =
   \operatorname{IFFT}
   \left(
   \operatorname{FFT}(|\bar{\mathbf{r}}^{BS}|)
   \odot
   \operatorname{FFT}(|\bar{\mathbf{r}}_0^{UE}|)^*
   \right)
   $$

7. 搜索相关峰：

   $$
   \delta_{max}=\arg\max |\mathbf{x}|
   $$

8. 将循环索引转换为有符号 shift：

   $$
   \delta_{BS-UE}
   =
   \begin{cases}
   \delta_{max}, & \delta_{max}\le P/2 \\
   \delta_{max}-P, & \delta_{max}>P/2
   \end{cases}
   $$

9. 换算 BS-UE TO 差值：

   $$
   \hat{\tau}_{TO}^{BS-UE}
   =
   \delta_{BS-UE}\Delta\tau_{bin}
   $$

10. 由配置和运行时参数计算：

    $$
    \tau_c
    =
    \tau_{DL}^{RF}
    +
    \tau_{UL}^{RF}
    -
    t_{DL-UL}^{BS}
    -
    t_{TA}^{UE}
    $$

11. 计算 UE 与 BS 的 TO：

    $$
    \hat{\tau}_{TO}^{UE}
    =
    \frac{\tau_c - \hat{\tau}_{TO}^{BS-UE}}{2}
    $$

    $$
    \hat{\tau}_{TO}^{BS}
    =
    \frac{\tau_c + \hat{\tau}_{TO}^{BS-UE}}{2}
    $$

12. 可选：计算 UE delay spectrum 需要反向循环平移的 bin 数：

    $$
    \delta_{TO}
    =
    \operatorname{round}
    \left(
    \frac{\hat{\tau}_{TO}^{UE}}{\Delta\tau_{bin}}
    \right)
    $$

13. 对 UE delay spectrum 或 delay-Doppler spectrum 沿 delay axis 执行：

    $$
    \operatorname{roll}(\cdot, -\delta_{TO})
    $$

---

## 17. 伪代码

```python
def estimate_to_ertm(
    h_bs,              # shape: [N]
    H_ue,              # shape: [N, M]
    eta_f,
    subcarrier_spacing,
    sample_rate,
    p_fft,
    tau_rf_dl_samples,
    tau_rf_ul_samples,
    duti_samples,
    tadv_samples,
    ue_ref_symbol=0,
):
    N = len(h_bs)
    P = p_fft

    tau_max = 1.0 / (eta_f * subcarrier_spacing)
    epsilon = 1.0 / (N * eta_f * subcarrier_spacing)
    delay_bin = tau_max / P
    delay_bin_samples = delay_bin * sample_rate

    # Frequency-domain CSI -> delay spectrum.
    r_bs = ifft_zero_padded(h_bs, P)
    r_ue = ifft_zero_padded(H_ue[:, ue_ref_symbol], P)

    # Power normalization.
    r_bs_bar = r_bs / sqrt(sum(abs(r_bs)**2) + eps)
    r_ue_bar = r_ue / sqrt(sum(abs(r_ue)**2) + eps)

    # Use amplitude spectra for robustness against CFO/random phase.
    a_bs = abs(r_bs_bar)
    a_ue = abs(r_ue_bar)

    # Cyclic cross-correlation via FFT/IFFT.
    x = ifft(fft(a_bs) * conj(fft(a_ue)))
    delta_max = argmax(abs(x))

    # Convert modulo index to signed cyclic shift.
    if delta_max <= P // 2:
        delta_bs_ue = delta_max
    else:
        delta_bs_ue = delta_max - P

    tau_to_bs_ue_samples = delta_bs_ue * delay_bin_samples

    # Constant term from RF calibration and runtime DUTI/TADV.
    # Current implementation keeps these terms in samples.
    tau_c_samples = (
        tau_rf_dl_samples
        + tau_rf_ul_samples
        - duti_samples
        - tadv_samples
    )

    tau_to_ue_samples = 0.5 * (tau_c_samples - tau_to_bs_ue_samples)
    tau_to_bs_samples = 0.5 * (tau_c_samples + tau_to_bs_ue_samples)

    delta_to = round(tau_to_ue_samples / delay_bin_samples)

    return {
        "tau_max": tau_max,
        "epsilon": epsilon,
        "delay_bin": delay_bin,
        "delay_bin_samples": delay_bin_samples,
        "delta_bs_ue": delta_bs_ue,
        "tau_to_bs_ue_samples": tau_to_bs_ue_samples,
        "tau_c_samples": tau_c_samples,
        "tau_to_ue_samples": tau_to_ue_samples,
        "tau_to_bs_samples": tau_to_bs_samples,
        "delta_to": delta_to,
        "corr_peak": abs(x[delta_max]),
    }
```

其中 `ifft_zero_padded` 表示：将长度为 `N` 的 sensing-subcarrier CSI 放入长度为 `P` 的向量，再执行 `P` 点 IFFT。实际实现时要注意 subcarrier 排列方式是否已经是连续 sensing bins。如果 sensing subcarrier 在原始 FFT grid 中是间隔抽取的，应先按 sensing-subcarrier 序列组织成长度 `N` 的有效 CSI，再做 `P` 点 IDFT。

---

## 18. 互相关结果的工程检查

为了避免错误峰值，建议记录并检查以下量：

| 变量 | 含义 | 检查建议 |
|---|---|---|
| `delta_max` | 相关峰位置 | 是否随场景变化连续 |
| `delta_bs_ue` | 有符号 cyclic shift | 是否落在合理范围 |
| `corr_peak` | 相关峰值 | 低 SNR 下是否明显高于旁瓣 |
| `tau_to_bs_ue` | BS-UE TO 差值 | 是否与 DUTI/TADV 或几何变化趋势一致 |
| `tau_to_ue` | UE 侧 TO | 是否在 delay spectrum 周期内稳定 |
| `delta_to` | 对齐用 bin shift | 是否出现不合理跳变 |

如果相关峰多峰或跳变严重，可能原因包括：

- BS 和 UE 的参考测量不在同一 timing adjustment period；
- 上下行测量之间时间间隔过大，TD 影响显著；
- 上下行使用的天线方向图差异太大，破坏信道互易性；
- SNR 过低；
- 静态 clutter 太少，delay spectrum 形状不稳定；
- delay spectrum folding 超出设计假设，即真实延迟跨越多个 `τ_max` 周期。

---

## 19. 与传统 RTT 的差异

传统 RTT 通常依赖某一条路径，尤其是 LOS 或最强路径：

```text
UE 测某条路径的 rx-to-tx gap
BS 测某条路径的 tx-to-rx gap
两者结合得到该路径的往返时间
```

在 LOS 下，这通常可行，因为 LOS path 很强，BS 和 UE 容易选到同一条路径。

在 NLOS rich multipath 下，最强路径可能不稳定，BS 和 UE 侧的最强路径也未必对应同一条物理路径。这会造成 path mismatch，导致 TO 估计出现大错误。

相比之下，eRTM 利用的是完整 delay spectrum 的整体形状：

$$
\mathbf{g}_{rx}^{BS}
\triangleq
\mathbf{g}_{rx}^{UE}
+
\tau_{TO}^{BS-UE}
$$

因此它不需要判断哪一条路径是 LOS，也不需要显式匹配某一条具体路径，而是通过循环互相关寻找整体 shift。

---

## 20. 与 ADD/后续 sensing 处理链的关系

eRTM 主要解决 CPI 初始 TO 问题。一个完整的异步 bi-static sensing 处理链通常还需要处理：

- TD：timing drift，即 CPI 内随 OFDM symbol 变化的 timing 漂移；
- CFO：carrier frequency offset；
- random phase：不同 OFDM symbol 上的随机相位；
- angle / Doppler / delay peak detection。

一个典型处理链可以是：

```text
UE 下行 CSI H_UE, BS 上行 CSI h_BS
        ↓
P 点 IFFT 得到 delay spectrum
        ↓
eRTM：归一化 + 幅度谱 cyclic cross-correlation
        ↓
估计 τ_TO^{BS-UE}, τ_TO^{UE}, τ_TO^{BS}
        ↓
沿 delay axis 循环平移，完成 TO suppression
        ↓
ADD 或其他算法继续处理 TD/CFO/random phase
        ↓
得到 TO/TD/CFO 补偿后的 delay-Doppler/angle 结果
```

---

## 21. 实现时的单位约定

理论推导可以使用秒作为统一时间单位；当前工程实现为了避免纳秒和采样率换算误差，将 eRTM TO 相关量统一使用 samples：

| 参数 | 建议内部单位 |
|---|---|
| `τ_DL^RF` | samples |
| `τ_UL^RF` | samples |
| `t_DL-UL^BS` / DUTI | samples |
| `t_TA^UE` / TADV | samples |
| `τ_c` | samples |
| `τ_TO^{BS-UE}` | samples |
| `τ_TO^{UE}` | samples |
| `τ_TO^{BS}` | samples |
| `τ_max` | second |
| `epsilon` | second |
| `delay_bin` | second |

如果底层 DUTI/TADV 是采样点、tick、Tc、Ts 或协议单位，需要在进入 eRTM 前统一换算成秒。

---

## 22. 核心结论

整个 eRTM 可以压缩为三句话：

1. BS 和 UE 看到的是近似相同的 multipath delay profile，但由于 TO 不同，两侧 delay spectrum 存在一个循环平移；
2. 用归一化幅度 delay spectrum 的 FFT/IFFT 循环互相关，可以稳健估计这个循环平移；
3. 结合 RF 校准延迟和运行时 DUTI/TADV 组成的 `τ_c`，即可得到 UE/BS 两侧 TO，并通过 delay-axis roll 完成 TO 抑制。

最终用于估计和校正的核心关系为：

$$
\hat{\tau}_{TO}^{BS-UE}
=
\delta_{BS-UE}\frac{\tau_{max}}{P}
$$

$$
\tau_c
=
\tau_{DL}^{RF}
+
\tau_{UL}^{RF}
-
t_{DL-UL}^{BS}
-
t_{TA}^{UE}
$$

$$
\hat{\tau}_{TO}^{UE}
=
\frac{\tau_c - \hat{\tau}_{TO}^{BS-UE}}{2}
$$

$$
\hat{\tau}_{TO}^{BS}
=
\frac{\tau_c + \hat{\tau}_{TO}^{BS-UE}}{2}
$$

$$
\delta_{TO}
=
\operatorname{round}\left(
\frac{\hat{\tau}_{TO}^{UE}}{\tau_{max}/P}
\right)
$$

$$
\mathbf{R}_{aligned}^{UE}
=
\operatorname{roll}(\mathbf{R}^{UE}, -\delta_{TO}, \text{axis}=\text{delay})
$$
