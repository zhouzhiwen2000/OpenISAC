export const footerAuthors = [
  { name: "Zhiwen Zhou", email: "zhiwen_zhou@seu.edu.cn" },
  { name: "Chaoyue Zhang", email: "chaoyue_zhang@seu.edu.cn" },
  { name: "Xiaoli Xu", email: "xiaolixu@seu.edu.cn" },
  { name: "Yong Zeng", email: "yong_zeng@seu.edu.cn" },
];

export const navItems = {
  en: [
    { key: "home", href: "index.html", label: "Home" },
    { key: "documentation", href: "documentation.html", label: "Documentation" },
    { key: "architecture", href: "architecture.html", label: "Architecture" },
    { key: "signal", href: "signal_processing.html", label: "Signal Processing" },
    {
      key: "github",
      href: "https://github.com/zhouzhiwen2000/OpenISAC",
      label: "GitHub",
      external: true,
    },
    {
      key: "community",
      href: "https://qm.qq.com/q/NIQRNGb0kY",
      label: "QQ Group",
      external: true,
    },
  ],
  zh: [
    { key: "home", href: "index_zh.html", label: "首页" },
    { key: "documentation", href: "documentation_zh.html", label: "文档" },
    { key: "architecture", href: "architecture_zh.html", label: "架构" },
    { key: "signal", href: "signal_processing_zh.html", label: "信号处理" },
    {
      key: "github",
      href: "https://github.com/zhouzhiwen2000/OpenISAC",
      label: "GitHub",
      external: true,
    },
    {
      key: "community",
      href: "https://qm.qq.com/q/NIQRNGb0kY",
      label: "QQ群",
      external: true,
    },
  ],
};

export const homeContent = {
  en: {
    title: "OpenISAC - Real-Time OFDM ISAC Platform",
    description: "OpenISAC project documentation website.",
    hero: {
      tagline: "A Versatile and High-Performance Open-Source Platform for Real-Time ISAC Experimentation",
      ctaLabel: "Get Started",
      ctaHref: "#hardware",
    },
    about: {
      title: "About OpenISAC",
      html: String.raw`<p>Integrated sensing and communication (ISAC) is envisioned to be one of the key technologies in 6G. OpenISAC is a versatile and high-performance open-source platform designed to bridge the gap between theory and practice.</p>
<p>Built entirely on open-source software (UHD + C++ + Python), it supports real-time OFDM-based sensing and communication. A key feature is its novel over-the-air synchronization mechanism, enabling robust bistatic operations without wired connections.</p>
<p>If you find this repository useful, please cite our paper:</p>
<p>Z. Zhou, C. Zhang, X. Xu, and Y. Zeng, "OpenISAC: An Open-Source Real-Time Experimentation Platform for OFDM-ISAC with Over-the-Air Synchronization," <i>arXiv preprint</i> arXiv:2601.03535, Jan. 2026.</p>
<p>[<a href="https://arxiv.org/pdf/2601.03535" target="_blank" rel="noreferrer noopener">arXiv</a>]</p>`,
    },
    affiliation: {
      title: "Affiliation",
      html: String.raw`<div class="logo-row logo-row-center">
  <img src="images/SEUlogo.png" alt="SEU Logo">
  <img src="images/PML.png" alt="PML Logo">
</div>
<p class="affiliation-copy"><b>Yong Zeng Group at the National Mobile Communications Research Laboratory, Southeast University and the Purple Mountain Laboratories</b></p>`,
    },
    community: {
      title: "Community",
      links: [
        { label: "Join our QQ Group", href: "https://qm.qq.com/q/NIQRNGb0kY" },
        { label: "Bilibili Channel (Yong Zeng Group)", href: "https://space.bilibili.com/627920129" },
      ],
      wechatLabel: "WeChat Official Account:",
      wechatAlt: "WeChat QR Code",
    },
    scope: {
      title: "What it is — and what it is not",
      html: String.raw`<h3>What OpenISAC is</h3>
<p>OpenISAC is a simple OFDM-based communication and sensing system designed for academic experiments and rapid algorithm validation. Its goal is to provide a clean, minimal, easy-to-modify OFDM platform so researchers can iterate quickly on PHY/sensing ideas without the overhead of a full standard-compliant stack.</p>
<p>Because it focuses on simplicity, OpenISAC typically requires less compute and can often run at higher sampling rates than more complex, feature-complete systems.</p>
<h3>What OpenISAC is not</h3>
<p>OpenISAC is not intended to be a standard-compliant implementation. It does not aim to comply with existing standards such as Wi-Fi or 5G NR.</p>
<p>It is also not meant to replace or compete with full-stack open-source standard implementations such as openwifi or OpenAirInterface. If your goal is interoperability, standards compliance, or a production-grade protocol stack, those projects are the right direction.</p>
<h3>When to use it</h3>
<ul>
  <li>Prototyping and testing new OFDM/ISAC algorithms</li>
  <li>Fast "idea -> experiment" cycles with a minimal PHY</li>
  <li>Research setups where interoperability is not required</li>
</ul>
<h3>When not to use it</h3>
<ul>
  <li>Building a Wi-Fi/NR-compatible system</li>
  <li>Needing real-world standard features (full MAC/stack behavior, interoperability, certification-oriented behavior, etc.)</li>
</ul>`,
    },
    sensingDemo: {
      title: "Sensing Demo",
      items: [
        {
          title: "Delay-Doppler Sensing",
          body: "Real-time delay-Doppler map of a moving drone.",
          videoUrl: "https://vids.zzw123app.top/videos/Drone_Moving.mp4",
        },
        {
          title: "Clutter Rejection",
          body: "Comparison of MTI clutter rejection filter ON and OFF.",
          videoUrl: "https://vids.zzw123app.top/videos/MTI_on_vs_off.mp4",
        },
        {
          title: "Micro-Doppler Sensing",
          body: "Micro-Doppler signatures of drone ascending and descending.",
          videoUrl: "https://vids.zzw123app.top/videos/MicroDoppler_Up_Down.mp4",
        },
      ],
    },
    commDemo: {
      title: "Communication Demo",
      items: [
        {
          title: "Video Streaming",
          body: "Real-time video streaming via ffmpeg over the ISAC link.",
          videoUrl: "https://vids.zzw123app.top/videos/Communication.mp4",
        },
      ],
    },
  },
  zh: {
    title: "OpenISAC - 实时 OFDM 通信感知一体化平台",
    description: "OpenISAC 项目文档站点。",
    hero: {
      tagline: "一个用于通信感知一体化（ISAC）研究的开源通用高性能实时实验平台",
      ctaLabel: "开始使用",
      ctaHref: "#hardware",
    },
    about: {
      title: "关于 OpenISAC",
      html: String.raw`<p>通信感知一体化 (ISAC) 被认为是 6G 的关键技术之一。OpenISAC 是一个通用且高性能的开源平台，旨在弥合理论与实践之间的差距。</p>
<p>它完全基于开源软件 (UHD、C++、Python) 构建，支持基于 OFDM 的实时通信和感知。其核心特性是新颖的空口 (Over-the-Air) 感知同步机制，无需有线连接即可实现稳定的双站感知。</p>
<p>如果您觉得这个仓库有用，请引用我们的论文：</p>
<p>Z. Zhou, C. Zhang, X. Xu, and Y. Zeng, "OpenISAC: An Open-Source Real-Time Experimentation Platform for OFDM-ISAC with Over-the-Air Synchronization," <i>arXiv preprint</i> arXiv:2601.03535, Jan. 2026.</p>
<p>[<a href="https://arxiv.org/pdf/2601.03535" target="_blank" rel="noreferrer noopener">arXiv</a>]</p>`,
    },
    affiliation: {
      title: "所属机构",
      html: String.raw`<div class="logo-row logo-row-center">
  <img src="images/SEUlogo.png" alt="SEU Logo">
  <img src="images/PML.png" alt="PML Logo">
</div>
<p class="affiliation-copy"><b>东南大学移动通信国家重点实验室 & 紫金山实验室 曾勇课题组</b></p>`,
    },
    community: {
      title: "社区",
      links: [
        { label: "加入我们的 QQ 群", href: "https://qm.qq.com/q/NIQRNGb0kY" },
        { label: "Bilibili 频道 (曾勇课题组)", href: "https://space.bilibili.com/627920129" },
      ],
      wechatLabel: "微信公众号:",
      wechatAlt: "WeChat QR Code",
    },
    scope: {
      title: "它是什么，又不是什么？",
      html: String.raw`<h3>OpenISAC 是什么？</h3>
<p>OpenISAC 是一个基于 OFDM 的通信与感知一体化（ISAC）系统，专为学术实验和快速算法验证而设计。</p>
<p>其目标是提供一个简洁且易于修改的 OFDM 平台，使研究人员能够快速迭代通信与感知算法，而无需处理复杂的标准协议栈（如 WIFI、LTE、NR 等）。</p>
<p>由于专注于简洁性，OpenISAC 通常需要更少的计算资源，因此相较于更复杂、功能更全面的系统，能够实现更高的采样率。</p>
<h3>OpenISAC 不是什么？</h3>
<p>OpenISAC 不旨在成为兼容标准的实现（它不符合 Wi-Fi 或 5G NR 等标准）。</p>
<p>它也不旨在替代或竞争全栈开源标准实现，如 openwifi 或 OpenAirInterface。如果您的目标是互操作性、标准合规性或生产级协议栈，那么上述项目是正确的方向。</p>
<h3>何时使用它</h3>
<ul>
  <li>原型设计以及测试新的通信、感知算法</li>
  <li>使用极简 PHY 快速实现新想法</li>
  <li>不需要互操作性的研究</li>
</ul>
<h3>何时不使用它</h3>
<ul>
  <li>构建兼容 Wi-Fi/NR 的系统</li>
  <li>需要标准通信系统的研究（完整的 MAC/协议栈、互操作性等）</li>
</ul>`,
    },
    sensingDemo: {
      title: "感知演示",
      items: [
        {
          title: "时延-多普勒感知",
          body: "运动无人机的实时时延-多普勒图。",
          videoUrl: "https://vids.zzw123app.top/videos/Drone_Moving.mp4",
        },
        {
          title: "杂波抑制",
          body: "MTI 杂波抑制滤波器开启与关闭对比。",
          videoUrl: "https://vids.zzw123app.top/videos/MTI_on_vs_off.mp4",
        },
        {
          title: "微多普勒感知",
          body: "无人机上升和下降的微多普勒特征。",
          videoUrl: "https://vids.zzw123app.top/videos/MicroDoppler_Up_Down.mp4",
        },
      ],
    },
    commDemo: {
      title: "通信演示",
      items: [
        {
          title: "视频流传输",
          body: "通过 ISAC 链路进行基于 ffmpeg 的实时视频流传输。",
          videoUrl: "https://vids.zzw123app.top/videos/Communication.mp4",
        },
      ],
    },
  },
};

export const architectureContent = {
  en: {
    title: "System Architecture - OpenISAC",
    description: "Architecture overview of OpenISAC.",
    sections: [
      {
        id: "system-architecture",
        title: "System Architecture",
        html: String.raw`<p>The OpenISAC testbed comprises a Base Station (BS) and a User Equipment (UE), each built around a Universal Software Radio Peripheral (USRP) synchronized by an oven-controlled crystal oscillator (OCXO).</p>
<figure class="inline-figure">
  <img src="images/SysArch.png" alt="System Architecture Diagram">
  <figcaption>Fig. 1. System architecture of OpenISAC.</figcaption>
</figure>
<p><strong>BS Node:</strong> A host PC connects to the USRP over USB/Ethernet, generates the ISAC baseband waveform, and streams this to the USRP. The USRP transmits the signal and captures the radar echoes on a separate receive antenna.</p>
<p><strong>UE Node:</strong> A host PC interfaces with the USRP to acquire the downlink signal. It uses an OCXO for synchronization, optionally disciplined via a DAC to minimize carrier/sampling offsets for bistatic sensing.</p>`,
      },
      {
        id: "bs-software",
        title: "Software Architecture of BS",
        html: String.raw`<p>The BS software is a multi-threaded pipeline that decouples I/O and computation using ring-buffer FIFOs.</p>
<figure class="inline-figure">
  <img src="images/SoftArchBS.png" alt="BS Software Diagram">
  <figcaption>Fig. 2. Software architecture of BS.</figcaption>
</figure>
<ul>
  <li><strong>Bit Processing:</strong> Handles UDP payloads, LDPC encoding, and scrambling.</li>
  <li><strong>OFDM Modulator:</strong> Performs QPSK mapping, pilot insertion, IFFT, and CP insertion. Pads frames with random bits if traffic is low.</li>
  <li><strong>Radio I/O:</strong> "USRP-TX" sends waveforms; "USRP-RX" captures radar streams.</li>
  <li><strong>Sensing Thread:</strong> Performs real-time monostatic sensing (OFDM demod, division, Range-Doppler map). Supports "stride" processing to balance load.</li>
</ul>`,
      },
      {
        id: "ue-software",
        title: "Software Architecture of UE",
        html: String.raw`<p>The UE is also a multi-threaded pipeline designed for robust synchronization and reception.</p>
<figure class="inline-figure">
  <img src="images/SoftArchUE.png" alt="UE Software Diagram">
  <figcaption>Fig. 3. Software architecture of UE.</figcaption>
</figure>
<ul>
  <li><strong>USRP RX:</strong> Acquires downlink baseband stream and performs timing adjustments.</li>
  <li><strong>OFDM Demodulator:</strong> Operates in two states:
    <ul>
      <li><em>SYNC_SEARCH:</em> Scans for ZC sync symbols to estimate frame boundary and CFO.</li>
      <li><em>NORMAL:</em> Performs FFT, channel estimation, equalization, and LLR computation. Re-enters search if lock is lost.</li>
    </ul>
  </li>
  <li><strong>Sensing Thread:</strong> Performs real-time bistatic sensing using {RX, TX} symbol pairs.</li>
  <li><strong>Bit Processing:</strong> Descrambles and LDPC decodes to recover the UDP payload.</li>
</ul>`,
      },
    ],
  },
  zh: {
    title: "系统架构 - OpenISAC",
    description: "OpenISAC 系统架构说明。",
    sections: [
      {
        id: "system-architecture",
        title: "系统架构",
        html: String.raw`<p>OpenISAC 实验平台包括一个基站 (BS) 和一个用户设备 (UE)，每个设备都基于USRP 构建，并由恒温晶振 (OCXO) 提供参考时钟。</p>
<figure class="inline-figure">
  <img src="images/SysArch.png" alt="系统架构图">
  <figcaption>图 1. OpenISAC 系统架构。</figcaption>
</figure>
<p><strong>BS 节点：</strong> 主控 PC 通过 USB/以太网连接 USRP，生成 ISAC 基带波形，并将其传输到 USRP。USRP 发送信号并在单独的接收天线上接收雷达回波。</p>
<p><strong>UE 节点：</strong> 主控 PC 与 USRP 通信以获取下行链路信号。USRP 使用 OCXO 作为参考时钟，可选择通过 DAC 进行驯服，以最小化双站感知的载波/采样频率偏差。</p>`,
      },
      {
        id: "bs-software",
        title: "BS 软件架构",
        html: String.raw`<p>BS 软件是一个多线程流水线，使用环形缓冲区 FIFO 将 I/O 与计算解耦。</p>
<figure class="inline-figure">
  <img src="images/SoftArchBS.png" alt="BS 软件架构图">
  <figcaption>图 2. BS 软件架构。</figcaption>
</figure>
<ul>
  <li><strong>比特处理：</strong> 处理 UDP 负载、LDPC 编码和加扰。</li>
  <li><strong>OFDM 调制器：</strong> 执行 QPSK 映射、导频插入、IFFT 和 CP 插入。如果流量较低，则用随机比特填充。</li>
  <li><strong>无线电 I/O：</strong> “USRP-TX”发送波形；“USRP-RX”接收雷达回波。</li>
  <li><strong>感知线程：</strong> 执行实时单站感知（OFDM 解调、除法、距离-多普勒图）。支持稀疏处理以降低运算负担。</li>
</ul>`,
      },
      {
        id: "ue-software",
        title: "UE 软件架构",
        html: String.raw`<p>UE 也是一个多线程流水线，专为鲁棒的同步和接收而设计。</p>
<figure class="inline-figure">
  <img src="images/SoftArchUE.png" alt="UE 软件架构图">
  <figcaption>图 3. UE 软件架构。</figcaption>
</figure>
<ul>
  <li><strong>USRP RX：</strong> 获取下行基带信号并执行定时调整。</li>
  <li><strong>OFDM 解调器：</strong> 在两种状态下运行：
    <ul>
      <li><em>SYNC_SEARCH：</em> 搜索 ZC 同步符号以估计帧边界和 CFO。</li>
      <li><em>NORMAL：</em> 执行 FFT、信道估计、均衡和 LLR 计算。如果失去锁定，则重新进入搜索状态。</li>
    </ul>
  </li>
  <li><strong>感知线程：</strong> 使用 {RX, TX} 符号对进行实时双站感知。</li>
  <li><strong>比特处理：</strong> 解扰和 LDPC 解码以恢复 UDP 负载。</li>
</ul>`,
      },
    ],
  },
};

export const signalContent = {
  en: {
    title: "Signal Processing - OpenISAC",
    description: "Signal processing overview of OpenISAC.",
    sections: [
      {
        id: "signal-model",
        title: "Signal Model",
        html: String.raw`<p>The BS-UE channel is modeled as:</p>
<div class="equation">$$ {h}_{\mathrm{UE}}\left( t,\tau \right) =\sum_{l=1}^L{\alpha _l\delta \left( \tau -\tau _l-\tau _d \right)
e^{j2\pi \left( f_{D,l}+\Delta f_c \right) t}} $$</div>
<p>where \(L\) is the number of multi-path components, \(\tau_d\) is the timing offset, and \(\Delta f_c\) is the carrier-frequency offset.</p>
<p>The monostatic sensing channel is:</p>
<div class="equation">$$ {h}_{\mathrm{BS}}\left( t,\tau \right) =\sum_{p=1}^P{\beta _{p}\delta \left( \tau -\tau _{s,p} \right)
e^{j2\pi f_{D,s,p} t}} $$</div>
<h3>Continuous Wave vs. Packet Radio</h3>
<p>OpenISAC adopts a continuous-wave transmission scheme to enable more accurate and flexible Doppler sensing, avoiding the jitter and irregular intervals of packet-based systems (like Wi-Fi).</p>
<figure class="inline-figure">
  <img src="images/PacketVSCW.png" alt="Packet Radio vs Continuous Wave">
  <figcaption>Fig. 4. Comparison between packet radio and continuous wave radio.</figcaption>
</figure>`,
      },
      {
        id: "monostatic-sensing",
        title: "Monostatic Sensing",
        html: String.raw`<figure class="inline-figure">
  <img src="images/FlowGraph.png" alt="Monostatic Sensing Flow Graph">
  <figcaption>Fig. 5. Signal processing procedure for monostatic sensing.</figcaption>
</figure>
<p>The processing pipeline includes:</p>
<ol>
  <li><strong>OFDM Demodulation & Element-wise Division:</strong> Removes the influence of transmitted data symbols to obtain the TF channel matrices.
    <div class="equation">$$ \left( \boldsymbol{F}_{\mathrm{BS},\gamma} \right) _{n,m}=\frac{\left(
    \boldsymbol{B}_{\mathrm{BS},\gamma} \right) _{n,m}}{b_{n,m,\gamma}} $$</div>
  </li>
  <li><strong>Clutter Suppression:</strong> Applies an improved Moving Target Indication (MTI) procedure (IIR High-pass filter) to suppress static clutter.
    <div class="equation">$$ ( \tilde{\boldsymbol{F}}_{\mathrm{BS}} )_{n,m} = \frac{1}{a_0} \left( \sum_{i=0}^{I} b_i (
    \grave{\boldsymbol{F}}_{\mathrm{BS}} )_{n,m-i} - \sum_{j=1}^{J} a_j (
    \tilde{\boldsymbol{F}}_{\mathrm{BS}} )_{n,m-j} \right) $$</div>
  </li>
  <li><strong>Delay-Doppler Processing:</strong> Computes the periodogram to estimate target range and velocity.
    <div class="equation">$$ \left(\mathrm{Per}_{\gamma}\right)_{k_{\tau},k_{f}} =\frac{1}{N M_s} \left|
    \sum_{m=0}^{M_s-1}\sum_{n=0}^{N-1} (\tilde{\boldsymbol{F}}_{\mathrm{BS},\gamma})_{n,m}\, w[n,m]\,
    e^{j2\pi \frac{n k_{\tau}}{N_{\mathrm{Per}}}} e^{-j2\pi \frac{m k_{f}}{M_{\mathrm{Per}}}} \right|^2 $$</div>
  </li>
  <li><strong>Micro-Doppler Processing:</strong>
    <p>After clutter suppression, micro-Doppler analysis operates directly on the slow-time stream per range bin. First, form the delay-time matrix via an IFFT:</p>
    <div class="equation">$$ \left(\boldsymbol{R}_{\mathrm{BS}}\right)_{k_{\tau},m}
    =\frac{1}{N}\sum_{n=0}^{N-1} \left(\tilde{\boldsymbol{F}}_{\mathrm{BS}}\right)_{n,m}\,e^{\,j2\pi
    \frac{n k_{\tau}}{N}} $$</div>
    <p>Then, select a working range bin \(k_{\tau}^\star\) and compute the Short-Time Fourier Transform (STFT):</p>
    <div class="equation">$$ \left(\boldsymbol{G}\right)_{m,k_f}
    =\sum_{\ell=0}^{M_w-1} r_{\mathrm{BS}}\!\left[mM_H+\ell\right]\; w_\mathrm{md}[\ell]\;
    e^{-j2\pi \frac{k_f\,\ell}{M_{\mathrm{md}}}} $$</div>
    <p>The spectrogram is then calculated as:</p>
    <div class="equation">$$ \left( \mathrm{SPT} \right) _{m,k_f}=\frac{1}{M_w}\left| \left( \boldsymbol{G} \right) _{m,k_f}
    \right|^2 $$</div>
  </li>
</ol>`,
      },
      {
        id: "ue-reception",
        title: "UE Communication Reception",
        html: String.raw`<figure class="inline-figure">
  <img src="images/FlowGraph_UE.png" alt="UE Processing Flow Graph">
  <figcaption>Fig. 6. Block diagram of UE communication reception.</figcaption>
</figure>
<p>The UE operates in two states:</p>
<h3>SYNC_SEARCH State</h3>
<p>The UE operates in a block-by-block fashion. In each iteration, it fetches a block of samples and performs the following:</p>
<ol>
  <li><strong>Frame Detection & Timing Estimation:</strong> Performs a sliding correlation with the known Zadoff-Chu (ZC) synchronization symbol \(s_{\mathrm{ZC}}[k]\):
    <div class="equation">$$ r[k]=\sum_{i=0}^{N_s-1}{y_{\mathrm{UE},\mathrm{sync}}\left[ k+i \right] s_{\mathrm{ZC}}^{*}\left[
    i \right]} $$</div>
    <p>A peak in the normalized correlation energy \(r_N[k]\) indicates the frame boundary and initial timing offset \(\hat{k}_{\mathrm{TO}}\):</p>
    <div class="equation">$$ r_N[k]=\frac{|r[k]|^2}{\sum_{n=0}^{N_{\mathrm{corr}}}|r[k]|^2} $$</div>
  </li>
  <li><strong>Coarse Frequency Offset Estimation:</strong> Uses CP-tail correlations across multiple symbols to estimate the fractional frequency offset \(\hat{f}_o\):
    <div class="equation">$$ \hat{f}_o=\frac{\mathrm{arg}\!\bigl( r_{\mathrm{CP}} \bigr)}{2\pi T} $$</div>
  </li>
  <li><strong>Correction:</strong> The estimated offsets are corrected via digital frequency retuning or OCXO adjustment, and sample alignment (padding/discarding).</li>
</ol>
<h3>NORMAL State</h3>
<p>Once synchronized, the UE transitions to the NORMAL state to process frames:</p>
<ol>
  <li><strong>OFDM Demodulation:</strong> Removes CPs and performs FFT to obtain frequency-domain symbols:
    <div class="equation">$$ \left( \boldsymbol{B}_{\mathrm{UE},\gamma} \right)_{n,m} = b_{n,m,\gamma} \left(
    \boldsymbol{H}_{\mathrm{UE},\gamma} \right)_{n,m} + \left( \boldsymbol{Z}_{\mathrm{UE},\gamma}
    \right)_{n,m} $$</div>
  </li>
  <li><strong>Channel Estimation:</strong> Uses the ZC symbol to estimate the channel response:
    <div class="equation">$$ (\hat{\boldsymbol{H}}_{\mathrm{UE},\gamma})_{n,m_{\mathrm{sync}}}=\frac{\left(
    \boldsymbol{B}_{\mathrm{UE},\gamma} \right) _{n,m_{\mathrm{sync}}}}{z_n} $$</div>
    <p>The full-frame channel estimates are obtained by propagating the estimate at the synchronization symbol \(m_{\mathrm{sync}}\):</p>
    <div class="equation">$$
    (\hat{\boldsymbol{H}}_{\mathrm{UE},\gamma})_{n,m}=(\hat{\boldsymbol{H}}_{\mathrm{UE},\gamma})_{n,m_{\mathrm{sync}}}\exp
    \bigl( j2\pi ( m-m_{\mathrm{sync}} )( \hat{f}_{o,\gamma}T_O-n\Delta fN_s\Delta \hat{T}_{s,\gamma} )
    \bigr) $$</div>
  </li>
  <li><strong>CFO/SFO Tracking:</strong> Pilots are used to track residual carrier frequency offset (CFO) and sampling frequency offset (SFO) via Weighted Linear Regression (WLS) on pilot phase errors:
    <div class="equation">$$ \hat{\boldsymbol{\theta}}_\gamma = \big( \boldsymbol{A}_\gamma^{{T}} \boldsymbol{W}_\gamma
    \boldsymbol{A}_\gamma \big)^{-1} \boldsymbol{A}_\gamma^{{T}} \boldsymbol{W}_\gamma
    \boldsymbol{\varphi}_{{UE},\gamma} $$</div>
  </li>
  <li><strong>Equalization & Decoding:</strong> Performs channel equalization via one-tap frequency-domain equalization:
    <div class="equation">$$ \hat{b}_{n,m,\gamma}=\frac{\left( \boldsymbol{B}_{\mathrm{UE},\gamma} \right)
    _{n,m}}{(\hat{\boldsymbol{H}}_{\mathrm{UE},\gamma})_{n,m}} $$</div>
    <p>Then, LLRs are computed, descrambled, and LDPC-decoded to recover the payload.</p>
  </li>
</ol>`,
      },
      {
        id: "ue-bistatic-sensing",
        title: "UE Bistatic Sensing",
        html: String.raw`<p>Signal processing for bistatic sensing involves reconstructing unknown modulation symbols and performing Over-the-Air (OTA) synchronization.</p>
<h3>Modulation Symbol Reconstruction</h3>
<p>The reconstructed QPSK data symbol is obtained by making hard decisions on the equalized symbols \(\hat{b}_{n,m,\gamma}\):</p>
<div class="equation">$$ \tilde{b}_{n,m,\gamma} = \frac{1}{\sqrt{2}}\big( \operatorname{sgn}(\mathrm{Re}\{\hat{b}_{n,m,\gamma}\})
+ j\operatorname{sgn}(\mathrm{Im}\{\hat{b}_{n,m,\gamma}\}) \big) $$</div>
<h3>OTA Synchronization</h3>
<p>Bistatic sensing requires robust real-time synchronization without a wired link. OpenISAC implements a low-complexity over-the-air (OTA) synchronization scheme:</p>
<ol>
  <li><strong>Fractional Timing Estimation:</strong> Refines the timing offset using Quinn's algorithm to obtain a fractional estimate \(\hat{\delta}_{\tau}\) from the delay-domain peak. The overall timing offset is estimated as:
    <div class="equation">$$ \hat{\tau}_{o,\gamma} = \frac{\hat{\delta}_{\tau}+k_{\max ,\gamma}}{f_s} $$</div>
  </li>
  <li><strong>SIO Tracking:</strong> Estimates the Sampling Interval Offset (SIO) \(\epsilon_{\mathrm{SIO},w}\) by performing linear regression on the timing corrections over a window of \(\Gamma_W\) frames:
    <div class="equation">$$ \tilde{k}_{\tau,\gamma_w+\ell} \approx \epsilon_{\mathrm{SIO},w}\,\ell + \hat{k}_{\tau,\gamma_w} $$</div>
  </li>
  <li><strong>Recursive Update:</strong> Maintains a smooth estimate of the cumulative sensing timing offset \(\hat{k}^{\mathrm{sens}}_{\tau,\gamma}\) to avoid jitter:
    <div class="equation">$$ \hat{k}^{\mathrm{sens}}_{\tau,\gamma} = \hat{k}^{\mathrm{sens}}_{\tau,\gamma-1} +
    \hat{\epsilon}_{\mathrm{SIO},w-1} - \hat{k}_{\mathrm{TO},\gamma-1} + \mu_\gamma e_\gamma $$</div>
    <p>where \(\mu_\gamma e_\gamma\) is a feedback correction term based on the tracking error.</p>
  </li>
  <li><strong>Channel Compensation:</strong> The estimated timing offset and SIO are applied to the bistatic channel matrix to compensate for synchronization errors:
    <div class="equation">$$ \left( \tilde{\boldsymbol{F}}_{\mathrm{UE},\gamma} \right) _{n,m} = \left(
    \boldsymbol{F}_{\mathrm{UE},\gamma} \right) _{n,m} e^{j2\pi n\Delta f\left( \hat{k}_{\tau
    ,\gamma}^{\mathrm{sens}}+mN_s\Delta \hat{T}_{as,w-1} \right)} $$</div>
  </li>
</ol>`,
      },
    ],
  },
  zh: {
    title: "信号处理 - OpenISAC",
    description: "OpenISAC 信号处理说明。",
    sections: [
      {
        id: "signal-model",
        title: "信号模型",
        html: String.raw`<p>BS-UE 信道建模为：</p>
<div class="equation">$$ {h}_{\mathrm{UE}}\left( t,\tau \right) =\sum_{l=1}^L{\alpha _l\delta \left( \tau -\tau _l-\tau _d \right)
e^{j2\pi \left( f_{D,l}+\Delta f_c \right) t}} $$</div>
<p>其中 \(L\) 是多径分量的数量，\(\tau_d\) 是定时偏差，\(\Delta f_c\) 是载波频率偏差。</p>
<p>单站感知信道为：</p>
<div class="equation">$$ {h}_{\mathrm{BS}}\left( t,\tau \right) =\sum_{p=1}^P{\beta _{p}\delta \left( \tau -\tau _{s,p} \right)
e^{j2\pi f_{D,s,p} t}} $$</div>
<h3>连续波与分组无线电</h3>
<p>OpenISAC 采用连续波传输方案，以实现更精确和灵活的多普勒感知，避免了基于分组的系统（如 Wi-Fi）的抖动和不规则间隔。</p>
<figure class="inline-figure">
  <img src="images/PacketVSCW.png" alt="Packet Radio vs Continuous Wave">
  <figcaption>图 4. 分组无线电与连续波无线电的比较。</figcaption>
</figure>`,
      },
      {
        id: "monostatic-sensing",
        title: "单站感知",
        html: String.raw`<figure class="inline-figure">
  <img src="images/FlowGraph.png" alt="Monostatic Sensing Flow Graph">
  <figcaption>图 5. 单站感知的信号处理流程。</figcaption>
</figure>
<p>处理流程包括：</p>
<ol>
  <li><strong>OFDM 解调与逐元素除法：</strong> 消除发送数据符号的影响以获得时频信道矩阵。
    <div class="equation">$$ \left( \boldsymbol{F}_{\mathrm{BS},\gamma} \right) _{n,m}=\frac{\left(
    \boldsymbol{B}_{\mathrm{BS},\gamma} \right) _{n,m}}{b_{n,m,\gamma}} $$</div>
  </li>
  <li><strong>杂波抑制：</strong> 应用改进的动目标显示 (MTI) 算法（IIR 高通滤波器）以抑制静态杂波。
    <div class="equation">$$ ( \tilde{\boldsymbol{F}}_{\mathrm{BS}} )_{n,m} = \frac{1}{a_0} \left( \sum_{i=0}^{I} b_i (
    \grave{\boldsymbol{F}}_{\mathrm{BS}} )_{n,m-i} - \sum_{j=1}^{J} a_j (
    \tilde{\boldsymbol{F}}_{\mathrm{BS}} )_{n,m-j} \right) $$</div>
  </li>
  <li><strong>时延-多普勒处理：</strong> 计算周期图以估计目标距离和速度。
    <div class="equation">$$ \left(\mathrm{Per}_{\gamma}\right)_{k_{\tau},k_{f}} =\frac{1}{N M_s} \left|
    \sum_{m=0}^{M_s-1}\sum_{n=0}^{N-1} (\tilde{\boldsymbol{F}}_{\mathrm{BS},\gamma})_{n,m}\, w[n,m]\,
    e^{j2\pi \frac{n k_{\tau}}{N_{\mathrm{Per}}}} e^{-j2\pi \frac{m k_{f}}{M_{\mathrm{Per}}}} \right|^2 $$</div>
  </li>
  <li><strong>微多普勒处理：</strong>
    <p>在杂波抑制之后，微多普勒分析直接在每个距离单元的慢时间流上进行。首先，通过 IFFT 形成时延-时间矩阵：</p>
    <div class="equation">$$ \left(\boldsymbol{R}_{\mathrm{BS}}\right)_{k_{\tau},m}
    =\frac{1}{N}\sum_{n=0}^{N-1} \left(\tilde{\boldsymbol{F}}_{\mathrm{BS}}\right)_{n,m}\,e^{\,j2\pi
    \frac{n k_{\tau}}{N}} $$</div>
    <p>然后，选择一个工作距离单元 \(k_{\tau}^\star\) 并计算短时傅里叶变换 (STFT)：</p>
    <div class="equation">$$ \left(\boldsymbol{G}\right)_{m,k_f}
    =\sum_{\ell=0}^{M_w-1} r_{\mathrm{BS}}\!\left[mM_H+\ell\right]\; w_\mathrm{md}[\ell]\;
    e^{-j2\pi \frac{k_f\,\ell}{M_{\mathrm{md}}}} $$</div>
    <p>然后计算微多普勒谱：</p>
    <div class="equation">$$ \left( \mathrm{SPT} \right) _{m,k_f}=\frac{1}{M_w}\left| \left( \boldsymbol{G} \right) _{m,k_f}
    \right|^2 $$</div>
  </li>
</ol>`,
      },
      {
        id: "ue-reception",
        title: "UE 通信接收",
        html: String.raw`<figure class="inline-figure">
  <img src="images/FlowGraph_UE.png" alt="UE Processing Flow Graph">
  <figcaption>图 6. UE 通信接收框图。</figcaption>
</figure>
<p>UE 在两种状态下运行：</p>
<h3>SYNC_SEARCH 状态</h3>
<p>UE 以逐块方式运行。在每次迭代中，它获取一块样本并执行以下操作：</p>
<ol>
  <li><strong>帧检测与定时估计：</strong> 与已知的 Zadoff–Chu (ZC) 同步符号 \(s_{\mathrm{ZC}}[k]\) 进行滑动相关：
    <div class="equation">$$ r[k]=\sum_{i=0}^{N_s-1}{y_{\mathrm{UE},\mathrm{sync}}\left[ k+i \right] s_{\mathrm{ZC}}^{*}\left[
    i \right]} $$</div>
    <p>归一化相关能量 \(r_N[k]\) 中的峰值指示帧边界和初始定时偏差 \(\hat{k}_{\mathrm{TO}}\)：</p>
    <div class="equation">$$ r_N[k]=\frac{|r[k]|^2}{\sum_{n=0}^{N_{\mathrm{corr}}}|r[k]|^2} $$</div>
  </li>
  <li><strong>粗频偏估计：</strong> 利用跨越多个符号的 CP 尾部相关性来估计分数倍频偏 \(\hat{f}_o\)：
    <div class="equation">$$ \hat{f}_o=\frac{\mathrm{arg}\!\bigl( r_{\mathrm{CP}} \bigr)}{2\pi T} $$</div>
  </li>
  <li><strong>校正：</strong> 通过数字频率调节或 OCXO 调整以及样本对齐（填充/丢弃）来校正估计的偏差。</li>
</ol>
<h3>NORMAL 状态</h3>
<p>一旦同步，UE 转换到 NORMAL 状态以处理帧：</p>
<ol>
  <li><strong>OFDM 解调：</strong> 移除 CP 并执行 FFT 以获得频域符号：
    <div class="equation">$$ \left( \boldsymbol{B}_{\mathrm{UE},\gamma} \right)_{n,m} = b_{n,m,\gamma} \left(
    \boldsymbol{H}_{\mathrm{UE},\gamma} \right)_{n,m} + \left( \boldsymbol{Z}_{\mathrm{UE},\gamma}
    \right)_{n,m} $$</div>
  </li>
  <li><strong>信道估计：</strong> 使用 ZC 符号估计信道响应：
    <div class="equation">$$ (\hat{\boldsymbol{H}}_{\mathrm{UE},\gamma})_{n,m_{\mathrm{sync}}}=\frac{\left(
    \boldsymbol{B}_{\mathrm{UE},\gamma} \right) _{n,m_{\mathrm{sync}}}}{z_n} $$</div>
    <p>全帧信道估计通过对同步符号 \(m_{\mathrm{sync}}\) 处的估计插值获得：</p>
    <div class="equation">$$
    (\hat{\boldsymbol{H}}_{\mathrm{UE},\gamma})_{n,m}=(\hat{\boldsymbol{H}}_{\mathrm{UE},\gamma})_{n,m_{\mathrm{sync}}}\exp
    \bigl( j2\pi ( m-m_{\mathrm{sync}} )( \hat{f}_{o,\gamma}T_O-n\Delta fN_s\Delta \hat{T}_{s,\gamma} )
    \bigr) $$</div>
  </li>
  <li><strong>CFO/SFO 跟踪：</strong> 通过导频相位误差的加权线性回归 (WLS) 来跟踪残留载波频率偏差 (CFO) 和采样频率偏差 (SFO)：
    <div class="equation">$$ \hat{\boldsymbol{\theta}}_\gamma = \big( \boldsymbol{A}_\gamma^{{T}} \boldsymbol{W}_\gamma
    \boldsymbol{A}_\gamma \big)^{-1} \boldsymbol{A}_\gamma^{{T}} \boldsymbol{W}_\gamma
    \boldsymbol{\varphi}_{{UE},\gamma} $$</div>
  </li>
  <li><strong>均衡与解码：</strong> 通过单抽头频域均衡实现信道均衡：
    <div class="equation">$$ \hat{b}_{n,m,\gamma}=\frac{\left( \boldsymbol{B}_{\mathrm{UE},\gamma} \right)
    _{n,m}}{(\hat{\boldsymbol{H}}_{\mathrm{UE},\gamma})_{n,m}} $$</div>
    <p>然后，计算 LLR，解扰并进行 LDPC 解码以恢复有效载荷。</p>
  </li>
</ol>`,
      },
      {
        id: "ue-bistatic-sensing",
        title: "UE 双站感知",
        html: String.raw`<p>双站感知的信号处理涉及重构未知的调制符号和执行空口 (OTA) 同步。</p>
<h3>调制符号重构</h3>
<p>重构的 QPSK 数据符号是通过对均衡后的符号 \(\hat{b}_{n,m,\gamma}\) 进行硬判决获得的：</p>
<div class="equation">$$ \tilde{b}_{n,m,\gamma} = \frac{1}{\sqrt{2}}\big( \operatorname{sgn}(\mathrm{Re}\{\hat{b}_{n,m,\gamma}\})
+ j\operatorname{sgn}(\mathrm{Im}\{\hat{b}_{n,m,\gamma}\}) \big) $$</div>
<h3>OTA 同步</h3>
<p>双站感知需要通过无线链路实现鲁棒实时同步。OpenISAC 实现了低复杂度的空口 (OTA) 同步方案：</p>
<ol>
  <li><strong>分数定时估计：</strong> 使用 Quinn 算法细化定时偏差，从时延域峰值获得分数估计 \(\hat{\delta}_{\tau}\)。总定时偏差估计为：
    <div class="equation">$$ \hat{\tau}_{o,\gamma} = \frac{\hat{\delta}_{\tau}+k_{\max ,\gamma}}{f_s} $$</div>
  </li>
  <li><strong>SIO 跟踪：</strong> 通过对 \(\Gamma_W\) 帧窗口上的定时偏差执行线性回归来估计采样间隔偏差 (SIO) \(\epsilon_{\mathrm{SIO},w}\)：
    <div class="equation">$$ \tilde{k}_{\tau,\gamma_w+\ell} \approx \epsilon_{\mathrm{SIO},w}\,\ell + \hat{k}_{\tau,\gamma_w} $$</div>
  </li>
  <li><strong>递归更新：</strong> 维护累积感知定时偏差 \(\hat{k}^{\mathrm{sens}}_{\tau,\gamma}\) 的平滑估计以避免抖动：
    <div class="equation">$$ \hat{k}^{\mathrm{sens}}_{\tau,\gamma} = \hat{k}^{\mathrm{sens}}_{\tau,\gamma-1} +
    \hat{\epsilon}_{\mathrm{SIO},w-1} - \hat{k}_{\mathrm{TO},\gamma-1} + \mu_\gamma e_\gamma $$</div>
    <p>其中 \(\mu_\gamma e_\gamma\) 是基于跟踪误差的反馈校正项。</p>
  </li>
  <li><strong>信道补偿：</strong> 将估计的定时偏差和 SIO 应用于双站信道矩阵以补偿同步误差：
    <div class="equation">$$ \left( \tilde{\boldsymbol{F}}_{\mathrm{UE},\gamma} \right) _{n,m} = \left(
    \boldsymbol{F}_{\mathrm{UE},\gamma} \right) _{n,m} e^{j2\pi n\Delta f\left( \hat{k}_{\tau
    ,\gamma}^{\mathrm{sens}}+mN_s\Delta \hat{T}_{as,w-1} \right)} $$</div>
  </li>
</ol>`,
      },
    ],
  },
};
