import { defineConfig } from "astro/config";
import { unified } from "@astrojs/markdown-remark";
import starlight from "@astrojs/starlight";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";

const site = process.env.OPENISAC_SITE_URL ?? "https://openisac.zzw123app.top";
const base = process.env.OPENISAC_SITE_BASE ?? "/";

const sidebar = [
  {
    label: "Overview",
    translations: { "zh-CN": "概览" },
    items: [
      { label: "OpenISAC Overview", translations: { "zh-CN": "OpenISAC 概览" }, link: "/docs/" },
    ],
  },
  {
    label: "Getting Started",
    translations: { "zh-CN": "快速开始" },
    items: [
      { label: "Hardware", translations: { "zh-CN": "硬件准备" }, link: "/docs/getting-started/hardware/" },
      { label: "Installation", translations: { "zh-CN": "软件安装" }, link: "/docs/getting-started/installation/" },
      { label: "Build", translations: { "zh-CN": "编译" }, link: "/docs/getting-started/build/" },
      { label: "System Performance Tuning", translations: { "zh-CN": "系统性能调优" }, link: "/docs/getting-started/performance-tuning/" },
      { label: "CPU Isolation and Execution", translations: { "zh-CN": "CPU 隔离和执行" }, link: "/docs/getting-started/cpu-isolation-execution/" },
      { label: "First OTA Run", translations: { "zh-CN": "首次 OTA 运行" }, link: "/docs/getting-started/first-ota-run/" },
    ],
  },
  {
    label: "Architecture",
    translations: { "zh-CN": "系统架构" },
    items: [
      { label: "System Architecture", translations: { "zh-CN": "整体架构" }, link: "/docs/architecture/system/" },
      { label: "BS Runtime Pipeline", translations: { "zh-CN": "BS 运行流水线" }, link: "/docs/architecture/bs-pipeline/" },
      { label: "UE Runtime Pipeline", translations: { "zh-CN": "UE 运行流水线" }, link: "/docs/architecture/ue-pipeline/" },
      { label: "Frontend and Transport", translations: { "zh-CN": "前端与传输" }, link: "/docs/architecture/frontend-transport/" },
    ],
  },
  {
    label: "Signal Processing",
    translations: { "zh-CN": "信号处理" },
    items: [
      { label: "Processing Overview", translations: { "zh-CN": "处理总览" }, link: "/docs/signal-processing/" },
      { label: "Signal Model", translations: { "zh-CN": "信号模型" }, link: "/docs/signal-processing/signal-model/" },
      { label: "OFDM Resources", translations: { "zh-CN": "OFDM 资源" }, link: "/docs/signal-processing/ofdm-resources/" },
      { label: "Sync, CFO, and SFO", translations: { "zh-CN": "同步、CFO 与 SFO" }, link: "/docs/signal-processing/sync-cfo-sfo/" },
      { label: "Downlink Communication", translations: { "zh-CN": "下行通信" }, link: "/docs/signal-processing/ue-reception/" },
      { label: "Uplink Communication", translations: { "zh-CN": "上行通信" }, link: "/docs/signal-processing/uplink-communication/" },
      { label: "Multichannel Monostatic Sensing", translations: { "zh-CN": "多通道单站感知" }, link: "/docs/signal-processing/monostatic-sensing/" },
      { label: "UE Bistatic Sensing", translations: { "zh-CN": "UE 双站感知" }, link: "/docs/signal-processing/bistatic-sensing/" },
      { label: "OTA and eRTM Timing", translations: { "zh-CN": "OTA 与 eRTM 定时" }, link: "/docs/signal-processing/ota-ertm-timing/" },
    ],
  },
  {
    label: "Modules and Features",
    translations: { "zh-CN": "模块与功能" },
    items: [
      { label: "Channel Simulator", translations: { "zh-CN": "信道仿真器" }, link: "/docs/modules/channel-simulator/" },
      { label: "Web Config Console", translations: { "zh-CN": "网页配置台" }, link: "/docs/modules/web-config-console/" },
      { label: "Calibration Workflows", translations: { "zh-CN": "校准流程" }, link: "/docs/modules/calibration-workflows/" },
      { label: "Video and UDP Workflows", translations: { "zh-CN": "视频与 UDP 工作流" }, link: "/docs/modules/video-udp-workflows/" },
      { label: "macOS and Development Notes", translations: { "zh-CN": "macOS 与开发说明" }, link: "/docs/modules/macos-development/" },
    ],
  },
  {
    label: "Reference",
    translations: { "zh-CN": "参考" },
    items: [
      { label: "BS YAML Reference", translations: { "zh-CN": "BS YAML 参考" }, link: "/docs/reference/bs-yaml/" },
      { label: "UE YAML Reference", translations: { "zh-CN": "UE YAML 参考" }, link: "/docs/reference/ue-yaml/" },
      { label: "Scripts and Tools", translations: { "zh-CN": "脚本与工具" }, link: "/docs/reference/scripts-tools/" },
      { label: "Troubleshooting", translations: { "zh-CN": "故障排查" }, link: "/docs/reference/troubleshooting/" },
    ],
  },
];

export default defineConfig({
  site,
  base,
  build: {
    format: "directory",
  },
  markdown: {
    processor: unified({
      remarkPlugins: [remarkMath],
      rehypePlugins: [rehypeKatex],
    }),
  },
  integrations: [
    starlight({
      title: {
        en: "OpenISAC Docs",
        "zh-CN": "OpenISAC 文档",
      },
      logo: {
        light: "./public/images/logo.svg",
        dark: "./public/images/logo_light.svg",
        alt: "OpenISAC",
      },
      favicon: "/images/icon.svg",
      locales: {
        root: {
          label: "English",
          lang: "en",
        },
        "zh-cn": {
          label: "简体中文",
          lang: "zh-CN",
        },
      },
      defaultLocale: "root",
      sidebar,
      social: [
        {
          icon: "github",
          label: "GitHub",
          href: "https://github.com/zhouzhiwen2000/OpenISAC",
        },
      ],
      lastUpdated: false,
      credits: false,
      components: {
        ThemeProvider: "./src/components/StarlightThemeProvider.astro",
      },
      customCss: ["katex/dist/katex.min.css", "./src/styles/starlight.css"],
    }),
  ],
});
