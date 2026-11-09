---
title: Docker 部署
description: 在容器中运行 OpenISAC，无需在宿主机安装 UHD、Aff3ct 等依赖。
---

OpenISAC Docker 镜像已包含 UHD、Aff3ct、BS、UE、ChannelSimulator 和 UdpLinkTest，宿主机无需安装或编译任何依赖。
容器仍直接使用宿主机的网卡、CPU 核心和 USB 设备，无线配置方式与原生安装相同；时序表现还取决于宿主机调度和调优。

## 环境要求

- **原生 Linux** 和 Docker Engine，推荐 Ubuntu 24.04。请使用标准安装的 Docker Engine，不要使用 rootless Docker。Docker Desktop 和 WSL 仅适合运行仿真。
- 支持 AVX2 的 x86-64 CPU（Intel Haswell、AMD Zen 或更新的型号）。

以下命令均在 OpenISAC 仓库根目录执行。当前账号不在 `docker` 组时，请在命令前加 `sudo`；容器生成的文件仍归属于你自己的用户。

## 快速上手：无 USRP 仿真

先构建一次 CPU 镜像（首次构建耗时较长）：

```bash
scripts/docker_build.sh cpu
```

然后在三个终端中依次启动信道仿真器、BS 和 UE：

```bash
# 终端 1：仿真无线信道
scripts/docker_run.sh ChannelSimulator /opt/openisac/config/BS_Sim.yaml

# 终端 2：基站
scripts/docker_run.sh BS sim

# 终端 3：用户端
scripts/docker_run.sh UE sim
```

在各终端按 `Ctrl+C` 停止。宿主机上的感知界面等 Python 工具连接容器的方式与连接原生程序完全相同，参见[感知界面操作指南](../../tools-workflows/sensing-viewers/)。

## 连接 USRP 运行

### 1. 配置宿主机网络

按照[首次 OTA 运行](../first-ota-run/)配置连接 USRP 的宿主机网卡（IP 地址、MTU）。使用 X310 时，还需提高宿主机的 socket 缓冲上限：

```bash
sudo sysctl -w net.core.rmem_max=50000000 net.core.wmem_max=50000000
```

该设置在重启后失效；如需保留，请将其写入 `/etc/sysctl.conf`。

### 2. 准备配置文件

将与硬件对应的预设复制到任意目录后再编辑（USRP 地址、频率、CPU 核心等）：

```bash
mkdir -p my-config
cp config/BS_X310.yaml my-config/BS.yaml
cp config/UE_X310.yaml my-config/UE.yaml
```

全部参数参见 [BS YAML 参考](../../reference/bs-yaml/)和 [UE YAML 参考](../../reference/ue-yaml/)。

### 3. 启动 BS 和 UE

```bash
scripts/docker_run.sh --config my-config/BS.yaml --name openisac-bs BS
scripts/docker_run.sh --config my-config/UE.yaml --name openisac-ue UE
```

使用说明：

- **修改配置后，下次启动生效。** 编辑 `my-config/` 中的文件并重启程序即可，容器不会修改你的文件。
- **日志和运行文件**写入当前目录下的 `docker-run/BS/` 和 `docker-run/UE/`。可用 `--work-dir DIR` 指定其他位置；同时运行的每个程序应使用各自的工作目录。
- **网络**与直接在宿主机上运行完全相同，无需端口映射。同时运行多个实例时，应在 YAML 中为每个实例设置不同的控制端口和显示端口。
- **YAML 中的 CPU 核心编号**即宿主机的 CPU 编号，与原生安装一致。

不使用 `--config` 时将采用内置预设：在程序名后加 `x310`（默认）、`b210`、`sim` 或 `sim-duplex`，例如 `scripts/docker_run.sh BS b210`。预设在首次运行时复制到工作目录，之后的运行沿用该副本。如需切换预设，请删除工作目录中的 `BS.yaml`/`UE.yaml`，或改用新的工作目录。

### B210（USB）

加上 `--usb`，使容器可以访问 USB 设备：

```bash
scripts/docker_run.sh --usb --config my-config/BS.yaml BS
```

宿主机必须允许当前用户通过 USB 访问 USRP，例如安装 UHD 时附带的 udev 规则。以太网 USRP 不需要 `--usb`。

## 实时性能

OpenISAC 稳定运行需要实时调度。启动脚本会自动授予该权限，各程序启动时也会进行检查。也可以手动检查：

```bash
scripts/docker_run.sh doctor
```

检查同样会拒绝容器无权使用的 YAML CPU 核心。如果因实时调度不可用而停止启动，且只需非实时的开发运行（例如在笔记本上），可设置 `OPENISAC_REQUIRE_RT=0` 启动：

```bash
OPENISAC_REQUIRE_RT=0 scripts/docker_run.sh BS sim
```

如需最稳定的时序，请先在宿主机按照[系统性能调优](../performance-tuning/)和 [CPU 隔离和执行](../cpu-isolation-execution/)操作，再将保存的隔离计划传给启动脚本：

```bash
sudo scripts/isolate_cpus.py          # 在宿主机上隔离核心（见上述链接）
scripts/docker_run.sh --isolation-plan /tmp/isolate_cpus_state.json \
  --config my-config/BS.yaml BS
```

启动容器时请使用 `--isolation-plan`，不要使用 `isolate_cpus.py run`。如果只想限制容器可用的 CPU，而不阻止其他程序使用这些 CPU，可使用 `--cpuset-cpus`，例如 `--cpuset-cpus 2-11`。

## USRP 固件兼容性

镜像使用 **UHD 4.11**。BS 或 UE 启动前，会检查配置中的 USRP 是否与该版本兼容。也可以不启动 OpenISAC 单独检查某台 USRP：

```bash
scripts/docker_run.sh usrp-check --args addr=192.168.30.2
```

如果 **X300/X310** 的 FPGA 镜像不兼容，会在 OpenISAC 启动前**自动更新**为匹配的官方镜像：

- 更新前先备份当前镜像，保存在 `~/.local/state/openisac/firmware/<序列号>/`（可用 `--firmware-state DIR` 指定其他位置）。如果以后可能需要恢复旧镜像，请保留该备份。
- **更新过程中不要断电或断开 USRP**，更新需要几分钟。部分设备在更新后需要断电重启，届时会有提示。
- 更新后，USRP 可能无法再配合其他地方安装的旧版 UHD 使用。
- 兼容的设备不会被改动；无法可靠识别设备时也不会写入。若一个设备字符串中包含多台 X310（`addr0=...,addr1=...`），请先对每个地址分别执行 `usrp-check`。
- 更新失败后，该 USRP 的自动更新会暂停。请将设备断电重启后再次启动。若仍不兼容，设备备份目录中的 `transaction.json` 包含恢复已保存镜像的命令。设备恢复正常后，将该文件移走即可重新启用自动更新。

如需关闭自动更新（例如使用自定义 FPGA 镜像时），请在启动时设置 `OPENISAC_AUTO_UPDATE_FIRMWARE=0`；此时遇到不兼容的设备会报错并停止启动。B200/B210 无需更新：每次打开设备时，UHD 都会加载其固件。

## 使用自定义配置仿真

将仿真 YAML（例如 `config/BS_Sim.yaml` 和 `config/UE_Sim.yaml` 的副本）放入一个目录，然后用它们启动三个程序：

```bash
scripts/docker_run.sh --config sim-config/BS.yaml ChannelSimulator /config/BS.yaml
scripts/docker_run.sh --config sim-config/BS.yaml BS
scripts/docker_run.sh --config sim-config/UE.yaml UE
```

在容器内，该目录位于 `/config/`。同时运行的多组仿真需要使用不同的 `simulation.session` 名称和端口。全部仿真参数参见[信道仿真器](../../tools-workflows/channel-simulator/)。

## 参考

### 启动脚本选项

`scripts/docker_run.sh [选项] 程序 [参数]`

| 选项 | 作用 |
| --- | --- |
| `--config FILE` | YAML 配置文件，每次启动重新读取 |
| `--work-dir DIR` | 日志和运行文件目录（默认 `docker-run/程序名`） |
| `--name NAME` | 容器名称，可用于 `docker stop` |
| `--usb` | 允许访问 USB USRP（B200/B210） |
| `--isolation-plan FILE` | 使用 `isolate_cpus.py` 保存的 CPU 集合 |
| `--cpuset-cpus LIST` | 限制容器可用的 CPU |
| `--memory SIZE` | 限制容器内存，例如 `4g`（默认不限制） |
| `--firmware-state DIR` | USRP 固件备份目录 |
| `--uhd-images DIR` | 使用宿主机目录中的 UHD 固件/FPGA 镜像 |
| `--image IMAGE` | 使用其他镜像代替 `openisac:cpu` |
| `--dry-run` | 只显示 Docker 命令，不执行 |

| 环境变量 | 作用 |
| --- | --- |
| `OPENISAC_REQUIRE_RT=0` | 允许在没有实时调度时启动 |
| `OPENISAC_AUTO_UPDATE_FIRMWARE=0` | 关闭 X300/X310 FPGA 自动更新 |

### 构建选项

`scripts/docker_build.sh cpu` 支持以下环境变量：

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `CPU_ARCH` | `x86-64-v3` | CPU 指令集；仅当镜像在与构建机相同型号的 CPU 上运行时才使用 `native` |
| `BUILD_JOBS` | `4` | 并行构建任务数；构建时内存不足可调低 |
| `UHD_REF` | `v4.11.0.0` | 镜像包含的 UHD 版本 |

### 停止

按 `Ctrl+C`，或对使用 `--name` 启动的容器执行 `docker stop NAME`。如果正在更新固件，容器会等待更新完成后再退出。

### 常见问题

| 现象 | 解决方法 |
| --- | --- |
| `Real-time prerequisites unavailable` | 在原生 Linux 上使用标准（rootful）Docker Engine；开发运行可设置 `OPENISAC_REQUIRE_RT=0` |
| `Configured CPU IDs are outside the container CPU set` | YAML 使用了被 `--cpuset-cpus` 或隔离计划排除的 CPU 核心，请调整其中之一 |
| 找不到 USRP | 检查 YAML 中的 USRP 地址和宿主机网络设置；B210 需加 `--usb` |
| 频繁出现溢出（`O`）或欠载（`U`） | 应用上文的 socket 缓冲设置，并参照[系统性能调优](../performance-tuning/) |
| 仿真无法启动 | 先启动 `ChannelSimulator`，并确保所有配置使用相同的 `simulation.session` |
