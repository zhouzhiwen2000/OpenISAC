---
title: Docker deployment
description: Run OpenISAC in a container without installing UHD, Aff3ct or other dependencies on the host.
---

The OpenISAC Docker image contains UHD, Aff3ct, BS, UE, ChannelSimulator and UdpLinkTest, so nothing
needs to be installed or compiled on the host. The container still uses the host's
network interfaces, CPU cores and USB devices directly. It uses the same radio
configuration as a native installation; timing also depends on host scheduling and tuning.

## Requirements

- **Native Linux** with Docker Engine; Ubuntu 24.04 is recommended. Use the standard
  Docker Engine installation, not rootless Docker. Docker Desktop and WSL are suitable
  for simulation only.
- An x86-64 CPU with AVX2 support (Intel Haswell, AMD Zen or newer).

Run all commands below from the root of the OpenISAC repository. If your account is not
in the `docker` group, prefix them with `sudo`; files created by the container still
belong to your own user.

## Quick start: simulation without a USRP

Build the CPU image once (the first build takes a while):

```bash
scripts/docker_build.sh cpu
```

Then start the channel simulator, the BS and the UE in three terminals, in this order:

```bash
# Terminal 1: simulated radio channel
scripts/docker_run.sh ChannelSimulator /opt/openisac/config/BS_Sim.yaml

# Terminal 2: base station
scripts/docker_run.sh BS sim

# Terminal 3: user equipment
scripts/docker_run.sh UE sim
```

Press `Ctrl+C` in each terminal to stop. Sensing viewers and other Python tools on the
host connect to the containers exactly as they connect to native programs; see
[Sensing Viewer Guide](../../tools-workflows/sensing-viewers/).

## Running with a USRP

### 1. Prepare the host network

Configure the host network interface connected to the USRP (IP address, MTU) as described
in [First OTA Run](../first-ota-run/). For X310, also raise the host socket-buffer limits:

```bash
sudo sysctl -w net.core.rmem_max=50000000 net.core.wmem_max=50000000
```

This setting is lost after a reboot; add it to `/etc/sysctl.conf` to keep it.

### 2. Prepare your configuration files

Copy the presets matching your hardware into a folder of your choice, then edit them
(USRP addresses, frequencies, CPU cores and so on):

```bash
mkdir -p my-config
cp config/BS_X310.yaml my-config/BS.yaml
cp config/UE_X310.yaml my-config/UE.yaml
```

See the [BS YAML Reference](../../reference/bs-yaml/) and
[UE YAML Reference](../../reference/ue-yaml/) for every option.

### 3. Start BS and UE

```bash
scripts/docker_run.sh --config my-config/BS.yaml --name openisac-bs BS
scripts/docker_run.sh --config my-config/UE.yaml --name openisac-ue UE
```

What to expect:

- **Configuration changes take effect on the next start.** Edit the file in `my-config/`
  and restart the program. The container never modifies your file.
- **Logs and runtime files** are written to `docker-run/BS/` and `docker-run/UE/` in the
  current directory. Use `--work-dir DIR` to choose another location, and give every
  program running at the same time its own work directory.
- **Networking** works as if the program ran directly on the host; no port mapping is
  needed. When several instances run at the same time, give each one different control
  and viewer ports in YAML.
- **CPU core numbers in YAML** are the host's CPU numbers, as in a native installation.

Without `--config`, a built-in preset is used: add `x310` (default), `b210`, `sim` or
`sim-duplex` after the program name, for example `scripts/docker_run.sh BS b210`. The
preset is copied into the work directory on the first run, and later runs reuse that
copy. To switch to another preset, delete `BS.yaml`/`UE.yaml` from the work directory or
use a new work directory.

### B210 (USB)

Add `--usb` so the container can access USB devices:

```bash
scripts/docker_run.sh --usb --config my-config/BS.yaml BS
```

The host must allow your user to access the USRP over USB, for example through the udev
rules installed with UHD. Ethernet USRPs do not need `--usb`.

## Real-time performance

OpenISAC needs real-time scheduling for stable operation. The launcher grants it
automatically, and every program checks it at startup. To run the check yourself:

```bash
scripts/docker_run.sh doctor
```

The check also rejects YAML CPU cores that the container is not allowed to use. If
startup stops because real-time scheduling is unavailable and you only need a
non-real-time development run, for example on a laptop, start with
`OPENISAC_REQUIRE_RT=0`:

```bash
OPENISAC_REQUIRE_RT=0 scripts/docker_run.sh BS sim
```

For the most stable timing, follow [System Performance Tuning](../performance-tuning/)
and [CPU Isolation and Execution](../cpu-isolation-execution/) on the host, then pass the
saved isolation plan to the launcher:

```bash
sudo scripts/isolate_cpus.py          # isolate cores on the host (see the linked guide)
scripts/docker_run.sh --isolation-plan /tmp/isolate_cpus_state.json \
  --config my-config/BS.yaml BS
```

Use `--isolation-plan` rather than `isolate_cpus.py run` to start containers. To only
limit which CPUs a container may use, without keeping other programs off them, use
`--cpuset-cpus`, for example `--cpuset-cpus 2-11`.

## USRP firmware compatibility

The image uses **UHD 4.11**. Before the BS or UE starts, the USRPs in its configuration
are checked for compatibility with this version. To check a USRP without starting
OpenISAC:

```bash
scripts/docker_run.sh usrp-check --args addr=192.168.30.2
```

If a configured USRP reports an incompatible FPGA/firmware version, startup
**automatically invokes UHD's image loader for that device type**. This applies to
all configured devices for which the installed UHD provides an image loader,
including X3xx, X4xx, N3xx, E3xx, B2xx and N200/N210. The container includes the
release's FPGA/firmware image sets and their companion files. UHD selects the image
for the detected model and hardware revision; a configured `fpga=` selects the
FPGA flavor.

- Update records are kept in `~/.local/state/openisac/firmware/<serial>/`
  (choose another location with `--firmware-state DIR`). X300/X310 also save two
  matching readbacks of the current Flash image before writing. Other device
  loaders use their model-specific recovery procedures; these records are not
  backups of the device's Flash or operating system.
- **Do not power off or disconnect the USRP during the update**, which takes a few
  minutes. Some devices need a power cycle afterwards; you are told when this is the case.
- After the update, the USRP may no longer work with older UHD versions installed
  elsewhere.
- Compatible devices are never modified, and nothing is written when the device cannot
  be identified reliably. Numbered device arguments (`addr0=...,addr1=...`,
  `serial0=...,serial1=...`, or `mgmt_addr0=...,mgmt_addr1=...`) are checked and updated
  one motherboard at a time, followed by a check of the combined configuration.
- If an update fails, automatic updates for that USRP are paused. Power-cycle the device
  if the loader requests it and start again. If it is still incompatible, inspect
  `transaction.json` and the update log in the device's state folder. X300/X310
  records include a command to restore the saved image; other models require their
  UHD recovery procedure. Once the device works again, move the pending transaction
  file away to re-enable automatic updates.

**X4xx MPM version differences produce a prominent terminal warning**, including
the device's MPM software version and the container UHD version, or the expected
and actual compatibility versions reported by UHD. Software release differences
and compatible newer MPM minor versions permit startup with a warning. An
incompatible MPM version stops startup and
requires updating the device-side MPM/system image to match the container's UHD.
The FPGA/firmware updater does not install MPM or replace the device OS. The same
warning applies to other MPM-based devices.

To turn automatic updates off, for example when you use a custom FPGA image, start with
`OPENISAC_AUTO_UPDATE_FIRMWARE=0`; an incompatible device then stops startup with an
error instead. USB radios require `--usb`; UHD handles their normal volatile image
loading, and a reported FPGA mismatch can also invoke the device's image loader.
Some firmware or legacy-device updates require a power cycle or the device's
dedicated recovery tool, as reported by UHD.

When using `--uhd-images DIR` with automatic updates, supply the complete,
unmodified FPGA/firmware image set from the same container build. Companion DTS
and checksum files are checked along with the images before an update.

## Simulation with your own configuration

Put the simulation YAML files (for example copies of `config/BS_Sim.yaml` and
`config/UE_Sim.yaml`) in a folder, then start the three programs with them:

```bash
scripts/docker_run.sh --config sim-config/BS.yaml ChannelSimulator /config/BS.yaml
scripts/docker_run.sh --config sim-config/BS.yaml BS
scripts/docker_run.sh --config sim-config/UE.yaml UE
```

Inside the container, the folder appears as `/config/`. Simulations running at the same
time need different `simulation.session` names and ports. See
[Channel Simulator](../../tools-workflows/channel-simulator/) for all simulation options.

## Reference

### Launcher options

`scripts/docker_run.sh [options] PROGRAM [arguments]`

| Option | Purpose |
| --- | --- |
| `--config FILE` | YAML configuration, re-read on every start |
| `--work-dir DIR` | Folder for logs and runtime files (default `docker-run/PROGRAM`) |
| `--name NAME` | Container name, for use with `docker stop` |
| `--usb` | Allow access to USB USRPs |
| `--isolation-plan FILE` | Use the CPUs saved by `isolate_cpus.py` |
| `--cpuset-cpus LIST` | Limit the CPUs the container may use |
| `--memory SIZE` | Limit container memory, for example `4g` (unlimited by default) |
| `--firmware-state DIR` | Folder for USRP update records and supported Flash backups |
| `--uhd-images DIR` | Use UHD firmware/FPGA images from a host folder |
| `--image IMAGE` | Use another image instead of `openisac:cpu` |
| `--dry-run` | Print the Docker command without running it |

| Environment variable | Purpose |
| --- | --- |
| `OPENISAC_REQUIRE_RT=0` | Allow starting without real-time scheduling |
| `OPENISAC_AUTO_UPDATE_FIRMWARE=0` | Disable automatic USRP FPGA/firmware updates |

### Build options

`scripts/docker_build.sh cpu` accepts these environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `CPU_ARCH` | `x86-64-v3` | CPU instruction set; use `native` only if the image runs on the same CPU model it is built on |
| `BUILD_JOBS` | `4` | Parallel build jobs; lower it if the build runs out of memory |
| `UHD_REF` | `v4.11.0.0` | UHD release to include |
| `APT_MIRROR` | `official` | Ubuntu package source: retain the base image's sources (`official`), or select `ustc` or `tuna` |

For example, build using the USTC Ubuntu mirror:

```bash
APT_MIRROR=ustc scripts/docker_build.sh cpu
```

Use `APT_MIRROR=tuna` for Tsinghua's mirror. With `sudo`, use
`sudo env APT_MIRROR=ustc scripts/docker_build.sh cpu`. Direct `docker build` calls
accept `--build-arg APT_MIRROR=ustc` as well. The choice applies to Ubuntu APT packages
in every build stage and the runtime image; Git repositories and UHD image downloads
use their own upstream URLs. Mirror package downloads use HTTP so minimal base images
can install CA certificates, with APT signature verification enabled.

### Stopping

Press `Ctrl+C`, or run `docker stop NAME` for a container started with `--name`. If a
firmware update is in progress, the container finishes it before exiting.

### Troubleshooting

| Symptom | Solution |
| --- | --- |
| `Real-time prerequisites unavailable` | Use standard (rootful) Docker Engine on native Linux, or start with `OPENISAC_REQUIRE_RT=0` for development runs |
| `Configured CPU IDs are outside the container CPU set` | The YAML uses CPU cores excluded by `--cpuset-cpus` or the isolation plan; adjust one of them |
| USRP not found | Check the USRP address in YAML and the host network settings; for B210, add `--usb` |
| Frequent overflows (`O`) or underflows (`U`) | Apply the socket-buffer setting above and follow [System Performance Tuning](../performance-tuning/) |
| Simulation does not start | Start `ChannelSimulator` first, and use the same `simulation.session` in all configurations |
