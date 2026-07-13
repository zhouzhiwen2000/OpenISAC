---
title: System Performance Tuning
description: Host tuning required for real-time OpenISAC experiments.
---

Run the provided script to optimize your system settings for real-time processing:

```bash
cd ~/OpenISAC
chmod +x scripts/set_performance.bash
./scripts/set_performance.bash
```

> **Note:** `secure_boot` needs to be turned off in your BIOS settings to enable `RT_RUNTIME_SHARE` functionality.

When UHD spawns a new thread, it may try to boost the thread's scheduling priority. If setting the new priority fails, the UHD software prints a warning to the console, as shown below.

```text
[WARNING] [UHD] Failed to set desired affinity for thread
```

To address this issue, non-privileged (non-root) users need to be given special permission to change the scheduling priority. This can be enabled by creating a group `usrp`, adding your user to it, and then appending the line `@usrp - rtprio 99` to the file `/etc/security/limits.conf`.

```bash
sudo groupadd usrp
sudo usermod -aG usrp $USER
```

Then add the line below to end of the file `/etc/security/limits.conf`:

```text
@usrp - rtprio  99
```

You must log out and log back into the account for the settings to take effect. In most Linux distributions, a list of groups and group members can be found in the `/etc/group` file.
