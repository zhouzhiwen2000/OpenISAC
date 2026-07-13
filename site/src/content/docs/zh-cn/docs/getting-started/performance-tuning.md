---
title: 系统性能调优
description: 实时 OpenISAC 实验所需的主机调优步骤。
---

运行提供的脚本以优化您的系统设置，以满足实时处理需求：

```bash
cd ~/OpenISAC
chmod +x scripts/set_performance.bash
./scripts/set_performance.bash
```

> **注意:** 如果您需要启用 `RT_RUNTIME_SHARE` 功能，则需要在 BIOS 设置中关闭 `secure_boot`。

当 UHD 生成新线程时，它可能会尝试提高线程的调度优先级。如果设置新优先级失败，UHD 软件将向控制台打印警告，如下所示：

```text
[WARNING] [UHD] Failed to set desired affinity for thread
```

为了解决这个问题，需要给予非特权 (非 root) 用户更改调度优先级的特殊权限。这可以通过创建一个组 `usrp`，将您的用户添加到该组，然后将行 `@usrp - rtprio 99` 追加到文件 `/etc/security/limits.conf` 来启用。

```bash
sudo groupadd usrp
sudo usermod -aG usrp $USER
```

然后将以下行添加到文件 `/etc/security/limits.conf` 的末尾：

```text
@usrp - rtprio  99
```

您必须注销并重新登录帐户才能使设置生效。在大多数 Linux 发行版中，组和组成员列表可以在 `/etc/group` 文件中找到。
