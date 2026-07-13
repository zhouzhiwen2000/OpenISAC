---
title: Build
description: Build commands and generated backend binaries.
---

OpenISAC uses CMake and requires C++17.

## Build OpenISAC

Build the project using CMake:

```bash
sudo apt-get install libyaml-cpp-dev libzmq3-dev cppzmq-dev
cd OpenISAC
mkdir build
cd build
cmake ..
make -j$(nproc)
```

> Backend↔Frontend communication (sensing streams + the control/params channel) runs over **ZeroMQ**. The backend binds PUB sockets for the sensing/debug streams and a ROUTER socket for control on the configured listen IP/ports; the sample YAMLs use `0.0.0.0` for sensing/debug/control listen IPs. Python viewers connect with SUB/DEALER sockets. Point a viewer at a remote backend with its `--host <ip>` flag or Backend IP field (default `127.0.0.1`).

## Generated binaries

The primary generated binaries are:

- `build/BS`
- `build/UE`

## Build type

Release builds are the normal runtime target. To make the build type explicit:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Clean rebuild

Use a clean rebuild when CMake cache state is suspect:

```bash
rm -rf build
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```
