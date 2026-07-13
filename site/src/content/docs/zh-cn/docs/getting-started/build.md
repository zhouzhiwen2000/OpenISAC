---
title: 编译
description: 后端编译命令和生成的二进制文件。
---

OpenISAC 使用 CMake，并要求 C++17。

## 构建 OpenISAC

使用 CMake 构建项目：

```bash
# 安装 libyaml-cpp-dev 与 ZeroMQ（libzmq + cppzmq 头文件）
sudo apt-get install libyaml-cpp-dev libzmq3-dev cppzmq-dev

cd OpenISAC
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

> 后端↔前端通信（感知数据流 + 控制/参数通道）现已基于 **ZeroMQ**。后端按配置的监听 IP/端口为感知/调试数据流绑定 PUB socket、为控制通道绑定 ROUTER socket；示例 YAML 使用 `0.0.0.0` 作为感知/调试/控制监听 IP。Python 前端用 SUB/DEALER socket 连接。让某个前端连接远端后端时，使用其 `--host <ip>` 参数或 Backend IP 输入框（默认 `127.0.0.1`）。

## 生成文件

主要生成文件为：

- `build/BS`
- `build/UE`

## 编译类型

本文默认使用 `-DCMAKE_BUILD_TYPE=Release` 构建，以生成适合实际运行的优化版本。

## 干净重编译

当 CMake 缓存状态可疑时使用：

```bash
rm -rf build
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```
