---
title: Troubleshooting
description: Common failure modes and first checks.
---

## Build fails

Check missing development packages first: UHD, Boost, FFTW3, yaml-cpp, ZeroMQ, OpenMP, and Aff3ct. If CMake cache state looks wrong, use a clean build.

## Runtime cannot find YAML

Launch from the directory containing `BS.yaml` or `UE.yaml`. The backend reads the current working directory, not `config/`.

## UHD stream errors

Reduce sample rate, check network or USB throughput, confirm device addresses and MTU, and avoid sharing high-throughput controllers.

## UE does not decode

Inspect timing lock, CFO/SFO behavior, channel estimates, and LDPC diagnostics in that order.

## Sensing output is unstable

Verify synchronization, calibration values, sensing stride, frontend rate, and whether the host is overloaded. For bistatic sensing, also inspect timing-offset assumptions.

## Frontend does not update

Check ZeroMQ endpoints, firewall rules, process launch order, and whether the backend is actually publishing the expected stream.
