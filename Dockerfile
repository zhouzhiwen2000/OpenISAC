# CPU runtime with UHD and Aff3ct built for the selected CPU instruction set.
ARG BUILD_IMAGE=ubuntu:24.04
ARG RUNTIME_IMAGE=ubuntu:24.04
FROM ubuntu:24.04 AS uhd-builder
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates cmake git pkg-config \
        libboost-filesystem-dev libboost-program-options-dev libboost-serialization-dev \
        libboost-test-dev libusb-1.0-0-dev libudev-dev \
        libgrpc++-dev libprotobuf-dev protobuf-compiler-grpc \
        python3-mako python3-numpy python3-packaging python3-requests python3-setuptools python3-yaml \
    && rm -rf /var/lib/apt/lists/*
ARG UHD_REF=v4.11.0.0
ARG BUILD_JOBS=4
RUN git clone --branch "${UHD_REF}" --depth 1 --recurse-submodules \
        https://github.com/EttusResearch/uhd.git /tmp/uhd \
    && git -C /tmp/uhd checkout -b UHD-container-release \
    && cmake -S /tmp/uhd/host -B /tmp/uhd/build \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/uhd \
        -DENABLE_TESTS=OFF -DENABLE_MANUAL=OFF -DENABLE_PYTHON_API=OFF \
        -DENABLE_EXAMPLES=ON -DENABLE_UTILS=ON \
    && cmake --build /tmp/uhd/build --parallel "${BUILD_JOBS}" \
    && cmake --install /tmp/uhd/build \
    && git -C /tmp/uhd rev-parse HEAD > /opt/uhd/share/uhd/openisac-source-revision.txt \
    && echo /opt/uhd/lib > /etc/ld.so.conf.d/openisac-uhd.conf && ldconfig \
    && rm -rf /tmp/uhd
RUN /opt/uhd/bin/uhd_images_downloader --types '^(x3|b2)' --yes --test \
        --install-location /opt/uhd/share/uhd/images \
    && test -s /opt/uhd/share/uhd/images/usrp_x310_fpga_XG.bit \
    && test -s /opt/uhd/share/uhd/images/usrp_b210_fpga.bin \
    && python3 -c 'import hashlib,json,pathlib; p=pathlib.Path("/opt/uhd/share/uhd/images"); (p.parent/"openisac-image-sha256.json").write_text(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in p.iterdir() if f.is_file() and f.suffix in (".bit",".bin",".hex")},sort_keys=True,indent=2)+"\n")'


FROM ${BUILD_IMAGE} AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG AFF3CT_REF=v4.1.2
ARG BUILD_JOBS=4
ARG CPU_ARCH=x86-64-v3

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates cmake git pkg-config \
        libboost-dev libboost-filesystem-dev libboost-program-options-dev \
        libboost-serialization-dev libboost-system-dev libboost-thread-dev \
        libusb-1.0-0 libgrpc++1.51t64 libprotobuf32t64 libfftw3-dev libyaml-cpp-dev \
        libzmq3-dev cppzmq-dev nlohmann-json3-dev python3-yaml \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --branch "${AFF3CT_REF}" --depth 1 https://github.com/aff3ct/aff3ct.git /tmp/aff3ct \
    && git -C /tmp/aff3ct submodule update --init --recursive \
    && cmake -S /tmp/aff3ct -B /tmp/aff3ct/build \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=${CPU_ARCH}" \
        -DAFF3CT_COMPILE_EXE=OFF -DAFF3CT_COMPILE_SHARED_LIB=ON \
        -DSPU_STACKTRACE=OFF -DSPU_STACKTRACE_SEGFAULT=OFF -DSPU_TESTS=OFF -DSPU_BENCHMARK=OFF \
    && cmake --build /tmp/aff3ct/build --parallel "${BUILD_JOBS}" \
    && cmake --install /tmp/aff3ct/build \
    && rm -rf /tmp/aff3ct

COPY --from=uhd-builder /opt/uhd /opt/uhd
# UHD's exported CMake package requires Boost.Test even when UHD tests are disabled.
RUN apt-get update && apt-get install -y --no-install-recommends libboost-test-dev \
    && rm -rf /var/lib/apt/lists/* \
    && echo /opt/uhd/lib > /etc/ld.so.conf.d/openisac-uhd.conf && ldconfig

WORKDIR /src
COPY CMakeLists.txt ./
COPY cmake ./cmake
COPY include ./include
COPY src ./src
COPY tests ./tests
COPY tools ./tools
ARG BUILD_TESTING=OFF
RUN cmake -S /src -B /src/build -DCMAKE_BUILD_TYPE=Release \
        -DUHD_DIR=/opt/uhd/lib/cmake/uhd -DCMAKE_PREFIX_PATH=/opt/uhd \
        -DOPENISAC_CPU_ARCH="${CPU_ARCH}" -DBUILD_TESTING="${BUILD_TESTING}" \
    && cmake --build /src/build --parallel "${BUILD_JOBS}" \
    && cmake --install /src/build --prefix /opt/openisac

FROM ${RUNTIME_IMAGE} AS runtime
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libfftw3-single3 libgomp1 \
        libboost-filesystem1.83.0 libboost-program-options1.83.0 libboost-serialization1.83.0 \
        libusb-1.0-0 libudev1 libgrpc++1.51t64 libprotobuf32t64 \
        libyaml-cpp0.8 libzmq5 python3-yaml python3-requests python3-packaging procps util-linux rt-tests \
    && rm -rf /var/lib/apt/lists/*

# Keep build toolchains and headers out of the runtime payload.
COPY --from=uhd-builder /opt/uhd/bin /opt/uhd/bin
COPY --from=uhd-builder /opt/uhd/lib /opt/uhd/lib
COPY --from=uhd-builder /opt/uhd/share /opt/uhd/share
COPY --from=builder /usr/local/lib/libaff3ct*.so* /usr/local/lib/
COPY --from=builder /opt/openisac /opt/openisac
COPY config /opt/openisac/config
COPY LDPC_504_1008.alist LDPC_504_1008G.alist PEGReg504x1008_Gen.alist /opt/openisac/
COPY docker-entrypoint.sh /usr/local/bin/openisac-run
COPY scripts/docker_preflight.py /opt/openisac/docker_preflight.py
COPY scripts/docker_uhd_prepare.py /opt/openisac/docker_uhd_prepare.py
RUN echo /opt/uhd/lib > /etc/ld.so.conf.d/openisac-uhd.conf \
    && ldconfig && chmod +x /usr/local/bin/openisac-run \
    && mkdir -p /work/build && chmod 1777 /work /work/build \
    && ln -s /opt/openisac/LDPC_504_1008.alist /work/LDPC_504_1008.alist \
    && ln -s /opt/openisac/LDPC_504_1008G.alist /work/LDPC_504_1008G.alist \
    && ln -s /opt/openisac/PEGReg504x1008_Gen.alist /work/PEGReg504x1008_Gen.alist
ENV OPENISAC_HOME=/opt/openisac OPENISAC_RUN_DIR=/work/build \
    UHD_IMAGES_DIR=/opt/uhd/share/uhd/images \
    PATH=/opt/openisac/bin:/opt/uhd/bin:/opt/uhd/lib/uhd/examples:${PATH}
WORKDIR /work/build
STOPSIGNAL SIGINT
ENTRYPOINT ["openisac-run"]
CMD ["--help"]
