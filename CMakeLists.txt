cmake_minimum_required(VERSION 3.18)
project(OpenISAC CXX)

include(CTest)

if(BUILD_TESTING)
    add_executable(TxBurstRecoveryTest tests/tx_burst_recovery_test.cpp)
    target_include_directories(TxBurstRecoveryTest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    add_test(NAME TxBurstRecovery COMMAND TxBurstRecoveryTest)
    set_tests_properties(TxBurstRecovery PROPERTIES LABELS "cpu;recovery")
endif()

if(POLICY CMP0167)
    cmake_policy(SET CMP0167 OLD)
endif()

if(APPLE)
    # Help CMake discover common Homebrew/MacPorts prefixes on macOS.
    list(PREPEND CMAKE_PREFIX_PATH
        /opt/homebrew
        /opt/homebrew/opt/libomp
        /opt/local
        /usr/local
    )
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find required libraries
find_package(Boost 1.66 CONFIG QUIET)
if(Boost_FOUND AND Boost_VERSION VERSION_GREATER_EQUAL 1.89.0)
    find_package(boost_system ${Boost_VERSION} EXACT CONFIG QUIET)
    if(NOT boost_system_FOUND)
        # Some Boost 1.89+ packages omit the standalone Boost.System CMake
        # component, but UHD 4.10 and older package configs still request it.
        set(OPENISAC_BOOST_VERSION "${Boost_VERSION}")
        set(boost_system_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cmake/boost_system")
        message(STATUS
            "Using header-only Boost.System compatibility target for Boost ${Boost_VERSION}"
        )
    endif()
endif()
find_package(UHD REQUIRED)

set(Boost_NO_BOOST_CMAKE ON)
find_package(Boost 1.66 REQUIRED)

find_package(Threads REQUIRED)
find_package(OpenMP QUIET)
if(NOT OpenMP_CXX_FOUND AND APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    find_path(OPENMP_INCLUDE_DIR
        NAMES omp.h
        HINTS
            /opt/homebrew/opt/libomp/include
            /opt/local/include
            /opt/local/include/libomp
            /usr/local/include
    )
    find_library(OPENMP_LIBRARY
        NAMES omp libomp
        HINTS
            /opt/homebrew/opt/libomp/lib
            /opt/local/lib
            /usr/local/lib
    )
    if(OPENMP_INCLUDE_DIR AND OPENMP_LIBRARY)
        add_library(OpenMP::OpenMP_CXX INTERFACE IMPORTED)
        set_target_properties(OpenMP::OpenMP_CXX PROPERTIES
            INTERFACE_COMPILE_OPTIONS "-Xpreprocessor;-fopenmp"
            INTERFACE_INCLUDE_DIRECTORIES "${OPENMP_INCLUDE_DIR}"
            INTERFACE_LINK_LIBRARIES "${OPENMP_LIBRARY}"
        )
        set(OpenMP_CXX_FOUND TRUE)
    endif()
endif()
if(NOT OpenMP_CXX_FOUND)
    message(FATAL_ERROR "OpenMP support is required. Install libomp and re-run CMake.")
endif()

find_package(PkgConfig QUIET)

set(AFF3CT_SEARCH_ROOTS
    ${AFF3CT_ROOT}
    $ENV{AFF3CT_ROOT}
    $ENV{AFF3CT_DIR}
    /opt/homebrew
    /opt/local
    /usr/local
)
find_package(AFF3CT CONFIG 3.0.2 QUIET
    PATHS ${AFF3CT_SEARCH_ROOTS}
    PATH_SUFFIXES lib/cmake/AFF3CT share/cmake/AFF3CT
)
if(NOT AFF3CT_FOUND)
    message(FATAL_ERROR
        "AFF3CT >= 3.0.2 not found. Install it and re-run CMake with "
        "-DAFF3CT_ROOT=<install-prefix> or -DAFF3CT_DIR=<cmake-config-dir>."
    )
endif()
# Get original header file paths
get_target_property(AFF3CT_INCLUDES aff3ct::aff3ct-shared-lib INTERFACE_INCLUDE_DIRECTORIES)

# Filter out non-existent paths
foreach(INCLUDE_PATH IN LISTS AFF3CT_INCLUDES)
    if(NOT EXISTS "${INCLUDE_PATH}")
        list(REMOVE_ITEM AFF3CT_INCLUDES "${INCLUDE_PATH}")
        message(WARNING "Removed non-existent include path: ${INCLUDE_PATH}")
    endif()
endforeach()

# Reset header file paths
set_target_properties(aff3ct::aff3ct-shared-lib PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${AFF3CT_INCLUDES}"
)

# Find third-party libraries
set(OPENISAC_DPDK_TARGET "")
if(PKG_CONFIG_FOUND)
    # Preserve pkg-config library paths and link options for non-system installs.
    pkg_check_modules(LIBDPDK QUIET IMPORTED_TARGET libdpdk)
    if(TARGET PkgConfig::LIBDPDK)
        set(OPENISAC_DPDK_TARGET PkgConfig::LIBDPDK)
    endif()
endif()

find_package(FFTW3f CONFIG QUIET)
if(NOT FFTW3f_FOUND AND PKG_CONFIG_FOUND)
    pkg_check_modules(FFTW3f QUIET fftw3f)
endif()
if(FFTW3f_FOUND)
    find_library(FFTW3f_LIBRARY
        NAMES fftw3f
        HINTS ${FFTW3f_LIBRARY_DIRS} /opt/homebrew/lib /opt/local/lib /usr/local/lib
    )
    if(FFTW3f_LIBRARY)
        set(FFTW3f_LIBRARIES "${FFTW3f_LIBRARY}")
    endif()
endif()
if(NOT FFTW3f_FOUND)
    find_path(FFTW3f_INCLUDE_DIRS
        NAMES fftw3.h
        HINTS /opt/homebrew/include /opt/local/include /usr/local/include
    )
    find_library(FFTW3f_LIBRARY
        NAMES fftw3f
        HINTS /opt/homebrew/lib /opt/local/lib /usr/local/lib
    )
    if(FFTW3f_INCLUDE_DIRS AND FFTW3f_LIBRARY)
        set(FFTW3f_LIBRARIES "${FFTW3f_LIBRARY}")
        set(FFTW3f_FOUND TRUE)
    endif()
endif()
find_library(FFTW3f_THREADS_LIBRARY
    NAMES fftw3f_threads
    HINTS /opt/homebrew/lib /opt/local/lib /usr/local/lib
)
if(NOT FFTW3f_FOUND OR NOT FFTW3f_THREADS_LIBRARY)
    message(FATAL_ERROR "FFTW3f and fftw3f_threads are required.")
endif()

find_package(yaml-cpp CONFIG QUIET)
if(yaml-cpp_FOUND)
    if(TARGET yaml-cpp::yaml-cpp)
        set(OPENISAC_YAMLCPP_TARGET yaml-cpp::yaml-cpp)
    elseif(TARGET yaml-cpp)
        set(OPENISAC_YAMLCPP_TARGET yaml-cpp)
    endif()
    if(DEFINED YAML_CPP_INCLUDE_DIR)
        set(YAMLCPP_INCLUDE_DIRS "${YAML_CPP_INCLUDE_DIR}")
    endif()
endif()
if(NOT yaml-cpp_FOUND AND PKG_CONFIG_FOUND)
    pkg_check_modules(YAMLCPP QUIET yaml-cpp)
endif()
if(NOT yaml-cpp_FOUND AND NOT YAMLCPP_FOUND)
    find_path(YAMLCPP_INCLUDE_DIRS
        NAMES yaml-cpp/yaml.h
        HINTS
            /opt/homebrew/include
            /opt/local/include
            /usr/local/include
            /opt/homebrew/Cellar/yaml-cpp/0.8.0/include
    )
    find_library(YAMLCPP_LIBRARY
        NAMES yaml-cpp
        HINTS
            /opt/homebrew/lib
            /opt/local/lib
            /usr/local/lib
            /opt/homebrew/Cellar/yaml-cpp/0.8.0/lib
    )
    if(YAMLCPP_INCLUDE_DIRS AND YAMLCPP_LIBRARY)
        set(YAMLCPP_LIBRARIES "${YAMLCPP_LIBRARY}")
        set(YAMLCPP_FOUND TRUE)
    endif()
endif()
if(NOT OPENISAC_YAMLCPP_TARGET AND NOT YAMLCPP_FOUND)
    message(FATAL_ERROR "yaml-cpp is required.")
endif()

# ZeroMQ (Backend <-> Frontend transport: sensing streams + control channel).
# Prefer the cppzmq CMake package (header-only, pulls in libzmq); fall back to
# pkg-config / manual discovery of libzmq + the cppzmq header.
find_package(cppzmq CONFIG QUIET)
if(cppzmq_FOUND AND TARGET cppzmq)
    set(OPENISAC_ZMQ_LINK cppzmq)
    set(OPENISAC_ZMQ_INCLUDE_DIRS "")
else()
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(ZMQ QUIET libzmq)
    endif()
    if(NOT ZMQ_FOUND)
        find_path(ZMQ_INCLUDE_DIRS
            NAMES zmq.hpp zmq.h
            HINTS /usr/include /usr/local/include /opt/homebrew/include /opt/local/include
        )
        find_library(ZMQ_LIBRARY
            NAMES zmq
            HINTS /usr/lib /usr/local/lib /opt/homebrew/lib /opt/local/lib
        )
        if(ZMQ_INCLUDE_DIRS AND ZMQ_LIBRARY)
            set(ZMQ_LIBRARIES "${ZMQ_LIBRARY}")
            set(ZMQ_FOUND TRUE)
        endif()
    endif()
    if(NOT ZMQ_FOUND)
        message(FATAL_ERROR "ZeroMQ (libzmq + cppzmq headers) is required.")
    endif()
    set(OPENISAC_ZMQ_LINK ${ZMQ_LIBRARIES})
    set(OPENISAC_ZMQ_INCLUDE_DIRS ${ZMQ_INCLUDE_DIRS})
endif()
# Check SIMD instruction set support (common optimization)
include(CheckCXXCompilerFlag)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|Intel")
    check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
    check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512F)
    check_cxx_compiler_flag("-mavx512dq" COMPILER_SUPPORTS_AVX512DQ)
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
    check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)

    set(ARCH_OPTS "")
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
        # -march=native will automatically enable AVX512 if CPU supports it
        list(APPEND ARCH_OPTS "-march=native")
    else()
        # Manual fallback: try AVX512 first, then AVX2
        if(COMPILER_SUPPORTS_AVX512F)
            list(APPEND ARCH_OPTS "-mavx512f")
            if(COMPILER_SUPPORTS_AVX512DQ)
                list(APPEND ARCH_OPTS "-mavx512dq")
            endif()
        elseif(COMPILER_SUPPORTS_AVX2)
            list(APPEND ARCH_OPTS "-mavx2")
        endif()
        if(COMPILER_SUPPORTS_FMA)
            list(APPEND ARCH_OPTS "-mfma")
        endif()
    endif()

    if(ARCH_OPTS)
        add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:${ARCH_OPTS}>")
        message(STATUS "Enabled SIMD optimizations: ${ARCH_OPTS}")
    endif()
endif()

# Common compile options
set(COMMON_COMPILE_OPTIONS
    -O3
    -Wall
    -Wextra
    -Wpedantic
    -funroll-loops          # Loop unrolling helps vectorization
    -ftree-vectorize        # Enable auto-vectorization
   #-fopt-info-vec-optimized # Report successful vectorizations (optional)
)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    list(APPEND COMMON_COMPILE_OPTIONS -Wno-unsupported-floating-point-opt)
endif()

set(OPENISAC_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(OPENISAC_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

# This I/O boundary validates finite configuration values. Preserve those checks
# even in UE targets that enable fast-math for their DSP implementation.
set_source_files_properties(${OPENISAC_SRC_DIR}/Xr1710gBackend.cpp
    PROPERTIES COMPILE_OPTIONS "-fno-fast-math")

#===========================================================
# BS Executable
#===========================================================
add_executable(BS
    ${OPENISAC_SRC_DIR}/BS.cpp
    ${OPENISAC_SRC_DIR}/SensingChannel.cpp
    ${OPENISAC_SRC_DIR}/UhdBackend.cpp
    ${OPENISAC_SRC_DIR}/SimBackend.cpp
    ${OPENISAC_SRC_DIR}/Xr1710gBackend.cpp
    ${OPENISAC_SRC_DIR}/AsyncLogger.cpp
    ${OPENISAC_SRC_DIR}/LDPCCodec.cpp
    ${OPENISAC_SRC_DIR}/LDPC5041008SIMD.cpp
)

target_include_directories(BS PRIVATE
    ${UHD_INCLUDE_DIRS}
    ${FFTW3f_INCLUDE_DIRS}
    ${YAMLCPP_INCLUDE_DIRS}
    ${OPENISAC_ZMQ_INCLUDE_DIRS}
    ${OPENISAC_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Boost_INCLUDE_DIRS}
)

target_link_libraries(BS PRIVATE
    ${UHD_LIBRARIES}
    Threads::Threads
    OpenMP::OpenMP_CXX
    ${OPENISAC_DPDK_TARGET}
    ${FFTW3f_LIBRARIES}
    ${FFTW3f_THREADS_LIBRARY}
    aff3ct::aff3ct-shared-lib
    ${OPENISAC_YAMLCPP_TARGET}
    ${YAMLCPP_LIBRARIES}
    ${OPENISAC_ZMQ_LINK}
    rt
)

target_compile_options(BS PRIVATE
    ${COMMON_COMPILE_OPTIONS}
)

#===========================================================
# UE Executable
#===========================================================
add_executable(UE
    ${OPENISAC_SRC_DIR}/UE.cpp
    ${OPENISAC_SRC_DIR}/SensingChannel.cpp
    ${OPENISAC_SRC_DIR}/UhdBackend.cpp
    ${OPENISAC_SRC_DIR}/SimBackend.cpp
    ${OPENISAC_SRC_DIR}/Xr1710gBackend.cpp
    ${OPENISAC_SRC_DIR}/AsyncLogger.cpp
    ${OPENISAC_SRC_DIR}/LDPCCodec.cpp
    ${OPENISAC_SRC_DIR}/LDPC5041008SIMD.cpp
)

target_include_directories(UE PRIVATE
    ${UHD_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${FFTW3f_INCLUDE_DIRS}
    ${YAMLCPP_INCLUDE_DIRS}
    ${OPENISAC_ZMQ_INCLUDE_DIRS}
    ${OPENISAC_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(UE PRIVATE
    ${UHD_LIBRARIES}
    ${FFTW3f_LIBRARIES}
    ${FFTW3f_THREADS_LIBRARY}
    Threads::Threads
    OpenMP::OpenMP_CXX
    aff3ct::aff3ct-shared-lib
    ${OPENISAC_YAMLCPP_TARGET}
    ${YAMLCPP_LIBRARIES}
    ${OPENISAC_ZMQ_LINK}
    rt
)

target_compile_options(UE PRIVATE
    ${COMMON_COMPILE_OPTIONS}
    -ffast-math  # UE specific optimization
)

#===========================================================
# ChannelSimulator Executable (run the system without USRP)
#===========================================================
add_executable(ChannelSimulator
    ${OPENISAC_SRC_DIR}/ChannelSimulator.cpp
    ${OPENISAC_SRC_DIR}/AsyncLogger.cpp
)

target_include_directories(ChannelSimulator PRIVATE
    ${UHD_INCLUDE_DIRS}
    ${YAMLCPP_INCLUDE_DIRS}
    ${OPENISAC_ZMQ_INCLUDE_DIRS}
    ${OPENISAC_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Boost_INCLUDE_DIRS}
)

target_link_libraries(ChannelSimulator PRIVATE
    ${UHD_LIBRARIES}
    Threads::Threads
    OpenMP::OpenMP_CXX
    ${OPENISAC_YAMLCPP_TARGET}
    ${YAMLCPP_LIBRARIES}
    ${OPENISAC_ZMQ_LINK}
    rt
)

target_compile_options(ChannelSimulator PRIVATE
    ${COMMON_COMPILE_OPTIONS}
    -ffast-math  # Enable SIMD vectorization of complex<float> arithmetic
)

#===========================================================
# Standalone UDP link tester
#===========================================================
add_executable(UdpLinkTest tools/udp_link_test.cpp)
target_include_directories(UdpLinkTest PRIVATE ${OPENISAC_INCLUDE_DIR} ${OPENISAC_ZMQ_INCLUDE_DIRS})
target_link_libraries(UdpLinkTest PRIVATE Threads::Threads ${OPENISAC_ZMQ_LINK}
    ${OPENISAC_YAMLCPP_TARGET} ${YAMLCPP_LIBRARIES})
target_compile_options(UdpLinkTest PRIVATE ${COMMON_COMPILE_OPTIONS})
if(BUILD_TESTING)
    add_executable(UdpLinkProtocolTest tests/udp_link_test_test.cpp)
    target_include_directories(UdpLinkProtocolTest PRIVATE ${OPENISAC_INCLUDE_DIR})
    target_compile_options(UdpLinkProtocolTest PRIVATE ${COMMON_COMPILE_OPTIONS})
    add_test(NAME UdpLinkProtocol COMMAND UdpLinkProtocolTest)
    set_tests_properties(UdpLinkProtocol PROPERTIES LABELS "cpu;network" TIMEOUT 10)
endif()

#===========================================================
# Standalone CPU LDPC fixed-point precision benchmark (no CUDA)
#===========================================================
add_executable(LDPCFixedBench tools/ldpc_fixed_bench.cpp)
target_include_directories(LDPCFixedBench PRIVATE
    ${OPENISAC_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(LDPCFixedBench PRIVATE
    Threads::Threads
    aff3ct::aff3ct-shared-lib
)
target_compile_options(LDPCFixedBench PRIVATE ${COMMON_COMPILE_OPTIONS})
message(STATUS "CPU target: LDPCFixedBench")

# Read-only XR1710G consumer/probe; no UHD dependency in the adapter itself.
add_executable(Xr1710gRxProbe tools/xr1710g_rx_probe.cpp ${OPENISAC_SRC_DIR}/Xr1710gBackend.cpp)
target_include_directories(Xr1710gRxProbe PRIVATE ${OPENISAC_INCLUDE_DIR})
target_link_libraries(Xr1710gRxProbe PRIVATE Threads::Threads OpenMP::OpenMP_CXX rt)
target_compile_options(Xr1710gRxProbe PRIVATE ${COMMON_COMPILE_OPTIONS})

add_executable(Xr1710gConvertBench tools/xr1710g_convert_bench.cpp)
target_include_directories(Xr1710gConvertBench PRIVATE ${OPENISAC_INCLUDE_DIR})
target_link_libraries(Xr1710gConvertBench PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(Xr1710gConvertBench PRIVATE ${COMMON_COMPILE_OPTIONS})

if(BUILD_TESTING)
    add_executable(Xr1710gBsBridge tests/xr1710g_bs_bridge.cpp)
    target_include_directories(Xr1710gBsBridge PRIVATE ${OPENISAC_INCLUDE_DIR})
    target_link_libraries(Xr1710gBsBridge PRIVATE Threads::Threads rt)
    target_compile_options(Xr1710gBsBridge PRIVATE ${COMMON_COMPILE_OPTIONS})

    add_executable(Xr1710gBackendTest tests/xr1710g_backend_test.cpp ${OPENISAC_SRC_DIR}/Xr1710gBackend.cpp)
    target_include_directories(Xr1710gBackendTest PRIVATE ${OPENISAC_INCLUDE_DIR})
    target_link_libraries(Xr1710gBackendTest PRIVATE Threads::Threads OpenMP::OpenMP_CXX rt)
    target_compile_options(Xr1710gBackendTest PRIVATE ${COMMON_COMPILE_OPTIONS})
    add_test(NAME Xr1710gBackend COMMAND Xr1710gBackendTest)
    set_tests_properties(Xr1710gBackend PROPERTIES LABELS "cpu;radio;xr1710g" TIMEOUT 15)

    add_executable(Xr1710gConfigTest tests/xr1710g_config_test.cpp
        ${OPENISAC_SRC_DIR}/Xr1710gBackend.cpp ${OPENISAC_SRC_DIR}/AsyncLogger.cpp)
    target_include_directories(Xr1710gConfigTest PRIVATE ${OPENISAC_INCLUDE_DIR}
        ${FFTW3f_INCLUDE_DIRS} ${YAMLCPP_INCLUDE_DIRS} ${OPENISAC_ZMQ_INCLUDE_DIRS} ${UHD_INCLUDE_DIRS})
    target_link_libraries(Xr1710gConfigTest PRIVATE Threads::Threads OpenMP::OpenMP_CXX
        ${OPENISAC_YAMLCPP_TARGET} ${YAMLCPP_LIBRARIES} ${OPENISAC_ZMQ_LINK} rt)
    target_compile_options(Xr1710gConfigTest PRIVATE ${COMMON_COMPILE_OPTIONS})
    add_test(NAME Xr1710gConfig COMMAND Xr1710gConfigTest "${CMAKE_CURRENT_SOURCE_DIR}")
    set_tests_properties(Xr1710gConfig PROPERTIES LABELS "cpu;radio;xr1710g" TIMEOUT 15)

    add_executable(BistaticTimelineTest
        tests/bistatic_timeline_test.cpp ${OPENISAC_SRC_DIR}/SensingChannel.cpp
        ${OPENISAC_SRC_DIR}/UhdBackend.cpp ${OPENISAC_SRC_DIR}/SimBackend.cpp
        ${OPENISAC_SRC_DIR}/Xr1710gBackend.cpp
        ${OPENISAC_SRC_DIR}/AsyncLogger.cpp)
    target_include_directories(BistaticTimelineTest PRIVATE
        ${OPENISAC_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR} ${FFTW3f_INCLUDE_DIRS}
        ${OPENISAC_ZMQ_INCLUDE_DIRS} ${UHD_INCLUDE_DIRS} ${YAMLCPP_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
    target_link_libraries(BistaticTimelineTest PRIVATE
        ${UHD_LIBRARIES} ${FFTW3f_LIBRARIES} ${FFTW3f_THREADS_LIBRARY}
        Threads::Threads OpenMP::OpenMP_CXX ${OPENISAC_YAMLCPP_TARGET}
        ${YAMLCPP_LIBRARIES} ${OPENISAC_ZMQ_LINK} rt)
    target_compile_options(BistaticTimelineTest PRIVATE ${COMMON_COMPILE_OPTIONS} -ffast-math)
    add_test(NAME BistaticTimeline COMMAND BistaticTimelineTest)
    set_tests_properties(BistaticTimeline PROPERTIES LABELS "cpu;sensing;recovery" TIMEOUT 15)

    add_executable(SyncPreambleTest tests/sync_preamble_test.cpp ${OPENISAC_SRC_DIR}/AsyncLogger.cpp)
    target_include_directories(SyncPreambleTest PRIVATE
        ${OPENISAC_INCLUDE_DIR} ${FFTW3f_INCLUDE_DIRS} ${OPENISAC_ZMQ_INCLUDE_DIRS}
        ${UHD_INCLUDE_DIRS} ${YAMLCPP_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
    target_link_libraries(SyncPreambleTest PRIVATE
        ${FFTW3f_LIBRARIES} Threads::Threads OpenMP::OpenMP_CXX ${OPENISAC_ZMQ_LINK})
    target_compile_options(SyncPreambleTest PRIVATE ${COMMON_COMPILE_OPTIONS})
    add_test(NAME SyncPreamble COMMAND SyncPreambleTest)
    set_tests_properties(SyncPreamble PROPERTIES LABELS "cpu;sync" TIMEOUT 30)

    add_executable(LDPCConfigTest
        tests/ldpc_config_test.cpp
        ${OPENISAC_SRC_DIR}/LDPCCodec.cpp
        ${OPENISAC_SRC_DIR}/LDPC5041008SIMD.cpp
        ${OPENISAC_SRC_DIR}/AsyncLogger.cpp
    )
    target_include_directories(LDPCConfigTest PRIVATE
        ${UHD_INCLUDE_DIRS} ${FFTW3f_INCLUDE_DIRS} ${YAMLCPP_INCLUDE_DIRS}
        ${OPENISAC_ZMQ_INCLUDE_DIRS} ${OPENISAC_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR} ${Boost_INCLUDE_DIRS}
    )
    target_link_libraries(LDPCConfigTest PRIVATE
        ${UHD_LIBRARIES} ${FFTW3f_LIBRARIES} ${FFTW3f_THREADS_LIBRARY}
        Threads::Threads OpenMP::OpenMP_CXX aff3ct::aff3ct-shared-lib
        ${OPENISAC_YAMLCPP_TARGET} ${YAMLCPP_LIBRARIES} ${OPENISAC_ZMQ_LINK} rt
    )
    target_compile_options(LDPCConfigTest PRIVATE ${COMMON_COMPILE_OPTIONS} -ffast-math)
    add_test(NAME LDPCConfig COMMAND LDPCConfigTest "${CMAKE_CURRENT_SOURCE_DIR}")
    set_tests_properties(LDPCConfig PROPERTIES LABELS "cpu;ldpc" TIMEOUT 120)

    add_executable(ModulationTest
        tests/modulation_test.cpp
        ${OPENISAC_SRC_DIR}/LDPCCodec.cpp
        ${OPENISAC_SRC_DIR}/LDPC5041008SIMD.cpp
        ${OPENISAC_SRC_DIR}/AsyncLogger.cpp
    )
    target_include_directories(ModulationTest PRIVATE
        ${UHD_INCLUDE_DIRS} ${FFTW3f_INCLUDE_DIRS} ${YAMLCPP_INCLUDE_DIRS}
        ${OPENISAC_ZMQ_INCLUDE_DIRS} ${OPENISAC_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR} ${Boost_INCLUDE_DIRS}
    )
    target_link_libraries(ModulationTest PRIVATE
        ${UHD_LIBRARIES} ${FFTW3f_LIBRARIES} ${FFTW3f_THREADS_LIBRARY}
        Threads::Threads OpenMP::OpenMP_CXX aff3ct::aff3ct-shared-lib
        ${OPENISAC_YAMLCPP_TARGET} ${YAMLCPP_LIBRARIES} ${OPENISAC_ZMQ_LINK} rt
    )
    target_compile_options(ModulationTest PRIVATE ${COMMON_COMPILE_OPTIONS} -ffast-math)
    add_test(NAME Modulation COMMAND ModulationTest "${CMAKE_CURRENT_SOURCE_DIR}")
    set_tests_properties(Modulation PROPERTIES LABELS "cpu;modulation;ldpc" TIMEOUT 120)

    add_executable(LDPCRateTest
        tests/ldpc_rate_test.cpp
        ${OPENISAC_SRC_DIR}/LDPCCodec.cpp
        ${OPENISAC_SRC_DIR}/LDPC5041008SIMD.cpp
        ${OPENISAC_SRC_DIR}/AsyncLogger.cpp
    )
    target_include_directories(LDPCRateTest PRIVATE
        ${UHD_INCLUDE_DIRS} ${FFTW3f_INCLUDE_DIRS} ${YAMLCPP_INCLUDE_DIRS}
        ${OPENISAC_ZMQ_INCLUDE_DIRS} ${OPENISAC_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR} ${Boost_INCLUDE_DIRS}
    )
    target_link_libraries(LDPCRateTest PRIVATE
        ${UHD_LIBRARIES} ${FFTW3f_LIBRARIES} ${FFTW3f_THREADS_LIBRARY}
        Threads::Threads OpenMP::OpenMP_CXX aff3ct::aff3ct-shared-lib
        ${OPENISAC_YAMLCPP_TARGET} ${YAMLCPP_LIBRARIES} ${OPENISAC_ZMQ_LINK} rt
    )
    target_compile_options(LDPCRateTest PRIVATE ${COMMON_COMPILE_OPTIONS} -ffast-math)
    add_test(NAME LDPCRate COMMAND LDPCRateTest "${CMAKE_CURRENT_SOURCE_DIR}")
    set_tests_properties(LDPCRate PROPERTIES LABELS "cpu;modulation;ldpc" TIMEOUT 120)

    add_executable(ModulationBench tools/modulation_bench.cpp)
    target_include_directories(ModulationBench PRIVATE ${OPENISAC_INCLUDE_DIR})
    target_link_libraries(ModulationBench PRIVATE OpenMP::OpenMP_CXX)
    target_compile_options(ModulationBench PRIVATE ${COMMON_COMPILE_OPTIONS} -ffast-math)
endif()

#===========================================================
# Installation Rules
#===========================================================
set(INSTALL_TARGETS BS UE UdpLinkTest)

install(TARGETS ${INSTALL_TARGETS}
    RUNTIME DESTINATION bin
)
