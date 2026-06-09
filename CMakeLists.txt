cmake_minimum_required(VERSION 3.18)
project(OFDM_Modulator CXX)

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
find_package(UHD REQUIRED)

# Detect whether this UHD version exposes post_output_action / post_input_action
# on uhd::tx_streamer / uhd::rx_streamer (added in newer UHD releases).
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_INCLUDES ${UHD_INCLUDE_DIRS})
set(CMAKE_REQUIRED_LIBRARIES ${UHD_LIBRARIES})
check_cxx_source_compiles("
#include <uhd/stream.hpp>
#include <uhd/rfnoc/actions.hpp>
struct T : uhd::tx_streamer {
    size_t get_num_channels() const override { return 0; }
    size_t get_max_num_samps() const override { return 0; }
    size_t send(const buffs_type&, size_t, const uhd::tx_metadata_t&, double) override { return 0; }
    bool recv_async_msg(uhd::async_metadata_t&, double) override { return false; }
    void post_output_action(const std::shared_ptr<uhd::rfnoc::action_info>&, size_t) override {}
};
int main() { return 0; }
" UHD_HAS_STREAMER_ACTION_HOOKS)
unset(CMAKE_REQUIRED_INCLUDES)
unset(CMAKE_REQUIRED_LIBRARIES)
if(UHD_HAS_STREAMER_ACTION_HOOKS)
    message(STATUS "UHD: streamer action hooks present — enabling SimStreamer overrides")
    add_compile_definitions(UHD_HAS_STREAMER_ACTION_HOOKS)
else()
    message(STATUS "UHD: streamer action hooks absent (UHD < 4.7) — SimStreamer overrides disabled")
endif()

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
if(PKG_CONFIG_FOUND)
    pkg_check_modules(LIBDPDK QUIET libdpdk)
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
        add_compile_options(${ARCH_OPTS})
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
set(OPENISAC_PLATFORM_LIBS "")
if(UNIX AND NOT APPLE)
    list(APPEND OPENISAC_PLATFORM_LIBS rt)
endif()

#===========================================================
# Modulator Executable
#===========================================================
add_executable(OFDMModulator
    ${OPENISAC_SRC_DIR}/OFDMModulator.cpp
    ${OPENISAC_SRC_DIR}/SensingChannel.cpp
    ${OPENISAC_SRC_DIR}/SimStreamer.cpp
    ${OPENISAC_SRC_DIR}/AsyncLogger.cpp
    ${OPENISAC_SRC_DIR}/LDPCCodec.cpp
    ${OPENISAC_SRC_DIR}/LDPC5041008SIMD.cpp
)

target_include_directories(OFDMModulator PRIVATE
    ${UHD_INCLUDE_DIRS}
    ${LIBDPDK_INCLUDE_DIRS}
    ${FFTW3f_INCLUDE_DIRS}
    ${YAMLCPP_INCLUDE_DIRS}
    ${OPENISAC_ZMQ_INCLUDE_DIRS}
    ${OPENISAC_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Boost_INCLUDE_DIRS}
)

target_link_libraries(OFDMModulator PRIVATE
    ${UHD_LIBRARIES}
    Threads::Threads
    OpenMP::OpenMP_CXX
    ${LIBDPDK_LIBRARIES}
    ${FFTW3f_LIBRARIES}
    ${FFTW3f_THREADS_LIBRARY}
    aff3ct::aff3ct-shared-lib
    ${OPENISAC_YAMLCPP_TARGET}
    ${YAMLCPP_LIBRARIES}
    ${OPENISAC_ZMQ_LINK}
    ${OPENISAC_PLATFORM_LIBS}
)

target_compile_options(OFDMModulator PRIVATE
    ${COMMON_COMPILE_OPTIONS}
)

#===========================================================
# Demodulator Executable
#===========================================================
add_executable(OFDMDemodulator
    ${OPENISAC_SRC_DIR}/OFDMDemodulator.cpp
    ${OPENISAC_SRC_DIR}/SensingChannel.cpp
    ${OPENISAC_SRC_DIR}/SimStreamer.cpp
    ${OPENISAC_SRC_DIR}/AsyncLogger.cpp
    ${OPENISAC_SRC_DIR}/LDPCCodec.cpp
    ${OPENISAC_SRC_DIR}/LDPC5041008SIMD.cpp
)

target_include_directories(OFDMDemodulator PRIVATE
    ${UHD_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${FFTW3f_INCLUDE_DIRS}
    ${YAMLCPP_INCLUDE_DIRS}
    ${OPENISAC_ZMQ_INCLUDE_DIRS}
    ${OPENISAC_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(OFDMDemodulator PRIVATE
    ${UHD_LIBRARIES}
    ${FFTW3f_LIBRARIES}
    ${FFTW3f_THREADS_LIBRARY}
    Threads::Threads
    OpenMP::OpenMP_CXX
    aff3ct::aff3ct-shared-lib
    ${OPENISAC_YAMLCPP_TARGET}
    ${YAMLCPP_LIBRARIES}
    ${OPENISAC_ZMQ_LINK}
    ${OPENISAC_PLATFORM_LIBS}
)

target_compile_options(OFDMDemodulator PRIVATE
    ${COMMON_COMPILE_OPTIONS}
    -ffast-math  # Demodulator specific optimization
)

#===========================================================
# ChannelSimulator Executable (run the system without USRP)
#===========================================================
add_executable(ChannelSimulator
    ${OPENISAC_SRC_DIR}/ChannelSimulator.cpp
    ${OPENISAC_SRC_DIR}/SimStreamer.cpp
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
    ${OPENISAC_PLATFORM_LIBS}
)

target_compile_options(ChannelSimulator PRIVATE
    ${COMMON_COMPILE_OPTIONS}
    -ffast-math  # Enable SIMD vectorization of complex<float> arithmetic
)

#===========================================================
# Installation Rules
#===========================================================
set(INSTALL_TARGETS OFDMModulator OFDMDemodulator)

install(TARGETS ${INSTALL_TARGETS}
    RUNTIME DESTINATION bin
)
