cmake_minimum_required(VERSION 3.12)
project(OFDM_Modulator CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find required libraries
find_package(UHD REQUIRED)
find_package(Boost 1.66 REQUIRED COMPONENTS system program_options)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)  # Require OpenMP support
find_package(PkgConfig REQUIRED)
set (AFF3CT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/")
find_package(AFF3CT CONFIG 3.0.2 REQUIRED)
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
pkg_check_modules(LIBDPDK QUIET libdpdk)
pkg_check_modules(FFTW3f REQUIRED fftw3f)
pkg_check_modules(YAMLCPP REQUIRED yaml-cpp)   # YAML configuration support
# Check SIMD instruction set support (common optimization)
include(CheckCXXCompilerFlag)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|Intel")
    check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
    check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)

    set(ARCH_OPTS "")
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
        list(APPEND ARCH_OPTS "-march=native")
    else()
        if(COMPILER_SUPPORTS_AVX2)
            list(APPEND ARCH_OPTS "-mavx2")
        endif()
        if(COMPILER_SUPPORTS_FMA)
            list(APPEND ARCH_OPTS "-mfma")
        endif()
    endif()

    if(ARCH_OPTS)
        add_compile_options(${ARCH_OPTS})
    endif()
endif()

# Common compile options
set(COMMON_COMPILE_OPTIONS
    -O3
    -Wall
    -Wextra
    -Wpedantic
)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    list(APPEND COMMON_COMPILE_OPTIONS -Wno-unsupported-floating-point-opt)
endif()

#===========================================================
# Modulator Executable
#===========================================================
add_executable(OFDMModulator
    OFDMModulator.cpp
)

target_include_directories(OFDMModulator PRIVATE
    ${UHD_INCLUDE_DIRS}
    ${LIBDPDK_INCLUDE_DIRS}
    ${FFTW3f_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Boost_INCLUDE_DIRS}
)

target_link_libraries(OFDMModulator PRIVATE
    ${UHD_LIBRARIES}
    Threads::Threads
    OpenMP::OpenMP_CXX
    ${LIBDPDK_LIBRARIES}
    ${FFTW3f_LIBRARIES}
    fftw3f_threads
    ${Boost_LIBRARIES}
    aff3ct::aff3ct-shared-lib
    ${YAMLCPP_LIBRARIES}
)

target_compile_options(OFDMModulator PRIVATE
    ${COMMON_COMPILE_OPTIONS}
)

#===========================================================
# Demodulator Executable
#===========================================================
add_executable(OFDMDemodulator
    OFDMDemodulator.cpp
)

target_include_directories(OFDMDemodulator PRIVATE
    ${UHD_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${FFTW3f_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(OFDMDemodulator PRIVATE 
    ${UHD_LIBRARIES}
    Boost::system
    ${FFTW3f_LIBRARIES}
    Threads::Threads
    OpenMP::OpenMP_CXX
    ${Boost_LIBRARIES}
    aff3ct::aff3ct-shared-lib
    ${YAMLCPP_LIBRARIES}
)

target_compile_options(OFDMDemodulator PRIVATE
    ${COMMON_COMPILE_OPTIONS}
    -ffast-math  # Demodulator specific optimization
)

#===========================================================
# Installation Rules
#===========================================================
install(TARGETS OFDMModulator OFDMDemodulator
    RUNTIME DESTINATION bin
)