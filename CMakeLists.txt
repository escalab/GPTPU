cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PROJ_NAME "gptpu_new")
project(${PROJ_NAME} VERSION 1.0)

set(CMAKE_C_FLAGS "-Wall -pthread")
set(CMAKE_C_FLAGS_DEBUG "-g -O0")
set(CMAKE_C_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -std=c++17 -lstdc++ -Wl,--no-as-needed -ldl")
set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})
set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})

set(TENSORFLOW_COMMIT a4dfb8d1a71385bd6d122e4f27f86dcebb96712d)

# GET CPU Architechture
if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
	set(EDGE_ARCH "k8")
 	set(TF_COMMAND make -j8 BUILD_WITH_NNAPI=false -C ${CMAKE_BINARY_DIR}/tensorflow/src/tf -f tensorflow/lite/tools/make/Makefile lib)
   	set(TF_INSTALL_PREFIX "linux_x86_64")
elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
   	set(EDGE_ARCH "${CMAKE_SYSTEM_PROCESSOR}")
   	set(TF_INSTALL_PREFIX "generic-aarch64_armv8-a")
   	set(CROSS_PREFIX "aarch64-linux-gnu-")
   	set(TF_COMMAND make -j4 TARGET=generic-aarch64 TARGET_ARCH=armv8-a -C ${CMAKE_BINARY_DIR}/tensorflow/src/tf -f tensorflow/lite/tools/make/Makefile CC=${CROSS_PREFIX}g++ CXX=${CROSS_PREFIX}g++ AR=${CROSS_PREFIX}ar CFLAGS=-fpermissive lib)
elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "armv7l")
   	message("We are going to assume that this build is for the rpi")
   	set(EDGE_ARCH "${CMAKE_SYSTEM_PROCESSOR}")
   	set(TF_INSTALL_PREFIX "rpi_armv7l")
   	set(CROSS_PREFIX "arm-linux-gnueabihf-")
   	set(TF_COMMAND make -j4 TARGET=rpi TARGET_ARCH=armv7l -C ${CMAKE_BINARY_DIR}/tensorflow/src/tf -f tensorflow/lite/tools/make/Makefile CC=${CROSS_PREFIX}g++ CXX=${CROSS_PREFIX}g++ AR=${CROSS_PREFIX}ar CFLAGS=-fpermissive lib)
else()
	message(FATAL_ERROR "Not implemented to build for: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

message("System ARCH is ${CMAKE_SYSTEM_PROCESSOR} ${EDGE_ARCH}")

include(ExternalProject)

ExternalProject_Add(tf
	GIT_REPOSITORY https://github.com/tensorflow/tensorflow
   	GIT_TAG ${TENSORFLOW_COMMIT}
   	PREFIX "tensorflow"
   	CONFIGURE_COMMAND ./tensorflow/lite/tools/make/download_dependencies.sh
   	BUILD_IN_SOURCE 1
   	BUILD_COMMAND ${TF_COMMAND}
   	BUILD_BYPRODUCTS libtensorflow-lite.a ${CMAKE_BINARY_DIR}/tensorflow/src/tf/tensorflow/lite/tools/make/downloads/fft2d/fftsg.c
   	BUILD_BYPRODUCTS libtensorflow-lite.a ${CMAKE_BINARY_DIR}/tensorflow/src/tf/tensorflow/lite/tools/optimize/sparsity/format_converter.cc
   	INSTALL_COMMAND cp -f ${CMAKE_BINARY_DIR}/tensorflow/src/tf/tensorflow/lite/tools/make/gen/${TF_INSTALL_PREFIX}/lib/libtensorflow-lite.a ${CMAKE_BINARY_DIR}/
	
)

# libedgetpu has armv7l as armv7a.
if(${EDGE_ARCH} STREQUAL "armv7l")
    set(EDGE_ARCH "armv7a")
endif()

set(TF_ET_SRC_LIB "${CMAKE_SOURCE_DIR}/libedgetpu/out/direct/${EDGE_ARCH}/libedgetpu.so.1.0")
set(TF_LITE_LIB "${CMAKE_BINARY_DIR}/libtensorflow-lite.a")
include_directories(${CMAKE_SOURCE_DIR}/)
include_directories(${CMAKE_SOURCE_DIR}/src/)
include_directories(${CMAKE_SOURCE_DIR}/include/)
include_directories(${CMAKE_SOURCE_DIR}/libedgetpu/)
include_directories(${CMAKE_BINARY_DIR}/tensorflow/src/tf/)
include_directories(${CMAKE_BINARY_DIR}/tensorflow/src/tf/tensorflow/lite/tools/make/downloads/flatbuffers/include/)
include_directories(${CMAKE_BINARY_DIR}/glog_install/include)
link_directories(${CMAKE_BINARY_DIR})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/out/${EDGE_ARCH})

#find_package(Eigen3 3.0.0)
add_subdirectory(utils)
add_subdirectory(src)

#===== build minial =====#
file(GLOB SRC_FILES "app/minimal.cc")
add_executable(minimal ${SRC_FILES})
target_link_libraries(minimal gptpu ${TF_LITE_LIB} ${TF_ET_SRC_LIB})
add_dependencies(minimal gptpu tf)

#===== build openctpu =====#
include(FindBLAS)
file(GLOB SRC_FILES "app/openctpu.cc")
add_executable(openctpu ${SRC_FILES})
if (BLAS_FOUND)
	target_compile_options(openctpu PRIVATE ${BLAS_LINKER_FLAGS})
	target_link_libraries(openctpu ${BLAS_LIBRARIES})
endif ()
target_link_libraries(openctpu gptpu ${TF_LITE_LIB} ${TF_ET_SRC_LIB})
add_dependencies(openctpu gptpu tf)

