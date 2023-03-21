ARG platform
ARG python_use

FROM ubuntu:20.04 AS build
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone && \
    apt-get update && \
    apt-get install -y tzdata make vim cmake python3 g++ git curl unzip zip libusb-1.0-0

# use default sources.list setup for libedgetpu
# reference: google-coral/libedgetpu/docker/Dockerfile
FROM build AS source_default
#COPY update_sources.sh /
#RUN /update_sources.sh

# aarch64 platform: edgetpu_compiler is not supported anymore.
COPY edgetpu_compiler /usr/bin

# alternative 1: use jetson-nano default sources.list
FROM build AS source_nano
COPY nano_sources.list /etc/apt/
#RUN cat /etc/apt/sources.list

# libedgetpu 
FROM source_${platform} AS build_no_python_part
RUN dpkg --add-architecture armhf
RUN dpkg --add-architecture arm64

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libc6-dev:arm64 \
    libc6-dev:armhf \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-all \
    python3-numpy \
    build-essential \
    crossbuild-essential-armhf \
    crossbuild-essential-arm64 \
    libusb-1.0-0-dev \
    libusb-1.0-0-dev:arm64 \
    libusb-1.0-0-dev:armhf \
    zlib1g-dev \
    zlib1g-dev:armhf \
    zlib1g-dev:arm64 \
    sudo \
    debhelper \
    pkg-config \
    zip \
    unzip \
    curl \
    wget \
    git \
    tree \
    software-properties-common \
    $(grep Ubuntu /etc/os-release > /dev/null && echo vim-common || echo xxd)

ARG BAZEL_VERSION=4.2.2
RUN wget -O /bazel https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh && \
    bash /bazel && \
    rm -f /bazel
RUN apt-get install -y libeigen3-dev && \
    sudo ln -s /usr/include/engin3/Eigen /usr/include/Eigen

#install edgetpu_compiler, not supportd on jetson-nano from official
#FROM build_no_python_part AS build2
#RUN apt-get install -y apt-transport-https ca-certificates && \ 
#    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add - && \
#    echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list && \
#    sudo apt-get update && \
#    sudo apt-get install edgetpu-compiler

# version-specific fixes
# mainly install tf=1.13.1 and python3.7
FROM build_no_python_part AS build_with_python_part
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.7 python3.7-distutils && \
    apt-get install -y python3-pip && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1 && \
    python3 -c 'import sys; print(sys.version_info[:])' && \
    pip3 install tensorflow==1.13.1 && \
    pip3 install keras && \
    pip3 install numpy --upgrade && \
    pip3 install protobuf==3.20.* && \
    apt-get install -y python3.7-dev

#install flatbuffers-compiler & misc
FROM build_${python_use}_python_part AS build4
RUN apt-get update && \
    apt-get install -y flatbuffers-compiler && \    
    apt-get install -y libopenblas-dev && \
    apt-get install -y libboost-all-dev

# final stage
FROM build4 AS FINAL
WORKDIR /home
RUN ln -sf $(which python3) /usr/bin/python && \
    echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/libedgetpu/out/direct/k8" >> ~/.bashrc

