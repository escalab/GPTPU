#!/bin/sh

GITTOP="$(git rev-parse --show-toplevel 2>&1)"

# source configure file
. "${GITTOP}/configure.cfg"

# paths setup
PROJ=${PROJ_NAME}
DOCKERFILE_PATH="${GITTOP}/docker"
IMAGE_NAME=${PROJ}_image
CONTAINER_NAME=${PROJ}_container
SRC_DIR="${GITTOP}"
SRC_TARGET_DIR="${SRC_MOUNT}"

echo "[${PROJ}] - building docker image from dockerfile..."
PLATFORM=${1:-default}
PYTHON_USE="with"
BUILDKIT=0
if [ ${PLATFORM} = "nano" ]
then
    PYTHON_USE="no"
    BUILDKIT=1
fi

# use DOCKER_BUILDKIT=1 to skip unused stage(s)
# minimum required docker version: 18.09+
DOCKER_BUILDKIT=${BUILDKIT} docker build \
                --build-arg platform=${PLATFORM} \
                --build-arg python_use=${PYTHON_USE} \
                -t ${IMAGE_NAME} ${DOCKERFILE_PATH}

docker stop ${CONTAINER_NAME}
docker rm ${CONTAINER_NAME}

# generate container
echo "[${PROJ}] - building docker container..."
docker run -d \
        -it \
        --privileged \
        --name ${CONTAINER_NAME} \
        --mount type=bind,source=${SRC_DIR},target=${SRC_TARGET_DIR} \
        ${IMAGE_NAME} \
        bash

