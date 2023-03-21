#!/bin/sh

GITTOP="$(git rev-parse --show-toplevel 2>&1)"

# source configure file
. "${GITTOP}/configure.cfg"

CONTAINER_NAME=${PROJ_NAME}_container

docker exec -it ${CONTAINER_NAME} bash
#docker exec -it -u $(id -u):$(id -g) ${CONTAINER_NAME} bash
