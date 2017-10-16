#!/bin/bash

# This is a wrapper file that helps you to run your script within our container.

# Examples of Usage (print Tensorflow's version within the container):
#   printf "import tensorflow \nprint('tensorflow version=',tensorflow.__version__)" > check_tf_version.py
#   ndrun -t tf-cu9-dnn7-py3 python3 check_tf_version.py 

# Add the flag -t, which let users to specify the image's tag name.
while getopts 't:' FLAG; do
  case "${FLAG}" in
    t) IMG_TAG="${OPTARG}" ;;
    *) error "Unexpected option ${FLAG}" ;;
  esac
done

# Intepreter such as Python/Python3 should be the second last argument.
INTEPRETER=$(echo "${@: -2:1}")

# Script to be executed should be the last argument.
EXE=$(echo "${@: -1}")

# Extract the path that contains the script to be executed.
HOST_VOL="$(dirname "$EXE")"

# If the path is not given explicitly,
# then the path is the current working directory.
if [ "$HOST_VOL" == '.' ] ;then
  HOST_VOL=$PWD
fi

# Extract the name of the script to be executed.
EXE="$(basename "$EXE")"

# The default working directory of our images is /notebook.
WORKSPACE=/notebook

# print the command that is going to be executed within the container.
echo "${INTEPRETER} ${WORKSPACE}/${EXE}"
# Command to initialize a temporary docker container
# in order to run the script.

# In order to use CNTK containers, we'll have to
# switch on the CNTK's virtual environment.
if [ "${IMG_TAG}" == "cntk-cu8-dnn6-py3" ] ;then
  SRC_CNTK="source /cntk/activate-cntk"
  nvidia-docker run \
  -it \
  -v ${HOST_VOL}:${WORKSPACE} \
  --rm \
  honghu/keras:${IMG_TAG} \
  "${SRC_CNTK} && ${INTEPRETER} ${WORKSPACE}/${EXE}"
else
  nvidia-docker run \
  -it \
  -v ${HOST_VOL}:${WORKSPACE} \
  --rm \
  honghu/keras:${IMG_TAG} \
  ${INTEPRETER} ${WORKSPACE}/${EXE}
fi
