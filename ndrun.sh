#!/bin/bash

# This is a wrapper file that helps you to run your script within our container.
#
# Example of Usage (print Tensorflow's version within the container):
#   printf "import tensorflow \nprint('tensorflow version=',tensorflow.__version__)" > check_tf_version.py
#   ndrun python3 check_tf_version.py 

# By default, we will use single GPU and choose Tensorflow as our backend. 
IMG_TAG='tf-cu9-dnn7-py3'
NUM_GPUS=1

# The value of $IMG_TAG and $NUM_GPUS can be changed through the input arguments, via the flag -t and -n.
while getopts 'vt:n:' FLAG; do
  case "${FLAG}" in
    v) VERBOSE=true ;;
    t) IMG_TAG="${OPTARG}" ;;
    n) NUM_GPUS="${OPTARG}" ;;
    *) error "Unexpected option ${FLAG}" ;;
  esac
done
shift $((OPTIND-1))

# After shifting away the optional arguments, intepreter such as Python/Python3 should now be the first argument.
INTEPRETER=$1
# Script to be executed should now be the second argument.
EXE=$2
# Extract flags for the script to be executed, if any.
ARGS=$(echo $@ | cut -d' ' -f 3-)

CONTAINER_VOL=/notebooks
HOST_VOL=${PWD}
# $CONTAINER_VOL is a volume on the container side.
# $HOST_VOL is a volume on the host side.
# Later, we will mount $HOST_VOL to $CONTAINER_VOL in order to share data between host and container.


CL='\033[1;32m'   # default color
NC='\033[0m'      # no color
if [ "$VERBOSE" == "true" ] ;then
  echo -e ${CL}intepreter= $INTEPRETER${NC}
  echo
  echo -e ${CL}name of the script to be executed= $EXE${NC}
  echo
  echo -e ${CL}args of the script to be executed= $ARGS${NC}
  echo
  echo -e ${CL}${HOST_VOL}' (on the side of host) will be mounted to '${CONTAINER_VOL}' (on the side of container).'${NC}
  echo
fi

# Multi-GPU topology: if 4 GPUs are used and you have DGX-1 or similar, due to its topology, it's faster to use GPU 0,1,3,4 instead of 0,1,2,3.
if [ "${NUM_GPUS}" == "4" ];then
  export NV_GPU=0,1,3,4
else
  export NV_GPU=$(seq -s , 0 $(($NUM_GPUS-1)))
fi

# Command to initialize a temporary docker container
# in order to run the script.

# In order to use CNTK containers, we'll have to
# switch on the CNTK's virtual environment.
if [ "${IMG_TAG}" == "cntk-cu8-dnn6-py3" ] ;then
  SRC_CNTK="source /cntk/activate-cntk"

  nvidia-docker run \
  -it \
  -v ${HOST_VOL}:${CONTAINER_VOL} \
  --rm \
  honghu/keras:${IMG_TAG} \
  "${SRC_CNTK} && ${INTEPRETER} ${CONTAINER_VOL}/${EXE} $ARGS"
else
  nvidia-docker run \
  -it \
  -v ${HOST_VOL}:${CONTAINER_VOL} \
  --rm \
  honghu/keras:${IMG_TAG} \
  ${INTEPRETER} ${CONTAINER_VOL}/${EXE} $ARGS
fi
