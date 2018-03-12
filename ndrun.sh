# Copyright 2017 Chi-Hung Weng
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
#
# This is a wrapper file that helps you to run your script using the docker images prepared by us.
#
# Example of Usage (print CNTK's version inside the running image):
#   printf "import cntk \nprint('cntk version=',cntk.__version__)" > check_cntk_version.py
#   ndrun -v -t cntk python3 check_cntk_version.py
# 
# Options available:
#   -t: tag of the image (e.g. tf-cu9-dnn7-py3, cntk-cu8-dnn6-py3)
#       or type of the image (e.g. tensorflow, cntk).
#   -n: number of GPUs to be used.
#   -v: verbose mode.
#

# By default, 1 GPU is to be used and the TensorFlow's docker image will be chosen. 
NUM_GPUS=1                    # number of GPUs to be used
IMG_TYPE='tensorflow'         # type of the image to be running

# By default, port 8888 will be opened as Jupyter Notebook starts.
# Remark: Jupyter Notebook starts only if there's no script given.
PORT=8888

# The above variables can be changed via the optional arguments -t, -n and -p.
while getopts 'vt:n:p:' FLAG; do
  case "${FLAG}" in
    v) VERBOSE=true ;;
    t) IMG_TYPE="${OPTARG}" ;;
    n) NUM_GPUS="${OPTARG}" ;;
    p) PORT="${OPTARG}" ;;
    *) error "Unexpected option ${FLAG}" ;;
  esac
done
shift $((OPTIND-1))

# Decide which tag of honghu/keras to be used later.
case "${IMG_TYPE}" in
  "tensorflow") IMG_TAG='tf-cu9-dnn7-py3-avx2-18.03' ;;
        "cntk") IMG_TAG='cntk-cu9-dnn7-py3-18.03' ;;
       "mxnet") IMG_TAG='mx-cu9-dnn7-py3-18.03' ;;
      "theano") IMG_TAG='theano-cu9-dnn7-py3-18.03' ;;
             *) IMG_TAG= ${IMG_TYPE} ;;
esac

# Define colors for verbose/error messages
CL_RED='\033[1;31m'   # color (light red)
CL_GREEN='\033[1;32m' # color (light green)
CL_BLUE='\033[1;34m'  # color (light blue)
NC='\033[0m'          # no color

# This script supports only NUM_GPUS=1, 2, 4 or 8.
if [ "${NUM_GPUS}" == "1" ] || \
   [ "${NUM_GPUS}" == "2" ] || \
   [ "${NUM_GPUS}" == "4" ] || \
   [ "${NUM_GPUS}" == "8" ] ;then
  :
else
  echo -e ${CL_RED} Error. Currently, this script supports only number of GPUs = 1, 2, 4 or 8.${NC}
  exit 1
fi
# As the optional arguments are shifted away, intepreter such as
# Python/Python3 should be the first argument.
INTEPRETER=$1

# Check if the intepreter is given.
if [ -z $INTEPRETER ];then
  echo -e ${CL_GREEN} An intepreter such as python/python3, is not given.${NC}
fi

# Script to be executed should be the second argument.
EXE=$2
if [ -z $EXE ];then
  echo -e ${CL_GREEN} You did not provide me the script you wish to execute.${NC}
fi

# Extract optional arguments of the script.
ARGS=$(echo $@ | cut -d' ' -f 3-)

CONTAINER_VOL=/notebooks
HOST_VOL=${PWD}
# $CONTAINER_VOL is a volume on the container side.
# $HOST_VOL is a volume on the host side.
# Later, we will mount $HOST_VOL to $CONTAINER_VOL in order to share data
# between both of these sides.

# Output some additional info if verbose is on.
CL='\033[1;32m'   # with color
NC='\033[0m'      # with no color
if [ "$VERBOSE" == "true" ] ;then
  echo -e ${CL_GREEN}Intepreter="\t" $INTEPRETER${NC}
  echo -e ${CL_GREEN}Script to run="\t" $EXE${NC}
  echo -e ${CL_GREEN}Script Args="\t" $ARGS${NC}
  echo -e ${CL_GREEN}'"'${HOST_VOL}'"'' (on the side of Host) will be mounted to ''"'${CONTAINER_VOL}'"'' (on the side of Container).'${NC}
fi

# If the user did not inform us which GPUs to be used, we will pick GPUs which are less busy.
if [ -z $NV_GPU ] ;then

  # Criterion for 'less busy': GPU utilization < 30 and has free memory > 2048MB.
  # This criterion can be adjusted according to your needs.
  threash_util=30
  threash_free_mem=2048
  
  # Retrieve GPU info.
  IFS=$''
  gpu_info=$(nvidia-smi --format=csv \
                        --query-gpu=utilization.gpu,memory.free \
              | tail -n +2)
  
  # Extract GPU utilization info.
  gpu_utils=$(echo $gpu_info | cut -d"," -f 1 | cut -d" " -f 1)
  
  # Extract GPU free-memory info.
  gpu_free_mem=$(echo $gpu_info | cut -d"," -f 2 | cut -d" " -f 2)  
  IFS=$'\n'
  gpu_utils=(${gpu_utils})
  gpu_free_mem=(${gpu_free_mem})
  num_all_gpus=${#gpu_utils[*]}
  IFS=$' '
  if [ "$VERBOSE" == "true" ] ;then
    echo -e ${CL}This system has ${num_all_gpus} GPUs.${NC}
  fi
  
  # Pick GPUs which are less busy.
  for (( j=0; j< $((${num_all_gpus}/${NUM_GPUS})); j++ ))
  do
    gpu_ids=$(seq -s " " $((${NUM_GPUS}*j)) $(( ${NUM_GPUS}*(j+1)-1   )) )
    count_not_use=0

    for gpu_id in ${gpu_ids}
    do
      if [ "${gpu_utils[${gpu_id}]}" -lt "${threash_util}" ] && \
         [ "${gpu_free_mem[${gpu_id}]}" -gt "${threash_free_mem}" ]; then
        count_not_use=$((${count_not_use}+1))
      fi
    done
    if [ "${count_not_use}" == "${NUM_GPUS}" ]; then
      NV_GPU=${gpu_ids//" "/,}
      break
    fi
  done

  # The following lines check some more configurations for number of GPUs to be used =2.
  if [ -z "${NV_GPU}" ] && [ "${NUM_GPUS}" == "2" ]; then
    pairs=(0 2 0 3 1 2 1 3 4 6 4 7 5 6 5 7)
    num_pairs=$((${#pairs[*]}/2))
    for ((j=0;j<${num_pairs};j++))
    do
      id_gpu1=${pairs[((2*j))]}
      id_gpu2=${pairs[((2*j+1))]}
      if [ "${gpu_utils[ id_gpu1 ]}" -lt  "${threash_util}" ] && \
         [ "${gpu_utils[ id_gpu2 ]}" -lt  "${threash_util}" ] && \
         [ "${gpu_free_mem[ id_gpu1 ]}" -gt "${threash_free_mem}" ] && \
         [ "${gpu_free_mem[ id_gpu2 ]}" -gt "${threash_free_mem}" ] ;then
        NV_GPU=${id_gpu1},${id_gpu2}
        break
      fi
    done
  fi

fi

export NV_GPU

if [ -z ${NV_GPU} ] ;then
  echo -e ${CL_RED} Error. No enough GPUs available!${NC}
  exit 1
elif [ "${VERBOSE}" == "true" ] ;then
  echo -e ${CL_GREEN}NV_GPU=${NV_GPU}${NC}
fi

# Start the Jupyter Notebook if no script is passed for running.
if [ -z ${INTEPRETER} ] && [ -z ${EXE} ] ;then
  echo -e ${CL_GREEN}Starting Jupyter Notebook...${NC}
  echo -e NV_GPU="${NV_GPU}"${NC}
  container_id=$(nvidia-docker run -it \
                    -v ${HOST_VOL}:${CONTAINER_VOL} \
                    -p ${PORT}:8888 \
                    -d \
                    --rm \
                    honghu/keras:${IMG_TAG} )
  if [ "$?" == "0" ] ;then
    container_id=$(echo ${container_id} | head -n 1 | cut -c1-12)

    # Sleep for a while. This is necessary since it takes time for
    # Jupyter Notebook to start.
    if [[ "${IMG_TAG}" =~ ^cntk.* ]] ;then
      sleep 2.5
    else
      sleep 1.5
    fi
    # After Jupyter Notebook starts, we are able to get its token.
    notebook_token=$(docker logs "${container_id}" | tail -n 1 | cut -d"=" -f 2)

    if [ -z ${my_ip} ]; then
      my_ip="localhost"
    fi
    # Tell the user how to connect to the Jupyter Notebook via the given token.
    echo -e ${NC} '*' To use Jupyter Notebook, open a browser and connect to the following address:${NC}
    echo -e ${CL_BLUE} "  http://${my_ip}:${PORT}/?token=${notebook_token}"${NC}
    if [ "${my_ip}" == "localhost" ] ;then
      echo -e ${NC} ' ' Replace '"'localhost'"' to the IP address \
                    that is visible to other computers, if you are not coming from localhost.${NC}
    fi
    echo -e ${NC} '*' To stop and remove this docker daemon, type:${NC}
    echo -e ${CL_BLUE} "  docker stop ${container_id}"${NC}
  else
    echo -e ${CL_RED} An error occured. It"'"s likely that the port you have specified is already in use.${NC}
    echo -e ${CL_RED} Use the option -p to use another port.${NC}
    exit 1
  fi

# Or, if a script is passed for running, run it.
# Notice that, in order to use CNTK's docker image, we'll have to
# switch on CNTK's virtual environment.
elif [[ "${IMG_TAG}" =~ ^cntk.* ]] ;then

  SRC_CNTK="source /cntk/activate-cntk"
  nvidia-docker run -it \
                    -v ${HOST_VOL}:${CONTAINER_VOL} \
                    --rm \
                    honghu/keras:${IMG_TAG} \
                    -c "${SRC_CNTK} && ${INTEPRETER} ${CONTAINER_VOL}/${EXE} $ARGS"
else
  nvidia-docker run -it \
                    -v ${HOST_VOL}:${CONTAINER_VOL} \
                    --rm \
                    honghu/keras:${IMG_TAG} \
                    ${INTEPRETER} ${CONTAINER_VOL}/${EXE} $ARGS
fi
