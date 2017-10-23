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
# 
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
IMG_TAG='tf-cu9-dnn7-py3'     # tag of the image to be running

# The above variables can be changed via the optional arguments -t and -n.
while getopts 'vt:n:' FLAG; do
  case "${FLAG}" in
    v) VERBOSE=true ;;
    t) IMG_TAG="${OPTARG}" ;;
    n) NUM_GPUS="${OPTARG}" ;;
    *) error "Unexpected option ${FLAG}" ;;
  esac
done
shift $((OPTIND-1))

# Allowing the use of "-t cntk" rather than "-t cntk-cu8-dnn6-py3".
case "${IMG_TAG}" in
  "tensorflow") IMG_TAG='tf-cu9-dnn7-py3' ;;
        "cntk") IMG_TAG='cntk-cu8-dnn6-py3' ;;
       "mxnet") IMG_TAG='mxnet-cu9-dnn7-py3' ;;
      "theano") IMG_TAG='theano-cu9-dnn7-py3' ;;
esac

# As the optional arguments are shifted away, intepreter such as
# Python/Python3 should be the first argument.
INTEPRETER=$1

CL_RED='\033[1;31m'   # color (red)
CL_GREEN='\033[1;32m' # color (green)
NC='\033[0m'          # no color

# Check if the intepreter is given.
if [ -z $INTEPRETER ];then
  echo -e ${CL_RED} Error. You need to specify an intepreter in order to run the script.${NC}
  exit 1
fi

# Script to be executed should be the second argument.
EXE=$2
if [ -z $EXE ];then
  echo -e ${CL_RED} Error. You should offer me the name of script you would like to execute.${NC}
  exit 1
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
  
  # extract GPU utilization info.
  gpu_utils=$(echo $gpu_info | cut -d"," -f 1 | cut -d" " -f 1)
  
  # extract GPU free-memory info.
  gpu_free_mem=$(echo $gpu_info | cut -d"," -f 2 | cut -d" " -f 2)  
  IFS=$'\n'
  gpu_utils=(${gpu_utils})
  gpu_free_mem=(${gpu_free_mem})
  num_all_gpus=${#gpu_utils[*]}
  IFS=$' '
  if [ "$VERBOSE" == "true" ] ;then
    echo -e ${CL}This system has ${num_all_gpus} GPUs.${NC}
  fi
  
  # pick GPUs which are less busy.
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

# Run the script.
# Notice that, in order to use the CNTK's docker image, we'll have to
# switch on the CNTK's virtual environment.
if [[ "${IMG_TAG}" =~ ^cntk.* ]] ;then

  SRC_CNTK="source /cntk/activate-cntk"
  nvidia-docker run -it \
                    -v ${HOST_VOL}:${CONTAINER_VOL} \
                    --rm \
                    honghu/keras:${IMG_TAG} \
                    "${SRC_CNTK} && ${INTEPRETER} ${CONTAINER_VOL}/${EXE} $ARGS"

else

  nvidia-docker run -it \
                    -v ${HOST_VOL}:${CONTAINER_VOL} \
                    --rm \
                    honghu/keras:${IMG_TAG} \
                    ${INTEPRETER} ${CONTAINER_VOL}/${EXE} $ARGS

fi
