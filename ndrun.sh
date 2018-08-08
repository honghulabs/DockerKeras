###########################################################################################################
#
# Copyright 2018 Chi-Hung Weng
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
##########################################################################################################
#
# NAME
#      ndrun - activate and use different docker images for deep learning research
#
# SYNOPSIS
#      ndrun [-v] [-h] [-t IMG_TAG/IMG_TYPE] [-n NUM_GPUS] [-a ALIAS] [--extra EXTRA_OPTS]
#
# COMMAND-LINE OPTIONS
#
#      -t [IMG_TAG/IMG_TYPE]
#      --tag [IMG_TAG]
#      --type [IMG_TYPE]
#            The image to be activated is chosen according to the given image's tag, [IMG_TAG]
#            or type, [IMG_TYPE].
#
#            The available tags can be found at:
#                  https://hub.docker.com/r/honghu/keras/tags/
#
#            The possible values of [IMG_TYPE] are:
#                  tensorflow, cntk, mxnet, theano
#
#            If you provide the type of image, then, the latest image of
#            the specified framework will be activated.
#
#            Remark:
#                  * the defult image's type is TensorFlow
#                  * the chosen image must belong to the repository: honghu/keras
# 
#      -n, --num_gpus [NUM_GPUS]
#            Set [NUM_GPUS], which is the number of GPUs to be used by the running image.
#
#            GPUs that are less busy will be selected.
# 
#            [NUM_GPUS] is set as 1 by default (single GPU mode).
#
#            Say, instead looking for 2 available GPUs automatically, one can pick up 2 GPUs manually by
#            specifying the environment variable NV_GPU. For example:
#
#                  # initalize a container that will run xxx.py using GPU3,4
#                  NV_GPU=3,4 ndrun python3 xxx.py
#                  # the container will be closed automatically once the run is finished
#
#      -a, --alias [ALIAS]
#            Alias of the docker container can be provided. This option takes effect only
#            when the image runs at the daemoen mode (Jupyter Notebook).
#
#      --extra [EXTRA_OPTS]
#            Extra docker options can be provided while activating the docker image.
#            This is especially useful, if you'd like to do some extra port/volume mapping
#            between the host machine and the docker container. For example:
#
#                  # Run the script (xxx.py). During the run, a volume at the host machine, i.e. 
#                  # at [host_dir_path], will be mounted as a volume locating at [container_dir_path], 
#                  # which is a path at the side of the docker container.
#                  ndrun --extra "-v [host_dir_path]:[container_dir_path]" -t mxnet python3 xxx.py
#
#      -v, --verbose
#            Entering the verbose mode.
#
#      -h, --help
#            Print a summary of the command-line options.
#
# Usage Example
#
#      print TensorFlow's version:
#
#            # creat a python script "check_tf_ver.py", which simply print out the version of TensorFlow
#            printf "import tensorflow as tf \nprint('TF version=',tf.__version__)" > check_tf_ver.py
#            # execute "check_tf_ver.py", using the latest TensorFlow image
#            ndrun -t tensorflow python3 check_tf_ver.py
#
#      print MXNet's version:
#
#            # creat a python script "check_mxnet_ver.py", which simply print out the version of TensorFlow
#            printf "import mxnet as mx \nprint('MXNet version=',mx.__version__)" > check_mxnet_ver.py
#            # execute "check_mxnet_ver.py", using the latest MXNet image
#            ndrun -t mx-cu9-dnn7-py3-18.03 python3 check_mxnet_ver.py
#
#       print info of the GPUs:
#
#             # The GPUs exposed to the running container can be monitored using nvidia-smi.
#             ndrun -t tensorflow -n 2 nvidia-smi
#
##########################################################################################################
#!/bin/bash
#
# Setting up default values.
# By default, 1 GPU is to be used and the TensorFlow's docker image will be selected. 
NUM_GPUS=1                # the default number of GPUs to be used
IMG_TAG='tf-latest'      # the default tag of the image to be running
PORT=8888                 # the default port to be forwarded inside the container
# By default, port 8888 will be opened as Jupyter Notebook starts.
# Remark: Jupyter Notebook starts only if there's no given script.

# The above variables can be changed via the optional arguments -n, -t and -p.
while getopts 'vht:n:p:a:-:' FLAG; do
  case "${FLAG}" in
    v) VERBOSE=true ;;
    h) HELP="true" ;;
    t) IMG_TYPE="${OPTARG}" 
       # Decide which tag of honghu/keras to be used later.
       case "${IMG_TYPE}" in
         "tensorflow") IMG_TAG='tf-latest' ;;
               "cntk") IMG_TAG='cntk-latest';;
              "mxnet") IMG_TAG='mx-latest' ;;
             "theano") IMG_TAG='theano-latest' ;;
                    *) IMG_TAG=${IMG_TYPE} ;;
       esac ;;
    n) NUM_GPUS="${OPTARG}" ;;
    p) PORT="${OPTARG}" ;;
    a) ALIAS="${OPTARG}" ;;
    -)
       case "${OPTARG}" in
         verbose) VERBOSE=true ;;
         help) HELP="true" ;;
         type)
             val="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
             IMG_TYPE=${val} ;;
         tag)
             val="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
             IMG_TAG=${val} ;;
         num_gpus)
             val="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
             NUM_GPUS=${val} ;;  
         port)
             val="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
             PORT=${val} ;;
         alias)
             val="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
             ALIAS=${val} ;;
         extra)
             val="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
             EXTRA_OPTS=${val} ;;
         *)
             if [ "$OPTERR" = 1 ] && [ "${optspec:0:1}" != ":" ]; then
                 echo "Error! Unknown option --${OPTARG}" >&2
                 exit 1
             fi ;;
       esac ;;
    *) if [ "$OPTERR" = 1 ] && [ "${optspec:0:1}" != ":" ]; then
                 echo "Error! Unknown option is given!" >&2
                 exit 1
             fi ;;
  esac
done
shift $((OPTIND-1))

# Display help.
if [ "$HELP" == "true" ] ;then
  bottom=$(grep -n "Usage Example" $0)
  bottom_line_num=$((${bottom:0:2}))
  up=$(grep -n "SYNOPSIS" $0)
  up_line_num=$((${up:0:2}))

  diff_lines=$((bottom_line_num-up_line_num+1))

  info=$(cat /sbin/ndrun | head -n $(($bottom_line_num-1)))
  info=$(echo "$info" | cut -b 2-)
  info=$(echo "$info" | tail -n $(($diff_lines)))
  echo "$info"
  exit 0
fi

# Define colors for verbose/error messages
CL_RED='\033[1;31m'    # color (light red)
CL_GREEN='\033[1;32m'  # color (light green)
CL_BLUE='\033[1;34m'   # color (light blue)
CL_PURPLE='\033[1;35m' # color (light purple)
NC='\033[0m'           # no color

# This script supports only NUM_GPUS=1, 2, 4 or 8.
if [ "${NUM_GPUS}" == "1" ] || \
   [ "${NUM_GPUS}" == "2" ] || \
   [ "${NUM_GPUS}" == "4" ] || \
   [ "${NUM_GPUS}" == "8" ] ;then
  :
else
  echo -e ${CL_RED}Error. Currently, this script supports only number of GPUs = 1, 2, 4 or 8.${NC}
  exit 1
fi
# As the optional arguments are shifted away, intepreter such as
# Python/Python3 should be the first argument.
INTEPRETER=$1

# Check if the intepreter is given.
if [ -z "$INTEPRETER" ];then
  echo -e ${CL_GREEN}An intepreter such as python/python3, is not given.${NC}
fi

# Script to be executed should be the second argument.
EXE=$2
if [ -z $EXE ];then
  echo -e ${CL_GREEN}You did not provide me the script you wish to execute.${NC}
fi

# Extract optional arguments of the script.
ARGS=$(echo $@ | cut -d' ' -f 3-)

CONTAINER_VOL=/workspace
HOST_VOL=${PWD}
# $CONTAINER_VOL is a volume on the container side.
# $HOST_VOL is a volume on the host side.
# Later, we will mount $HOST_VOL to $CONTAINER_VOL automatically such that data
# can be shared between both sides.

# Output some additional info if verbose is on.
CL='\033[1;32m'   # with color
NC='\033[0m'      # with no color
if [ "$VERBOSE" == "true" ] ;then
  # echo -e ${CL_GREEN}INTEPRETER="\t" $INTEPRETER${NC}
  # echo -e ${CL_GREEN}EXE="\t" $EXE${NC}
  # echo -e ${CL_GREEN}ARGS="\t" $ARGS${NC}
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
  echo -e ${CL_RED}Error. No enough GPUs available!${NC}
  exit 1
else
  echo -e ${CL_GREEN}NV_GPU="${NV_GPU}"${NC}
fi

# Start the Jupyter Notebook if no script is passed for running.
if [ -z "${INTEPRETER}" ] && [ -z "${EXE}" ] ;then
  echo -e ${CL_GREEN}Starting Jupyter Notebook...${NC}

  if [ -z "${ALIAS}" ] ;then
    docker_cmd="nvidia-docker run -it \
                    -v ${HOST_VOL}:${CONTAINER_VOL} \
                    -p ${PORT}:8888 \
                    ${EXTRA_OPTS} \
                    -d honghu/keras:${IMG_TAG}"
    if [ "${VERBOSE}" == "true" ] ;then
      echo -e ${CL_PURPLE}${docker_cmd}
    fi
    container_id=$($docker_cmd)  
  else
    docker_cmd="nvidia-docker run -it \
                    --name ${ALIAS} \
                    -v ${HOST_VOL}:${CONTAINER_VOL} \
                    -p ${PORT}:8888 \
                    ${EXTRA_OPTS} \
                    -d honghu/keras:${IMG_TAG}"
    if [ "${VERBOSE}" == "true" ] ;then
      echo -e ${CL_PURPLE}${docker_cmd}
    fi
    container_id=$($docker_cmd)  
  fi

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
    echo ""
    echo -e ${NC} '*' To use Jupyter Notebook, open a browser and connect to the following address:${NC}
    echo -e ${CL_BLUE} "  http://${my_ip}:${PORT}/?token=${notebook_token}"${NC}
    echo -e ${NC} '*' To stop and remove this daemon container, type:${NC}
    echo -e ${CL_BLUE} "  docker stop ${container_id} && docker rm ${container_id}"${NC}
    echo -e ${NC} '*' To enter into this docker container, type:${NC}
    echo -e ${CL_BLUE} "  docker exec -it ${container_id} /bin/bash"${NC}
  else
    echo -e ${CL_RED}An error has occured.
    exit 1
  fi

# Or, if a command is given, execute it.
elif [ -z $EXE ] ;then
  CMD=${INTEPRETER}
  echo -e ${CL_GREEN}Executing the given command...${NC}

  docker_cmd="nvidia-docker run -it \
                    -v ${HOST_VOL}:${CONTAINER_VOL} \
                    --rm \
                    ${EXTRA_OPTS} \
                    honghu/keras:${IMG_TAG} \
                    ${CMD}"
  if [ "${VERBOSE}" == "true" ] ;then
    echo -e ${CL_PURPLE}${docker_cmd}
  fi
  ${docker_cmd}
  

# Or, if a script is passed for running, run it.
else
  docker_cmd="nvidia-docker run -it \
                    -v ${HOST_VOL}:${CONTAINER_VOL} \
                    --rm \
                    ${EXTRA_OPTS} \
                    honghu/keras:${IMG_TAG} \
                    ${INTEPRETER} ${CONTAINER_VOL}/${EXE} $ARGS"
  if [ "${VERBOSE}" == "true" ] ;then
    echo -e ${CL_PURPLE}${docker_cmd}
  fi
  ${docker_cmd}
fi
