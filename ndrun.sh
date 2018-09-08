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
#      ndrun - run a docker container for your deep-learning research
#
# SYNOPSIS
#      ndrun [-v] [-h] [-t IMG_TAG/IMG_TYPE] [-n NUM_GPUS] [-a ALIAS] [--extra EXTRA_OPTS]
#
# COMMAND-LINE OPTIONS
#
#      -t [IMG_TAG/IMG_TYPE]
#      --tag [IMG_TAG]
#      --type [IMG_TYPE]
#            The docker image is selected according to the given image's tag, [IMG_TAG]
#            or type, [IMG_TYPE].
#
#            The available tags can be found at:
#                  https://hub.docker.com/r/honghu/keras/tags/
#
#            The possible image types are:
#                  tensorflow, cntk, mxnet, theano
#
#            If you select an image via [IMG_TYPE], then, it means that you select the latest image
#            corresponds to that type. For example, --type tensorflow (-t tensorflow)
#            indicates that the image honghu/keras:tf-latest will be selected.
#
#            Remark:
#                  * if neither type nor tag is given, the image honghu/keras:tf-latest 
#                    will be selected by default
#                  * only images within the repository: honghu/keras can be selected
# 
#      -n, --num_gpus [NUM_GPUS]
#            [NUM_GPUS]: number of GPUs visible to the running container.
#
#            GPUs that are less busy will be selected automatically. Criteria for 'less busy': 
#            GPU utilization < 30 and has free RAM > 2048MB.
#
#            Remark:
#                  * the default value of [NUM_GPUS] is 1
#                  * GPUs can also be selected manually through setting the environment variable "NV_GPU".
#                    For example:
#                      # run "script.py" using GPU3,4
#                      NV_GPU=3,4 ndrun python3 script.py
#                      # the container will be closed and removed automatically 
#                      # once the script has completed running
#
#      -a, --alias [ALIAS]
#            Alias of the docker container can be provided. This option takes effect only
#            when the container runs in "detached mode".
#            (if you start Jupyter Notebook, it will run a container in detached mode, meaning that
#             Jupyter Notebook will keep alive in the background).
#
#      --extra [EXTRA_OPTS]
#            Extra options can be provided while running a container.
#            This is useful, especially if you'd like to do some extra port/volume mapping
#            between the host machine and the container. For example:
#
#                  # During the run-time of "script.py", an extra volume at [host_dir_path] (on the host machine),
#                  # will be mapped to [container_dir_path], which is a location on the container's side.
#                  ndrun --extra "-v [host_dir_path]:[container_dir_path]" -t mxnet python3 script.py
#                  # the above line is equivalent to the following docker command:
#                  # nvidia-docker run -it -v $PWD:/workspace --rm -v [host_dir_path]:[container_dir_path] \
#                  # honghu/keras:mx-latest python3 /workspace/script.py
#
#             Remark: 
#               * ndrun automatically maps "$PWD" (on the host side) to "/workspace" (on the container's side)
#                 for you. If you'd like to make extra volumes visible to the container, use
#                 --extra "-v [host_dir1]:[container_dir1] -v [host_dir2]:[container_dir2] ......" 
#
#      -v, --verbose
#            Entering the verbose mode.
#
#      -h, --help
#            Print a summary of the command-line options.
#
# Usage Examples
#
#      * print TensorFlow's version:
#
#            # creat a python script "check_tf_ver.py", which simply prints out the version of TensorFlow
#            printf "import tensorflow as tf \nprint('TF version=',tf.__version__)" > check_tf_ver.py
#            # execute "check_tf_ver.py", using the latest TensorFlow image
#            ndrun -t tensorflow python3 check_tf_ver.py
#
#      * print CNTK's version:
#
#            # creat a python script "check_cntk_ver.py", which simply prints out the version of MXNet
#            printf "import cntk \nprint('CNTK version=',cntk.__version__)" > check_cntk_ver.py
#            # execute "check_mxnet_ver.py", using the latest MXNet image
#            ndrun -t cntk python3 check_cntk_ver.py
#
#       * print GPUs' info:
#
#             # the GPUs exposed to the running container can be monitored using nvidia-smi.
#             # Say, we expose two GPUs to the TensorFlow container, and then check the status
#             # of the exposed GPUs from the container:
#             ndrun -t tensorflow -n 2 nvidia-smi
#
#       * open a container's BASH:
#
#             ndrun -t mxnet bash
#             # you can do any experiment there. Upon exiting, the container will be stopped and removed.
#
#       * open a container's IPython:
#
#             ndrun -t cntk ipython
#             # you can do any experiment there. Upon exiting, the container will be stopped and removed.
#
##########################################################################################################
#!/bin/bash
#
# Setting up some default values. By default, 1 GPU is to be used and the TensorFlow's container is selected. 
NUM_GPUS=1                # the default number of GPUs to be used
IMG_TAG='tf-latest'       # the default tag of the image to be running
PORT=8888                 # the default port to be forwarded inside the container
# By default, port 8888 will be opened as Jupyter Notebook starts.

# Remark: Jupyter Notebook starts only if there's no given command for running, For example:
#           # run a Jupyter Notebook from a MXNet container
#           ndrun -t mxnet

# Parsing the input arguments.
while getopts 'vht:n:p:a:u:-:' FLAG; do
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
    u) USR="${OPTARG}" ;;
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
         user)
             val="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
             USR=${val} ;;
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

# Display help upon request.
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

# Define some colors for verbose/error messages.
CL_RED='\033[1;31m'    # color (light red)
CL_GREEN='\033[1;32m'  # color (light green)
CL_BLUE='\033[1;34m'   # color (light blue)
CL_PURPLE='\033[1;35m' # color (light purple)
NC='\033[0m'           # no color

# This script supports NUM_GPUS=1, 2, 4 or 8.
if [ "${NUM_GPUS}" == "1" ] || \
   [ "${NUM_GPUS}" == "2" ] || \
   [ "${NUM_GPUS}" == "4" ] || \
   [ "${NUM_GPUS}" == "8" ] ;then
  :
else
  echo -e ${CL_RED}To let this script determine which GPUs to be used, the number of GPUs can only be 1, 2, 4 or 8.${NC}
  echo -e ${CL_RED}If you want to select GPUs on your own, you can pass NV_GPU to this script.${NC}
  echo -e ${CL_RED}As an example, the following command line runs a script using GPU0, GPU1 and GPU3:${NC}
  echo -e ${CL_RED}  NV_GPU=0,1,3 ndrun python3 script.py${NC}
  exit 1
fi

# If an username is provided, we will create an image for that user.
# Remark: if we don't do this, you'll be "root" once you've entered into the container, which is considered insecure. 
if [ ! -z "${USR}" ] ;then
  IMG_TAG_NEW_USER="${IMG_TAG}-${USR}"
  imgs_info=$(docker images honghu/keras)
  if [[ ${imgs_info} =~ "${IMG_TAG_NEW_USER}" ]] ;then
    echo -e ${CL_GREEN}The image honghu/keras:"${IMG_TAG_NEW_USER}" exists.${NC}
  else
    echo -e ${CL_GREEN}You have provided a username. We will now create a docker image for that user.${NC}
    dkfile_path="${HOME}/dockerfiles"
    if [ ! -d "${dkfile_path}" ] ;then
      mkdir -p ${dkfile_path}
    fi

    # Create the Dockerfile for the user.
    USR_ID=$(id -u ${USR})
    GRP_ID=$(id -g ${USR})
    printf "FROM honghu/keras:${IMG_TAG}\nRUN groupadd -g ${GRP_ID} ${USR} \nRUN useradd -rm -s /bin/bash -u ${USR_ID} -g ${GRP_ID} ${USR}\nUSER ${USR}\nENV HOME /home/${USR}" > ${dkfile_path}/Dockerfile
    if [[ ${IMG_TAG} =~ "mx" ]] ;then
      # Tell Keras to use MXNet as its backend.
      printf "\nRUN mkdir -p /home/${USR}/.keras && wget -O /home/${USR}/.keras/keras.json https://raw.githubusercontent.com/chi-hung/DockerKeras/master/keras-mxnet.json" >> ${dkfile_path}/Dockerfile
    elif [[ ${IMG_TAG} =~ "cntk" ]] ;then
      # Tell Keras to use CNTK as its backend.
      printf "\nRUN mkdir -p /home/${USR}/.keras && wget -O /home/${USR}/.keras/keras.json https://raw.githubusercontent.com/chi-hung/DockerKeras/master/keras-cntk.json" >> ${dkfile_path}/Dockerfile  
    elif [[ ${IMG_TAG} =~ "theano" ]] ;then
      # Tell Keras to use Theano as its backend.
      printf "\nRUN mkdir -p /home/${USR}/.keras && wget -O /home/${USR}/.keras/keras.json https://raw.githubusercontent.com/chi-hung/DockerKeras/master/keras-cntk.json" >> ${dkfile_path}/Dockerfile
      printf "\nRUN sed -i -e 's/cntk/theano/g' /home/${USR}/.keras/keras.json" >> ${dkfile_path}/Dockerfile
    fi

    # Setting-up Jupyter Notebook
      printf "\nRUN mkdir -p /home/${USR}/.jupyter && wget -O /home/${USR}/.jupyter/jupyter_notebook_config.py https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/tools/docker/jupyter_notebook_config.py" >> ${dkfile_path}/Dockerfile  

    # Switch to "/workspace".
    printf "\nWORKDIR /workspace" >> ${dkfile_path}/Dockerfile

    # Dockerfile is created. Accordingly, let's build the image.
    nvidia-docker build -t "honghu/keras:${IMG_TAG_NEW_USER}" ${dkfile_path}
  fi
  IMG_TAG=${IMG_TAG_NEW_USER}
fi

INTEPRETER=$1
# INTEPRETER could be, say, "python3", or "bash".

# Check if the intepreter is given.
if [ -z "$INTEPRETER" ];then
  echo -e ${CL_GREEN}An intepreter such as "python3" or "bash", is not given.${NC}
fi

# Script to be executed should be the second argument.
EXE=$2
if [ -z $EXE ];then
  echo -e ${CL_GREEN}You did not provide me the script you would like to execute.${NC}
fi

# Extract optional arguments of the script.
ARGS=$(echo $@ | cut -d' ' -f 3-)

CONTAINER_VOL=/workspace
HOST_VOL=${PWD}
# "/workspace" is a directory on the container side.
# "${PWD}" is the working directory on the host side.
# Later, we'll map "${PWD}" to "/workspace" such that
# data can be shared between host and container.

# Output some additional info if verbose is on.
CL='\033[1;32m'   # with color
NC='\033[0m'      # with no color
if [ "$VERBOSE" == "true" ] ;then
  # echo -e ${CL_GREEN}INTEPRETER="\t" $INTEPRETER${NC}
  # echo -e ${CL_GREEN}EXE="\t" $EXE${NC}
  # echo -e ${CL_GREEN}ARGS="\t" $ARGS${NC}
  echo -e ${CL_GREEN}'"'${HOST_VOL}'"'' (on the side of HOST) will be mounted to ''"'${CONTAINER_VOL}'"'' (on the side of CONTAINER).'${NC}
fi

# If the user did not inform us which GPUs to be used, we will pick GPUs which are less busy.
if [ -z $NV_GPU ] ;then

  # Criteria for 'less busy': GPU utilization < 30 and has free RAM > 2048MB.
  threash_util=30
  threash_free_mem=2048
  # The above criteria can be adjusted according to your needs.

  # Obtain GPU info.
  IFS=$''
  gpu_info=$(nvidia-smi --format=csv \
                        --query-gpu=utilization.gpu,memory.free | tail -n +2)
  
  # Obtain GPU utilization info.
  gpu_utils=$(echo $gpu_info | cut -d"," -f 1 | cut -d" " -f 1)
  
  # Obtain GPU free-memory info.
  gpu_free_mem=$(echo $gpu_info | cut -d"," -f 2 | cut -d" " -f 2)  
  IFS=$'\n'
  gpu_utils=(${gpu_utils})
  gpu_free_mem=(${gpu_free_mem})
  num_all_gpus=${#gpu_utils[*]}
  IFS=$' '
  if [ "$VERBOSE" == "true" ] ;then
    echo -e ${CL}This system has ${num_all_gpus} GPUs.${NC}
  fi

  # Find the available GPU(s), if 1 GPU(all GPUs) is(are) requested.
  if [ "${NUM_GPUS}" == "1" ] || [ "${NUM_GPUS}" == "${num_all_gpus}" ]; then
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
  # If 4 GPUs are requested, find 4 available GPUs.
  elif [ "${NUM_GPUS}" == "4" ] ;then
    confs=(0 1 2 3 4 5 6 7 \
           3 0 4 7 2 1 5 6 \
           0 1 5 4 3 2 6 7 \
           0 1 3 4 1 0 2 5 5 4 1 6 4 5 0 7 3 7 0 2 7 3 4 6 2 6 1 3 6 2 5 7)
           # Suppose that the system has NVLINK: 
           #   9 NVLINKs: (0123) and (4567). 
           #   7 NVLINKs: (3047) and (2156).
           #   6 NVLINKs: (0154) and (3267).
           #   5 NVLINKs: (0134),(1025),(5416),(4507),(3702),(7346),(2613),(6257).
    num_configs=$((${#confs[*]}/4))
    for ((j=0;j<${num_configs};j++))
    do
      id_gpu1=${confs[((4*j))]}
      id_gpu2=${confs[((4*j+1))]}
      id_gpu3=${confs[((4*j+2))]}
      id_gpu4=${confs[((4*j+3))]}
      if [ "${id_gpu1}" -ge "${num_all_gpus}" ] || [ "${id_gpu2}" -ge "${num_all_gpus}" ] || \
         [ "${id_gpu3}" -ge "${num_all_gpus}" ] || [ "${id_gpu4}" -ge "${num_all_gpus}" ] ;then
        break
      elif [ "${gpu_utils[ id_gpu1 ]}" -lt  "${threash_util}" ] && \
           [ "${gpu_utils[ id_gpu2 ]}" -lt  "${threash_util}" ] && \
           [ "${gpu_utils[ id_gpu3 ]}" -lt  "${threash_util}" ] && \
           [ "${gpu_utils[ id_gpu4 ]}" -lt  "${threash_util}" ] && \
           [ "${gpu_free_mem[ id_gpu1 ]}" -gt "${threash_free_mem}" ] && \
           [ "${gpu_free_mem[ id_gpu2 ]}" -gt "${threash_free_mem}" ] && \
           [ "${gpu_free_mem[ id_gpu3 ]}" -gt "${threash_free_mem}" ] && \
           [ "${gpu_free_mem[ id_gpu4 ]}" -gt "${threash_free_mem}" ] ;then
        NV_GPU=${id_gpu1},${id_gpu2},${id_gpu3},${id_gpu4}
        break
      fi
    done
  # if 2 GPUs are requested, find 2 available GPUs.
  elif [ "${NUM_GPUS}" == "2" ]; then

    topo=$(nvidia-smi topo -m) # check the topology: see if NVLINK exists
    if [[ $"topo" =~ "NV1" ]] ;then
      if [ "$VERBOSE" == "true" ] ;then
        echo "NVLINK Detected"
      fi
      pairs=(0 3 1 2 0 4 1 5 2 3 4 7 5 6 6 7 \
             0 1 0 2 1 3 4 5 4 6 5 7 2 6 3 7 \
             3 4 0 7 2 5 1 6 0 5 1 4 3 6 2 7 2 4 3 5 1 7 0 6) 
             # 2 NVLINKs: the first 8 pairs.
             # 1 NVLINK: the next 8 pairs.
             # 0 NVLINK: the rest 12 pairs.
      num_pairs=$((${#pairs[*]}/2))
      for ((j=0;j<${num_pairs};j++))
      do
        id_gpu1=${pairs[((2*j))]}
        id_gpu2=${pairs[((2*j+1))]}
        if [ "${id_gpu1}" -ge "${num_all_gpus}" ] || [ "${id_gpu2}" -ge "${num_all_gpus}" ] ;then
          break
        elif [ "${gpu_utils[ id_gpu1 ]}" -lt  "${threash_util}" ] && \
             [ "${gpu_utils[ id_gpu2 ]}" -lt  "${threash_util}" ] && \
             [ "${gpu_free_mem[ id_gpu1 ]}" -gt "${threash_free_mem}" ] && \
             [ "${gpu_free_mem[ id_gpu2 ]}" -gt "${threash_free_mem}" ] ;then
          NV_GPU=${id_gpu1},${id_gpu2}
          break
        fi
      done
    # else (NVLINK is not detected):
    else
      pairs=(0 1 2 3 4 5 6 7 0 2 0 3 1 2 1 3 4 6 4 7 5 6 5 7 \
             0 4 0 5 0 6 0 7 1 4 1 5 1 6 1 7 2 4 2 5 2 6 2 7 3 4 3 5 3 6 3 7) 
             # the first 12 pairs shall be faster than the rest 16 pairs on most of the servers & workstations
      num_pairs=$((${#pairs[*]}/2))
      for ((j=0;j<${num_pairs};j++))
      do
        id_gpu1=${pairs[((2*j))]}
        id_gpu2=${pairs[((2*j+1))]}
        if [ "${id_gpu1}" -ge "${num_all_gpus}" ] || [ "${id_gpu2}" -ge "${num_all_gpus}" ] ;then
          break
        elif [ "${gpu_utils[ id_gpu1 ]}" -lt  "${threash_util}" ] && \
             [ "${gpu_utils[ id_gpu2 ]}" -lt  "${threash_util}" ] && \
             [ "${gpu_free_mem[ id_gpu1 ]}" -gt "${threash_free_mem}" ] && \
             [ "${gpu_free_mem[ id_gpu2 ]}" -gt "${threash_free_mem}" ] ;then
          NV_GPU=${id_gpu1},${id_gpu2}
          break
        fi
      done
    fi
  fi
fi

if [ -z ${NV_GPU} ] ;then
  echo -e ${CL_RED}Error. No available GPUs are found!${NC}
  exit 1
else
  echo -e ${CL_GREEN}NV_GPU="${NV_GPU}"${NC}
  export NV_GPU
fi

# Start Jupyter Notebook if no script is passed for running.
if [ -z "${INTEPRETER}" ] && [ -z "${EXE}" ] ;then
  echo -e ${CL_GREEN}Starting Jupyter Notebook...${NC}

  if [ -z "${ALIAS}" ] ;then
    docker_cmd="nvidia-docker run -it \
                    -v ${HOST_VOL}:${CONTAINER_VOL} \
                    -p ${PORT}:8888 \
                    ${EXTRA_OPTS} \
                    -d honghu/keras:${IMG_TAG}"
    if [ "${VERBOSE}" == "true" ] ;then
      echo -e ${CL_PURPLE}${docker_cmd}${NC}
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
      echo -e ${CL_PURPLE}${docker_cmd}${NC}
    fi
    container_id=$($docker_cmd)  
  fi

  if [ "$?" == "0" ] ;then
    container_id=$(echo ${container_id} | head -n 1 | cut -c1-12)

    # Take a sleep for a while. This is necessary since it takes time for
    # Jupyter Notebook to start.
    sleep 1.5
    # After Jupyter Notebook starts, we are able to get its token.
    notebook_token=$(docker logs "${container_id}" | tail -n 1 | cut -d"=" -f 2)

    if [ -z ${ip} ]; then
      ip="localhost"
    fi
    # Tell the user how to connect to the Jupyter Notebook via the given token.
    echo ""
    echo -e ${NC} '*' To use Jupyter Notebook, open a browser and connect to the following address:${NC}
    echo -e ${CL_BLUE} "  http://${ip}:${PORT}/?token=${notebook_token}"${NC}
    echo -e ${NC} '*' To stop and remove this container, type:${NC}
    echo -e ${CL_BLUE} "  docker stop ${container_id} && docker rm ${container_id}"${NC}
    echo -e ${NC} '*' To enter into this container, type:${NC}
    echo -e ${CL_BLUE} "  docker exec -it ${container_id} bash"${NC}
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
    echo -e ${CL_PURPLE}${docker_cmd}${NC}
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
    echo -e ${CL_PURPLE}${docker_cmd}${NC}
  fi
  ${docker_cmd}
fi
