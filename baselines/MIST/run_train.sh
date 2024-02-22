#!/bin/sh

cd $SLURM_SUBMIT_DIR

echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_VERSION=$CUDA_VERSION"

export NCCL_DEBUG=INFO NCCL_P2P_DISABLE=1 NCCL_P2P_LEVEL=5

ml restore cuda11_0
while [[ $# -gt 1 ]]
    do
        key="$1"
        case $key in
            -c|--config_path)
            CONFIG_PATH="$2"
            shift # past argument
            ;;
            -o|--out_dir)
            OUTPUT_DIR="$2"
            shift # past argument
            ;;
            #-cluster|--n_cluster)
            #C="$2"
            #shift
            #;;
            -nms|--nms_iou)
            N="$2"
            shift
            ;;
            -lmda|--lmda)
            L="$2"
            shift
            ;;
            -thres|--thres)
            T="$2"
            shift
            ;;
            -iou|--iou)
            IOU="$2"
            shift
            ;;
            -temp|--temp)
            TEMP="$2"
            shift
            ;;
            -loss|--loss)
            LOSS="$2"
            shift
            ;;
            -g|--gpu_num)
            G="$2"
            shift
            ;;
            *) # unknown option
            ;;
        esac
    shift # past argument or value
    done
echo ${CONFIG_PATH}
echo ${TAG}

SAMPLES_DIR=$HOME/OD-WSCL

date

python -m torch.distributed.launch --master_port=$RANDOM --nproc_per_node=${G} \
    $SAMPLES_DIR/tools/train_net.py --config-file "${CONFIG_PATH}" \
    OUTPUT_DIR $HOME/OD-WSCL/output/${OUTPUT_DIR} \
    nms ${N} lmda ${L} thres ${T} iou ${IOU} temp ${TEMP} loss ${LOSS}

date
