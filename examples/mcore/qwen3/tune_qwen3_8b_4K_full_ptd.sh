#!/bin/bash
# export ASCEND_SLOG_PRINT_TO_STDOUT=1
# export ASCEND_GLOBAL_LOG_LEVEL=1

#export TORCH_DISABLE_SHARE_RDZV_TCP_STORE=1
#export TORCH_DIST_INIT_BARRIER=1
#export GLOO_SOCKET_IFNAME=v.100
#export TORCH_DISTRIBUTED_DEBUG=DETAIL 开启这个环境变量会创建gloo通信组的喔
#export ASCEND_LAUNCH_BLOCKING=1
#export CUDA_VISIBLE_DEVICES=0,2,4,7
#export ASCEND_RT_VISIBLE_DEVICES=0,2,4,7

# export FAULT_TOLERANCE=True
# export ALLOW_TP_SIZE=1,2,4,8
# export TP8_CKPT_LOAD_DIR="/home/user2/workplace/model_weight/model_mcore/Qwen3-8B-tp8-pp1"
# export TP4_CKPT_LOAD_DIR="/home/user2/workplace/model_weight/model_mcore/Qwen3-8B-tp4-pp1"




export HCCL_SOCKET_IFNAME=v.100
export TORCH_DISABLE_SHARE_RDZV_TCP_STORE=1
export HCCL_HOST_SOCKET_PORT_RANGE="auto"
export HCCL_NPU_SOCKET_PORT_RANGE="auto"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

NPUS_PER_NODE=8
MASTER_ADDR=11.11.10.23
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
CKPT_LOAD_DIR="/home/user2/workplace/model_weight/model_mcore/Qwen3-8B-tp2-pp2"
CKPT_SAVE_DIR="./ckpt/qwen3-8B"
DATA_PATH="./finetune_dataset/alpaca"
TOKENIZER_PATH="/home/user2/workplace/model_weight/model_from_hf/Qwen3-8B"


TP=2
PP=2
MBS=2
GBS=16
#global batch size = micro batch size × gradient accumulation steps × data parallel workers

DISTRIBUTED_ARGS="
    --max-restarts=0
    --rdzv-backend=c10d
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

#--use-distributed-optimizer \
GPT_ARGS="
    --use-mcore-models \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --kv-channels 128 \
    --qk-layernorm \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --use-flash-attn \
    --num-layers 36 \
    --hidden-size 4096  \
    --use-rotary-position-embeddings \
    --num-attention-heads 32 \
    --ffn-hidden-size 12288 \
    --max-position-embeddings 32768 \
    --seq-length 4096 \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 151936 \
    --rotary-base 1000000 \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --disable-bias-linear \
    --train-iters 2000 \
    --swiglu \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --normalization RMSNorm \
    --position-embedding-type rope \
    --norm-epsilon 1e-6 \
    --hidden-dropout 0 \
    --attention-dropout 0 \
    --no-gradient-accumulation-fusion \
    --attention-softmax-in-fp32 \
    --exit-on-missing-checkpoint \
    --no-masked-softmax-fusion \
    --group-query-attention \
    --untie-embeddings-and-output-weights \
    --num-query-groups 8 \
    --min-lr 1.25e-7 \
    --lr 1.25e-6 \
    --weight-decay 1e-1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --no-load-optim \
    --no-load-rng \
    --seed 42 \
    --bf16 \
    --ckpt-format torch
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 1000 \
    --eval-interval 1000 \
    --eval-iters 0 \
"

TUNE_ARGS="
    --finetune \
    --stage sft \
    --is-instruction-dataset \
    --prompt-type qwen3 \
    --no-pad-to-seq-lengths
"
#--disable-gloo-group \
torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $TUNE_ARGS \
    --distributed-backend nccl \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    --transformer-impl local \
    2>&1 | tee ./logs/tune_qwen3_8b_full.log