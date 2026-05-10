# 修改 ascend-toolkit 路径
export CUDA_DEVICE_MAX_CONNECTIONS=1
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt_v2.py \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 2 \
    --target-pipeline-parallel-size 2 \
    --load-dir /home/user2/workplace/model_weight/model_from_hf/Qwen3-4B \
    --save-dir /home/user2/workplace/model_weight/model_mcore/Qwen3-4B-tp2-pp2 \
    --model-type-hf qwen3