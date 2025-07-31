# 启动vllm服务

# 模型路径
MODEL_PATH=MissCover-Qwen2.5VL-7B

# 使用GPUs的编号
export CUDA_VISIBLE_DEVICES=5

# 启动服务
# --model 模型路径
# --tensor-parallel-size 2 使用2个GPU
# --enforce-eager 强制使用eager模式
# --served-model-name 服务中的模型名称
# --max-model-len=8092 最大上下文长度
# --port 9999 端口号
# --host 0.0.0.0 主机地址
python -m vllm.entrypoints.openai.api_server --model $MODEL_PATH --tensor-parallel-size 2 --enforce-eager --served-model-name MissCover-Qwen2.5VL-7B --max-model-len=8092 --port 9999 --host 0.0.0.0 
