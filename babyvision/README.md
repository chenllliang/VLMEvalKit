

# 安装并测试BabyVision过程

1. 参考VLMEvalKit安装，pip install -e .

2. 获取 BabyVision1114.tsv, 放置在 ~/LMUData 下

3. 使用VLLM serve judge模型和VLM模型

judge 模型默认在8002端口
```bash
# JUDGE_MODEL
MODEL_PATH="/home/cl/data/cl/models/Qwen3-30B-A3B-Instruct-2507"

# deploy judge model , 注意修改 served-model-name ， 不要包含路径
export CUDA_VISIBLE_DEVICES=0,1
vllm serve ${MODEL_PATH} \
    --served-model-name Qwen3-30B-A3B-Instruct-2507 \
    --port 8001 \
    --host 0.0.0.0 \
    --dtype bfloat16 

```


VLM 模型默认在8001端口，参见config.py中修改
```bash

MODEL_PATH="/home/cl/data/cl/models/Qwen3-VL-4B-Instruct"

# deploy, DP=6 6路并行加速推理
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
vllm serve ${MODEL_PATH} \
    --served-model-name Qwen3-VL-4B-Instruct \
    --port 8001 \
    --host 0.0.0.0 \
    --dtype bfloat16 \
    --data-parallel-size 6

```

4. 启动测试

```bash

python run.py --data BabyVision1114 --model Qwen3-VL-4B-Instruct-VLLM --judge Qwen3-30B-A3B-Instruct-2507 --api-nproc 12 --verbose 

# 模型修改 参考 config.py , 所有模型不管API还是本地，都用统一的GPT4V类型即可

```

测试完后结果在 ./outputs 文件夹下 