# trajectory

Qwen2-7B
    BF16:15G
    Instruct-AWQ:5.2G

1. AutoModelForCausalLM.from_pretrained 直接加载,显存在上面基础上+2G。
Instruct-AWQ:5.2G 推理： 42s/question
BF16:15G: 18s  没控制好，结尾总是生成乱七八糟的东西

2. 但是用vllm加载，有多少内存都不够用 OOM。3090、V100（且不支持BF16）  需要设置max_len_seq
# python -m vllm.entrypoints.openai.api_server --model Qwen2-7B --max-model-len 8096



全参数微调
1. deepspeed stage 2 OOM
2. deepspeed stage 3
    "deepspeed.ops.op_builder.builder.CUDAMismatchException: >- DeepSpeed Op Builder: Installed CUDA version 11.8 does not match the version torch was compiled with 12.1, unable to compile cuda/cpp extensions without a matching cuda version."
机器的确是cuda11.8 + pytorch2.11 " 可能是pytorch被升级了导致的，无法使用stage3

lora微调的确是可以的，很节约显存
L20 48G显卡，模型约17G



# Qwen2 生成NER任务，0微调的情况
1. 写代码，提交

# UniversalNER 下载数据
1. 数据可能需要晚上下载会更快（使用的人更少）
