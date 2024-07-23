from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

device = "cuda" # the device to load the model onto
# Instead of using model.chat(), we directly use model.generate()
# But you need to use tokenizer.apply_chat_template() to format your inputs as shown below
questions = ["Give me a short introduction to large language model.",
        "中国房地产为何会衰退",
        "中国35岁现象是什么",
        "如何才能家庭和睦呢",
        "作为一个中国人，对未来感到前途渺茫",
        "帮我写一个李商隐的诗"
        ]

AWQ_MODEL = "/root/autodl-tmp/Qwen2-7B-Instruct-AWQ/"
BF16_MODEL = "/root/autodl-tmp/Qwen2-7B/"

def get_model(name):
    # Now you do not need to add "trust_remote_code=True"
    model = AutoModelForCausalLM.from_pretrained(name,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(name)
    return model, tokenizer


def test_BF16():
    model = AutoModelForCausalLM.from_pretrained(
        BF16_MODEL,
        torch_dtype="auto",
        device_map="auto"
    )
    model.config.pad_token_id = model.config.eos_token_id
    tokenizer = AutoTokenizer.from_pretrained(BF16_MODEL)

    start_time = time.time()
    for prompt in questions:
        messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
        ]

        text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        # Instead of using model.chat(), we directly use model.generate()
        # But you need to use tokenizer.apply_chat_template() to format your inputs as shown below

        # Directly use generate() and tokenizer.decode() to get the output.
        # Use `max_new_tokens` to control the maximum output length.
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_length = 512,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        print('\n-----------------\n')
        print(text)
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response)
    consume = (time.time() - start_time) / len(questions)
    print(f"{torch.cuda.max_memory_allocated(device)//1e6}MB")
    print(f"{consume} s")


def time_experiment(prefix=False):
    # model, tokenizer = get_model(AWQ_MODEL)
    model, tokenizer = get_model(BF16_MODEL)
    start_time = time.time()
    if not prefix:
        model_inputs = tokenizer(questions, padding=True, return_tensors="pt").to(device)
        # Directly use generate() and tokenizer.decode() to get the output.
        # Use `max_new_tokens` to control the maximum output length.
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        for res in response:
            print(res)
    else:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([prompt], return_tensors="pt").to(device)

    consume = (time.time() - start_time) / len(questions)
    print(f"{torch.cuda.max_memory_allocated(device)//1e6}MB")
    print(f"{consume} s")

# time_experiment()
# test_BF16()
