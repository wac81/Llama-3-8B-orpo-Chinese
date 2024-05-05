import torch
from peft import AutoPeftModelForCausalLM,PeftModel 
from transformers import AutoTokenizer, TextStreamer, AutoModelForCausalLM,BitsAndBytesConfig
 
peft_model_id = "wac81/Llama-3-8B-orpo-Chinese"
base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

# Load Model with PEFT adapter
model = AutoPeftModelForCausalLM.from_pretrained(
  peft_model_id,
#   torch_dtype=torch.float16,
#   torch_dtype=torch.bfloat16,
#   quantization_config= {"load_in_4bit": True},  # cpu 不支持量化，如果device cpu就不要选
  device_map="cpu"
)


tokenizer = AutoTokenizer.from_pretrained(base_model_name)

model.merge_and_unload()
model.save_pretrained("merged_adapters")


# compare base model

# model = AutoModelForCausalLM.from_pretrained(
#   base_model,
# #   torch_dtype=torch.float16,
# #   torch_dtype=torch.bfloat16,

# #   quantization_config= {"load_in_4bit": True},  # cpu 不支持量化，如果device cpu就不要选
#   device_map="cpu"
# )

# tokenizer = AutoTokenizer.from_pretrained(base_model)


while True:
    inp = input("user:")
    template_inp = [{
                    "role": "user",
                    "content": inp
                }]


    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Test on sample
    input_ids = tokenizer.apply_chat_template(template_inp,add_generation_prompt=True,return_tensors="pt").to(model.device)

    print('system:')

    outputs = model.generate(
        input_ids,
        streamer = streamer,
        max_new_tokens=512,
        eos_token_id= tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]


