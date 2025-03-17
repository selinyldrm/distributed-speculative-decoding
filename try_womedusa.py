
import os
# CUDAVISIBLE DEVICES
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import LlamaTokenizer, LlamaForCausalLM
import transformers
from fastchat.model import load_model, get_conversation_template
from needle_in_a_haystack.prompt import Prompter
from torch.profiler import profile, record_function, ProfilerActivity
import time
from fastchat.serve.cli import SimpleChatIO
from fastchat.model.model_adapter import get_conversation_template
from needle_in_a_haystack.prompt import Prompter

INT_MAX = torch.iinfo(torch.int64).max

model = LlamaForCausalLM.from_pretrained(
    "lmsys/vicuna-7b-v1.3",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_flash_attention_2=False,
    attn_implementation="eager"
    )

tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.3")
chatio = SimpleChatIO()
conv = get_conversation_template("lmsys/vicuna-7b-v1.3")

# while 1 :
    # inp = chatio.prompt_for_input(conv.roles[0])
prompter = Prompter(
            tokenizer
    )
context_len=3000
context = prompter.generate_context(context_len, 50)

inp = prompter.generate_prompt(context, context_len, 50)
conv.append_message(conv.roles[0], inp)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

input_ids = tokenizer.encode(prompt, return_tensors='pt')
input_ids_len = input_ids.size(1)
input_ids = input_ids.cuda() 

torch.cuda.synchronize()
start_time = time.time()  # Record the start time
outputs = model.generate(input_ids, max_new_tokens=4096, do_sample=False)
end_time = time.time()  # Record the end time
torch.cuda.synchronize()
output_ids = outputs[0][input_ids_len:]
# be consistent with the template's stop_token_ids
if conv.stop_token_ids:
    stop_token_ids_index = [
        i
        for i, id in enumerate(output_ids)
        if id in conv.stop_token_ids
    ]
    if len(stop_token_ids_index) > 0:
        output_ids = output_ids[: stop_token_ids_index[0]]

output = tokenizer.decode(
    output_ids,
    spaces_between_special_tokens=False,
)
print(output)
elapsed_time = end_time - start_time  # Calculate elapsed time
print(f"Elapsed time: {elapsed_time:.3f} seconds")
conv.update_last_message(output)
