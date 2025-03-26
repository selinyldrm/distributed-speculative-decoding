from transformers import GPT2Tokenizer, GPT2Model, AutoModelForCausalLM, AutoTokenizer
import torch

from fastchat.llm_judge.common import load_questions, temperature_config
from tqdm import tqdm 
import os, json

from datetime import datetime

question_file = f"./llm_judge/data/mt_bench/question.jsonl"
answer_file = f"/work1/deming/seliny2/Medusa/llm_judge/data/mt_bench/reference_answer/"
os.makedirs(answer_file, exist_ok=True)
answer_file += f"Llama-2-70b-chat-hf.jsonl"
questions = load_questions(question_file, None, None)

# from transformers import pipeline, set_seed
# generator = pipeline('text-generation', model='gpt2')
# set_seed(42)

# tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-70b-chat-hf')
# model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-70b-chat-hf',
#     torch_dtype=torch.float16,  
#     pad_token_id=tokenizer.pad_token_id,
#     use_cache= True,
#     ).to("cuda")


from openai import OpenAI
client = OpenAI()

response = client.responses.create(
  model="gpt-4o",
  input="Hey, what day is today?",
  text={
    "format": {
      "type": "text"
    }
  },
  reasoning={},
  tools=[],
  temperature=1,
  max_output_tokens=2048,
  top_p=1,
  store=True
)
print(response.output[0].content[0].text)
# for idx, question in enumerate(tqdm(questions)):
   
#     text = question["turns"][0]
#     encoded_input = tokenizer(text, return_tensors='pt').input_ids.to("cuda")
#     input_len = encoded_input.shape[1]
#     # input_len = encoded_input.shape[1]
#     output = model.generate(encoded_input, max_new_tokens=input_len+60)
#     print(output[0][input_len:])
#     output = tokenizer.decode(output[0][input_len:], skip_special_tokens=True)
#     # output = generator(text, max_length=input_len+60)[0]["generated_text"]
#     print(output)
#     choices = {"reference": output}

#     # Dump answers
#     os.makedirs(os.path.dirname(answer_file), exist_ok=True)
#     with open(os.path.expanduser(answer_file), "a") as fout:
#         ans_json = {
#             "question_id": question["question_id"],
#             "choices": choices,
#         }
#         fout.write(json.dumps(ans_json) + "\n")
#     fout.close()
# print("Model eval done...", flush=True)