# Adapted from: https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/cli.py
"""
Chat with a model with command line interface.

Usage:
python3 -m medusa.inference.cli --model <model_name_or_path>
Other commands:
- Type "!!exit" or an empty line to exit.
- Type "!!reset" to start a new conversation.
- Type "!!remove" to remove the last prompt.
- Type "!!regen" to regenerate the last message.
- Type "!!save <filename>" to save the conversation history to a json file.
- Type "!!load <filename>" to load a conversation history from a json file.
"""
import argparse
import os
import re
import sys
import torch
import json
from transformers import AutoModelForCausalLM, AutoConfig, LlamaTokenizer, AutoTokenizer
import deepspeed
def main(args):
    
    try:
        from transformers.integrations.deepspeed import HfDeepSpeedConfig
        ds_config =  "/work1/deming/seliny2/Medusa/deepspeed_config.json"
        hfdsc = HfDeepSpeedConfig(ds_config)
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        torch.cuda.set_device(local_rank)
        torch.backends.cuda.enable_flash_sdp(False)
        deepspeed.init_distributed()

        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16,
        )

        model_engine = deepspeed.initialize(
            config=ds_config,
            model=model,
            model_parameters=model.parameters())[0]
        model = model_engine.module

        tokenizer = AutoTokenizer.from_pretrained(args.base_model)


        from fastchat.llm_judge.common import load_questions, temperature_config
        from tqdm import tqdm 

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        question_file = f"./llm_judge/data/mt_bench/question.jsonl"
        answer_file = f"./llm_judge/data/mt_bench/" + os.path.basename(args.base_model) + f"/{timestamp}/model_answer/"
        os.makedirs(answer_file, exist_ok=True)
        answer_file += f"{local_rank}.jsonl"
        questions = load_questions(question_file, None, None)
        # while True:

        for idx, question in enumerate(tqdm(questions)):
            if (not (idx>=69 and idx <= 76)):
                continue
            if question["category"] in temperature_config:
                temperature = temperature_config[question["category"]]
            else:
                temperature = 0.7

            choices = []
            wall_time = []
            inp = question["turns"][0]
            
            prompt = inp
            print(prompt)
            input_ids = tokenizer.encode(prompt, return_tensors='pt') 

            import ctypes, time
            hip_lib = ctypes.cdll.LoadLibrary("libamdhip64.so")
            attention_mask = torch.ones_like(input_ids).to(f"cuda:{local_rank}")

            print(f"Rank {local_rank} is starting generation...", flush=True)
            hip_lib.hipDeviceSynchronize()
            start_time = time.time()
            
            outputs = model.generate(
                    torch.as_tensor(input_ids).to(f"cuda:{local_rank}"),
                    attention_mask=attention_mask, 
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache= True,
                    max_new_tokens=60,
                )
            hip_lib.hipDeviceSynchronize()
            end_time = time.time()
            wall_time = end_time - start_time
            print(f"Rank {local_rank} generated.", flush=True)
           
            num_generated_tokens = outputs.shape[1] - input_ids.shape[1]
            outputs = outputs[0][len(input_ids[0]) :]

            output = tokenizer.decode(
                outputs,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )
            
            output = output.strip()
            print(output)
            
            choices = {"output": output, "num_generated_tokens": num_generated_tokens, "wall_time": wall_time}

            # Dump answers
            os.makedirs(os.path.dirname(answer_file), exist_ok=True)
            with open(os.path.expanduser(answer_file), "a") as fout:
                ans_json = {
                    "question_id": question["question_id"],
                    "choices": choices,
                }
                fout.write(json.dumps(ans_json) + "\n")
            fout.close()
            print("Model eval done...", flush=True)
       
    except KeyboardInterrupt:
        print("exit...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True, help="Base model name or path.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument(
        "--style",
        type=str,
        default="simple",
        choices=["simple", "rich", "programmatic"],
        help="Display style.",
    )
    parser.add_argument(
        "--multiline",
        action="store_true",
        help="Enable multiline input. Use ESC+Enter for newline.",
    )
    parser.add_argument(
        "--mouse",
        action="store_true",
        help="[Rich Style]: Enable mouse support for cursor positioning.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print useful debug information (e.g., prompts)",
    )
    args = parser.parse_args()
    main(args)