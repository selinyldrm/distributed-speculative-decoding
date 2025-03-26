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
from fastchat.serve.cli import SimpleChatIO, RichChatIO, ProgrammaticChatIO
from fastchat.model.model_adapter import get_conversation_template
from fastchat.conversation import get_conv_template
import json
from medusa.model.medusa_model_legacy import MedusaModel
# from needle_in_a_haystack.prompt import Prompter



def main(args):
    if args.style == "simple":
        chatio = SimpleChatIO(args.multiline)
    elif args.style == "rich":
        chatio = RichChatIO(args.multiline, args.mouse)
    elif args.style == "programmatic":
        chatio = ProgrammaticChatIO()
    else:
        raise ValueError(f"Invalid style for console: {args.style}")
    try:
        from transformers.integrations.deepspeed import HfDeepSpeedConfig
        import deepspeed
        from transformers import AutoModelForCausalLM, AutoConfig
        ds_config =  "/work1/deming/seliny2/Medusa/deepspeed_config.json"
        hfdsc = HfDeepSpeedConfig(ds_config)
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        torch.cuda.set_device(local_rank)
        deepspeed.init_distributed()
        torch.backends.cuda.enable_flash_sdp(False)

        model = MedusaModel.from_pretrained(
            args.model,
            args.base_model,
            medusa_num_heads=5,
            torch_dtype=torch.float16,
        )

        model_engine = deepspeed.initialize(
            config=ds_config,
            model=model,
            model_parameters=model.parameters())[0]
        model = model_engine.module

        tokenizer = model.get_tokenizer()
        conv = None

        def new_chat():
            return get_conversation_template(args.model)

        def reload_conv(conv):
            """
            Reprints the conversation from the start.
            """
            for message in conv.messages[conv.offset :]:
                chatio.prompt_for_output(message[0])
                chatio.print_output(message[1])

        from fastchat.llm_judge.common import load_questions, temperature_config
        from fastchat.model import load_model, get_conversation_template
        from tqdm import tqdm 

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        question_file = f"./llm_judge/data/mt_bench/question.jsonl"
        answer_file = f"./llm_judge/data/mt_bench/" + os.path.basename(args.model) + f"/{timestamp}/model_answer/"
        os.makedirs(answer_file, exist_ok=True)
        answer_file += f"{local_rank}.jsonl"
        questions = load_questions(question_file, None, None)
        # while True:

        alpaca_template = """
    {% if messages[0]['role'] == 'system' %}
        {% set loop_messages = messages[1:] %}
        {% set system_message = messages[0]['content'] %}
    {% else %}
        {% set loop_messages = messages %}
        {% set system_message = false %}
    {% endif %}

    {% for message in loop_messages %}
        {% if message['role'] == 'user' %}
            {{ '### Instruction:\n' + message['content'] + '\n' }}
        {% elif message['role'] == 'assistant' %}
            {{ '### Response:\n' + message['content'] + '\n' }}
        {% endif %}
    {% endfor %}
    """
        
        for idx, question in enumerate(tqdm(questions)):
            model.base_model.model.medusa_mask = {}
            model.base_model.model.medusa_mask[local_rank] = None
            model.medusa_buffers = {}
            model.medusa_buffers[local_rank] = None
            model.medusa_choices = {}
            model.medusa_choices[local_rank] = None
            model.past_key_values = {}
            model.past_key_values[local_rank] = None
            model.past_key_values_data = {}
            model.past_key_values_data[local_rank] = None
            model.current_length_data = {}
            model.current_length_data[local_rank] = None
            if question["category"] in temperature_config:
                temperature = temperature_config[question["category"]]
            else:
                temperature = 0.7

            choices = []
            new_tokens = []
            wall_time = []
            inp = question["turns"][0]
            
            prompt = f"### Instruction:\n{inp}\n" 
            print(prompt)
            input_ids = tokenizer([prompt]).input_ids
            print(f"Rank {local_rank} is starting generation...", flush=True)
            outputs, tokens, wall_time = model.medusa_generate(
                    torch.as_tensor(input_ids).to(f"cuda:{local_rank}"),
                    temperature=temperature,
                    max_steps=args.max_steps
                )
            print(f"Rank {local_rank} generated.", flush=True)
            outputs = outputs[0][len(input_ids[0]) :]
            output = tokenizer.decode(
                outputs,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )
            output = output.strip()
            print(output)
            new_tokens.append(int(tokens))
            choices = {"output": output, "new_tokens": new_tokens, "wall_time": wall_time}
           

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
    parser.add_argument("--model", type=str, required=True, help="Model name or path.")
    parser.add_argument("--base_model", type=str, required=True, help="Base model name or path.")
    parser.add_argument(
        "--load-in-8bit", action="store_true", help="Use 8-bit quantization"
    )
    parser.add_argument(
        "--load-in-4bit", action="store_true", help="Use 4-bit quantization"
    )
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument(
        "--conv-system-msg", type=str, default=None, help="Conversation system message."
    )
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