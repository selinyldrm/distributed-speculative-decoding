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
from medusa.model.medusa_model import MedusaModel
# from needle_in_a_haystack.prompt import Prompter

from transformers.integrations.deepspeed import HfDeepSpeedConfig
import deepspeed

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
        ds_config =  "/work1/deming/seliny2/Medusa/deepspeed_config.json"
        hfdsc = HfDeepSpeedConfig(ds_config)
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        torch.cuda.set_device(local_rank)
        deepspeed.init_distributed()
        model = MedusaModel.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            # low_cpu_mem_usage=True,
            # device_map="auto", turn off for DEEPSPEED !!!
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
            trust_remote_code=True,
        )
        ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
        ds_engine.module.eval()
        model = ds_engine.module
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

        # while True:
        
        if not conv:
            conv = new_chat()

        inp = None
        if local_rank == 0:
            inp = chatio.prompt_for_input(conv.roles[0])

        if torch.distributed.is_initialized():  # Ensure distributed is initialized
            inp = [inp]  # Wrap in a list to make it compatible with broadcast
            torch.distributed.broadcast_object_list(inp, src=0)
            inp = inp[0]  # Unwrap the list
            
        torch.cuda.synchronize()
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        try:
            chatio.prompt_for_output(conv.roles[1])
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(
                model.base_model.device
            )
            outputs = chatio.stream_output(
                model.medusa_generate(
                    input_ids,
                    temperature=args.temperature,
                    max_steps=args.max_steps,
                )
            )
            conv.update_last_message(outputs.strip())
        except KeyboardInterrupt:
            print("stopped generation.")
            # If generation didn't finish
            if conv.messages[-1][1] is None:
                conv.messages.pop()
                # Remove last user message, so there isn't a double up
                if conv.messages[-1][0] == conv.roles[0]:
                    conv.messages.pop()

                reload_conv(conv)
        torch.cuda.synchronize()

    except KeyboardInterrupt:
        print("exit...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name or path.")
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
    parser.add_argument("--max-steps", type=int, default=4096)
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