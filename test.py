from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, LlamaForCausalLM, TextStreamer
import torch, pdb, os
from accelerate import init_empty_weights
from accelerate import load_checkpoint_and_dispatch
import datasets
import sys
import argparse
from tqdm import tqdm
import json
import time

def question_iterator(filename):
    # read each json line as question
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            question = data.get('question')
            if question:
                yield question

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="meta/Llama-2-7b-chat-hf")
    parser.add_argument("--llm", type=bool, default=False)
    args = parser.parse_args()

    model_name_or_path = args.model
    llm = args.llm

    MAX_LENGTH = 1024*128
    MAX_GEN_LENGTH = 1024 * 4
    config = AutoConfig.from_pretrained(model_name_or_path)

    if llm:
        print("acc load")
        with init_empty_weights():
            # model = LlamaForCausalLM._from_config(config, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
            model = LlamaForCausalLM._from_config(config, torch_dtype=torch.bfloat16)

        model.tie_weights()
        try:
            model = load_checkpoint_and_dispatch(model, weights_location, dtype=torch.float16, device_map="auto")
        except:
            model = load_checkpoint_and_dispatch(model, model_name_or_path, dtype=torch.float16, device_map="auto")
    else:
        print("normal load")
        model = LlamaForCausalLM.from_pretrained(model_name_or_path, config=config, torch_dtype=torch.bfloat16, device_map="auto")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    version = [[16, 32], [32, 32], [64, 32], [32, 16]]
    filenames = [f"Counting_Stars_{m}_{n}.jsonl" for m, n in version]
    for filename in filenames:
        for input_text in question_iterator(filename):
            input_tokens = tokenizer(input_text,
                return_tensors="pt",
                return_attention_mask=False,
                truncation=True,
                max_length=MAX_LENGTH,
                padding=False)

            streamer = TextStreamer(tokenizer)
            generation_output = model.generate(
                input_tokens['input_ids'].cuda(), 
                max_new_tokens=MAX_GEN_LENGTH,
                streamer=streamer,
                use_cache=True,
                return_dict_in_generate=True)


            output = tokenizer.decode(generation_output.sequences[0])
            print(">>>")
            print(output)
            print("<<<")



if __name__ == "__main__":
    main()

