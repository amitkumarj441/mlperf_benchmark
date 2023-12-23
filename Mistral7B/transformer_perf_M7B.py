import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda:0")

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=4,
        help="Number of times to run the experiments",
    )
    return parser

def get_model():
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        use_flash_attention_2=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    return model, tokenizer

def perf_benchmark(model, inputs, max_new_tokens):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()

    start_event.record()
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True, do_sample=False, eos_token_id=-1)
    end_event.record()
  
    torch.cuda.synchronize()
    latency_s = start_event.elapsed_time(end_event) * 1e-3
    max_memory = torch.cuda.max_memory_allocated(device)
    return latency_s, max_memory, out


def get_text():
    # It generates over 10k tokens
    # Customizable depending upon the case
    text = ["""Summarize the given text article in detail:\n""" * 1000]
    return text


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    model, tokenizer = get_model()
    text = get_text()

    inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)

    print(f"Context length: {inputs.input_ids.shape[-1]}")

    # Warmup
    _ = model.generate(**inputs, max_new_tokens=1, use_cache=True, do_sample=False, eos_token_id=-1)

    total_latency = 0
    total_max_memory = 0

    for _ in range(args.num_batches):
        latency_s, max_memory, generated_text = perf_benchmark(model, inputs, args.max_new_tokens)
        total_latency += latency_s
        total_max_memory += max_memory

    mean_latency = total_latency / args.num_batches

    print(f"Mean latency: {mean_latency}")
    print(f"{args.max_new_tokens / mean_latency} tokens / s")
    print(f"Mean maximum allocated memory: {max_memory / args.num_batches}")

    print(tokenizer.batch_decode(generated_text, skip_special_tokens=False))
