# Writes the llm_inference input for the short-chat task of edge-llm-bench
# (https://github.com/john-rocky/edge-llm-bench/blob/main/prompts/short-chat.md): one request with the prompt file's
# text, the model's chat template in its default mode (Qwen3: thinking on), greedy (temperature 0, top_k 1), at most
# 128 new tokens, batch 1. --logprobs 2 adds the top-2 log-probabilities of every step (used for the output check).
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("prompt_file")
ap.add_argument("out")
ap.add_argument("--logprobs", type=int, default=0)
a = ap.parse_args()
text = open(a.prompt_file, encoding="utf-8").read()
doc = {
    "batch_size": 1,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 1,
    "max_generate_length": 128,
    "apply_chat_template": True,
    "enable_thinking": True,
    "requests": [{"messages": [{"role": "user", "content": text}]}],
}
if a.logprobs:
    doc["num_logprobs"] = a.logprobs
json.dump(doc, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("prompt", repr(text), "logprobs", a.logprobs)
