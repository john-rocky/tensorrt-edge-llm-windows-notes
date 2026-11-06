# Writes the llm_inference input JSON for the output check: the chat template with thinking off, greedy (temperature 0,
# top_k 1), up to 64 new tokens, and the top-2 log-probabilities per step (top-1 = the emitted token under greedy).
import json
import sys

prompts = json.load(open(sys.argv[1], encoding="utf-8"))["prompts"]
doc = {
    "batch_size": 1,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 1,
    "max_generate_length": 64,
    "num_logprobs": 2,
    "apply_chat_template": True,
    "enable_thinking": False,
    "requests": [{"messages": [{"role": "user", "content": p["text"]}]} for p in prompts],
}
json.dump(doc, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(doc["requests"]), "requests")
