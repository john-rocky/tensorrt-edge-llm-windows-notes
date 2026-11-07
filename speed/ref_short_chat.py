# transformers greedy reference for the short-chat output check: the model's chat template in its default mode
# (Qwen3: thinking on), do_sample=False, up to 128 new tokens, fp16 on the GPU. Writes the prompt ids, the generated ids,
# the text and the gap between the two largest logits at every step (check/ref_generate.py with the template default).
import argparse
import json
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--prompt-file", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--dtype", default="float16")
ap.add_argument("--device", default="cuda")
ap.add_argument("--max-new-tokens", type=int, default=128)
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).to(a.device).eval()
eos = model.generation_config.eos_token_id
eos = eos if isinstance(eos, list) else [eos]
text = open(a.prompt_file, encoding="utf-8").read()
prompt_text = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
enc = tok(prompt_text, add_special_tokens=False, return_tensors="pt")
ids = enc.input_ids.to(a.device)
t0 = time.time()
with torch.no_grad():
    out = model.generate(ids, attention_mask=enc.attention_mask.to(a.device), max_new_tokens=a.max_new_tokens,
                         do_sample=False, temperature=None, top_p=None, top_k=None, output_scores=True,
                         return_dict_in_generate=True)
gen = out.sequences[0, ids.shape[1]:].tolist()
gaps = []
for s in out.scores:
    top = torch.topk(s[0].float(), 2)
    gaps.append(round(float(top.values[0] - top.values[1]), 4))
meta = {"model": a.model, "dtype": a.dtype, "device": a.device, "torch": torch.__version__,
        "transformers": transformers.__version__, "eos_token_id": eos, "max_new_tokens": a.max_new_tokens}
if a.device == "cuda":
    meta["gpu"] = torch.cuda.get_device_name(0)
row = {"id": "short-chat", "prompt_text": prompt_text, "prompt_ids": ids[0].tolist(), "generated_ids": gen,
       "text": tok.decode(gen, skip_special_tokens=True), "top2_logit_gap": gaps, "seconds": round(time.time() - t0, 3)}
json.dump({"meta": meta, "rows": [row]}, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(ids[0]), "prompt tokens,", len(gen), "generated:", repr(row["text"][:120]))
