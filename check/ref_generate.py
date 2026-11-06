# Greedy reference for the output check: transformers with the model's chat template (enable_thinking=False),
# do_sample=False, up to 64 new tokens. Writes the prompt ids, the generated ids, the text, and the gap between the
# two largest logits at every step (a small gap marks a near-tie where two fp16 implementations may part ways).
import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype)).to(args.device).eval()
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    prompts = json.load(open(args.prompts, encoding="utf-8"))["prompts"]
    rows = []
    for p in prompts:
        messages = [{"role": "user", "content": p["text"]}]
        prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                              enable_thinking=False)
        enc = tok(prompt_text, add_special_tokens=False, return_tensors="pt")
        ids = enc.input_ids.to(args.device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(ids, attention_mask=enc.attention_mask.to(args.device),
                                 max_new_tokens=args.max_new_tokens, do_sample=False, temperature=None, top_p=None,
                                 top_k=None, output_scores=True, return_dict_in_generate=True)
        gen = out.sequences[0, ids.shape[1]:].tolist()
        gaps = []
        for s in out.scores:
            top = torch.topk(s[0].float(), 2)
            gaps.append(round(float(top.values[0] - top.values[1]), 4))
        rows.append({
            "id": p["id"],
            "prompt_text": prompt_text,
            "prompt_ids": ids[0].tolist(),
            "generated_ids": gen,
            "text": tok.decode(gen, skip_special_tokens=True),
            "top2_logit_gap": gaps,
            "seconds": round(time.time() - t0, 3),
        })
        print(p["id"], len(gen), repr(rows[-1]["text"][:80]), flush=True)
    meta = {"model": args.model, "dtype": args.dtype, "device": args.device, "torch": torch.__version__,
            "eos_token_id": eos, "max_new_tokens": args.max_new_tokens}
    if args.device == "cuda":
        meta["gpu"] = torch.cuda.get_device_name(0)
    import transformers
    meta["transformers"] = transformers.__version__
    json.dump({"meta": meta, "rows": rows}, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
