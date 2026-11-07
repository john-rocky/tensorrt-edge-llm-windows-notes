# Compares one llm_inference answer (an input made with --logprobs 2: under greedy, the top-1 of every step is the
# emitted token) with the transformers reference of ref_short_chat.py: first the prompt token ids (llm_inference's
# formatted_complete_request re-tokenized), then the generated ids step by step. Writes the result as JSON and prints
# one line; where the two part ways, it gives the first differing step and the top-2 gaps there on each side.
import argparse
import json

from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--edge", required=True)
ap.add_argument("--ref", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.model)
e = json.load(open(a.edge, encoding="utf-8"))["responses"][0]
ref = json.load(open(a.ref, encoding="utf-8"))
r = ref["rows"][0]
lp = e.get("logprobs") or []
e_ids = [step[0]["token_id"] for step in lp]
p_ids = tok(e.get("formatted_complete_request", ""), add_special_tokens=False).input_ids
div = next((i for i, (x, y) in enumerate(zip(e_ids, r["generated_ids"])) if x != y), None)
if div is None and len(e_ids) != len(r["generated_ids"]):
    div = min(len(e_ids), len(r["generated_ids"]))
res = {
    "prompt_ids_same": p_ids == r["prompt_ids"], "prompt_tokens_edge": len(p_ids), "prompt_tokens_ref": len(r["prompt_ids"]),
    "edge_generated": len(e_ids), "ref_generated": len(r["generated_ids"]), "finish_reason": e.get("finish_reason"),
    "identical": div is None, "first_divergence": div,
    "ref_top2_gap_at_div": r["top2_logit_gap"][div] if div is not None and div < len(r["top2_logit_gap"]) else None,
    "edge_top2_gap_at_div": (round(lp[div][0]["logprob"] - lp[div][1]["logprob"], 4)
                             if div is not None and div < len(lp) and len(lp[div]) > 1 else None),
    "edge_text": e.get("output_text", ""), "ref_text": r["text"], "ref_meta": ref["meta"],
}
json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"prompt ids equal: {res['prompt_ids_same']} ({res['prompt_tokens_edge']} tokens), generated: Edge-LLM "
      f"{res['edge_generated']}, transformers {res['ref_generated']}, same ids: {res['identical']}"
      + ("" if div is None else f" (first difference at step {div})"))
