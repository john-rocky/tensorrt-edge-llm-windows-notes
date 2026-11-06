# Scores the Edge-LLM llm_inference output against the transformers greedy references. A prompt agrees when its
# generated token ids (Edge-LLM: the top-1 of num_logprobs; greedy, so top-1 is the emitted token) equal the
# transformers fp16 ids over the first 64 tokens, or up to the end-of-sequence token; PASS = at least 7 of 8 agree and
# no empty answer. The prompt ids are checked first (formatted_complete_request re-tokenized with the model's
# tokenizer against apply_chat_template).
import argparse
import json

from transformers import AutoTokenizer


def strip_eos(ids, eos):
    return ids[:-1] if ids and ids[-1] in eos else ids


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge", required=True)
    ap.add_argument("--ref", required=True, help="transformers fp16 reference (the line)")
    ap.add_argument("--ref32", default="", help="transformers fp32 reference (instrument only)")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    edge = json.load(open(args.edge, encoding="utf-8"))
    ref = json.load(open(args.ref, encoding="utf-8"))
    ref32 = json.load(open(args.ref32, encoding="utf-8")) if args.ref32 else None
    prompts = json.load(open(args.prompts, encoding="utf-8"))["prompts"]
    eos = set(ref["meta"]["eos_token_id"])
    responses = sorted(edge["responses"], key=lambda r: r.get("request_idx", 0))
    assert len(responses) == len(prompts) == len(ref["rows"]), "row counts differ"

    rows = []
    for i, p in enumerate(prompts):
        e, r = responses[i], ref["rows"][i]
        assert r["id"] == p["id"]
        lp = e.get("logprobs") or []
        e_ids = [step[0]["token_id"] for step in lp]
        e_count = e.get("generated_token_count")
        prompt_ids = tok(e.get("formatted_complete_request", ""), add_special_tokens=False).input_ids
        a, b = strip_eos(e_ids, eos), strip_eos(r["generated_ids"], eos)
        div = first_divergence(a, b)
        ref_gap = r["top2_logit_gap"][div] if div is not None and div < len(r["top2_logit_gap"]) else None
        edge_gap = None
        if div is not None and div < len(lp) and len(lp[div]) > 1:
            edge_gap = round(lp[div][0]["logprob"] - lp[div][1]["logprob"], 4)
        text = e.get("output_text", "")
        row = {
            "id": p["id"],
            "prompt_ids_same": prompt_ids == r["prompt_ids"],
            "prompt_len_edge": len(prompt_ids),
            "prompt_len_ref": len(r["prompt_ids"]),
            "edge_count": e_count,
            "edge_logprob_steps": len(lp),
            "ref_count": len(r["generated_ids"]),
            "finish_reason": e.get("finish_reason"),
            "agree_fp16": div is None,
            "first_divergence": div,
            "ref_top2_gap_at_div": ref_gap,
            "edge_logprob_gap_at_div": edge_gap,
            "empty": (not text.strip()) or not e_count,
            "fact_ok": (any(x in text for x in p["expect"]) if p["expect"] else None),
            "text": text,
            "ref_text": r["text"],
        }
        if ref32:
            r32 = ref32["rows"][i]
            b32 = strip_eos(r32["generated_ids"], eos)
            row["agree_fp32"] = first_divergence(a, b32) is None
            row["fp16_vs_fp32_div"] = first_divergence(b, b32)
        rows.append(row)

    n_agree = sum(r["agree_fp16"] for r in rows)
    n_empty = sum(r["empty"] for r in rows)
    n_prompt = sum(r["prompt_ids_same"] for r in rows)
    verdict = "PASS" if (n_agree >= 7 and n_empty == 0 and n_prompt == len(rows)) else "FAIL"
    summary = {"agree_fp16": n_agree, "of": len(rows), "empty": n_empty, "prompt_ids_same": n_prompt,
               "verdict": verdict, "line": ">= 7 of 8 agree with transformers fp16, 0 empty, prompt ids equal",
               "edge_file": args.edge, "ref_meta": ref["meta"], "ref32_meta": ref32["meta"] if ref32 else None,
               "rows": rows}
    json.dump(summary, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    head = ["id", "prompt ids =", "Edge-LLM n", "fp16 n", "= fp16", "first div", "fp16 top-2 gap", "Edge top-2 gap",
            "= fp32", "fp16 vs fp32 div", "fact", "finish", "Edge-LLM text (start)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        cells = [r["id"], str(r["prompt_ids_same"]), str(r["edge_count"]), str(r["ref_count"]),
                 "yes" if r["agree_fp16"] else "no", str(r["first_divergence"]), str(r["ref_top2_gap_at_div"]),
                 str(r["edge_logprob_gap_at_div"]), str(r.get("agree_fp32", "-")), str(r.get("fp16_vs_fp32_div", "-")),
                 str(r["fact_ok"]), str(r["finish_reason"]),
                 r["text"][:60].replace("\n", " ").replace("|", "/")]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(f"agree with transformers fp16: {n_agree}/{len(rows)}, empty: {n_empty}, prompt ids equal: "
                 f"{n_prompt}/{len(rows)} -> {verdict}")
    open(args.md, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
