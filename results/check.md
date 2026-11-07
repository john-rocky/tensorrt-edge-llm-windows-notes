| id | prompt ids = | Edge-LLM n | fp16 n | = fp16 | first div | fp16 top-2 gap | Edge top-2 gap | = fp32 | fp16 vs fp32 div | fact | finish | Edge-LLM text (start) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fact_paris | True | 2 | 2 | yes | None | None | None | - | - | True | end-of-sequence | Paris |
| fact_spider | True | 9 | 9 | yes | None | None | None | - | - | True | end-of-sequence | A spider has **8 legs**. |
| fact_gold | True | 11 | 11 | yes | None | None | None | - | - | True | end-of-sequence | The chemical symbol for gold is **Au**. |
| fact_tokyo_ja | True | 2 | 2 | yes | None | None | None | - | - | False | end-of-sequence | 京都 |
| gpu | True | 19 | 19 | yes | None | None | None | - | - | None | end-of-sequence | A GPU is a parallel processor used to accelerate computation |
| haiku | True | 17 | 17 | yes | None | None | None | - | - | None | end-of-sequence | Golden leaves fall,   The air turns crisp,   A season of cha |
| colors | True | 43 | 43 | yes | None | None | None | - | - | None | end-of-sequence | The three primary colors of light are **red, blue, and green |
| code_factorial | True | 64 | 64 | yes | None | None | None | - | - | None | max-length | Here's a simple Python function that returns the factorial o |

agree with transformers fp16: 8/8, empty: 0, prompt ids equal: 8/8 -> PASS
