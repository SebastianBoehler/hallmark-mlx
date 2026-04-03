These files are the right next artifact for training-set construction: small, reviewed seed traces.

Two categories live here:

1. `reviewed_seed_traces.jsonl`: manually reviewed supervision targets.
2. `live_bootstrap_traces.jsonl`: traces generated from real tool calls against curated citation inputs.
3. `reviewed_seed_traces_round2.jsonl`: additional reviewed traces promoted from live tool-grounded runs.
4. `live_bootstrap_traces_round2.jsonl`: second batch of realistic traces with mandatory first-turn tool calls.
5. `reviewed_seed_traces_round3.jsonl`: reviewed edge-case traces where the first tool path fails or is inconclusive and the policy must recover with another verifier or abstain.
6. `reviewed_seed_traces_round4.jsonl`: reviewed broken-DOI recovery traces where the first verifier fails with a real error and the policy retries with a different tool path.
7. `reviewed_seed_traces_round5.jsonl`: reviewed contract-repair traces that reinforce correct `bibtex_updater.check_bibtex` arguments, malformed-BibTeX recovery, and valid final decision enums.
8. `reviewed_seed_traces_round6.jsonl`: reviewed short-chain finalization traces that teach when one strong tool result is enough, when one fallback is enough, and when an early abstention is safer than continuing to search.
9. `reviewed_seed_traces_combined.jsonl`: the merged reviewed corpus currently recommended for local MLX LoRA runs.

Use them as follows:

1. Add new `VerificationInput` records to `seed_inputs.jsonl` or `live_bootstrap_inputs_round2.jsonl`.
2. Run `hallmark-mlx bootstrap-traces` to generate raw traces.
3. Manually review and promote only the good ones into the reviewed trace files.
4. Keep `live_bootstrap_traces.jsonl` and `live_bootstrap_traces_round2.jsonl` as realistic tool-grounded reference data and as stress tests for transcript length.
5. Use `reviewed_seed_traces_combined.jsonl` as the starting reviewed corpus before `build-dataset` and MLX LoRA training.

The goal is not volume yet. The goal is to teach the policy:

- when to call tools,
- how to compare candidates,
- when to prefer a published version over a preprint,
- when to repair BibTeX,
- and when to abstain.

The round-2 reviewed traces are filtered so that every formatted training example starts with a first assistant turn that contains a `<tool_call>` block.

The round-3 reviewed traces add edge cases where:

- the first tool call fails with a rate limit or weak evidence,
- the assistant switches to another verification source,
- and the policy either recovers a grounded result or abstains conservatively.

The round-4 reviewed traces focus on broken DOI and broken BibTeX recovery:

- the first DOI resolution call fails with a real `404`,
- the assistant falls back to bibliographic search,
- an alternate source then confirms the recovered DOI,
- or the policy abstains if the fallback search stays ambiguous.

The round-5 reviewed traces tighten the protocol contract itself:

- `bibtex_updater.check_bibtex` now consistently receives the full `bibtex` argument,
- malformed BibTeX is recovered through a search-based fallback path instead of inventing a new verdict token,
- and verified / corrected outcomes stay inside the allowed final verdict enum.

The round-6 reviewed traces add explicit tool-budget supervision:

- several cases should finalize after exactly one strong tool call,
- several recovery cases should finalize after exactly two calls,
- and one short-chain abstention case teaches the model to stop after the first ambiguous fallback instead of looping.

The default training configs now use `training.example_format: tool_transcript_steps` and `max_seq_length: 6144` because the reviewed raw traces are full chains and should not be aggressively truncated.
