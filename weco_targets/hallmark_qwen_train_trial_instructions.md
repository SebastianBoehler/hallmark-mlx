Optimize for the best guarded frontier after a short Qwen LoRA run, not raw training loss.

Primary objective:
- maximize `frontier_score`

Important constraints:
- preserve `compare32` held-out quality while improving `search64`
- preserve deterministic finalization
- keep BibTeX-first behavior strong for `bibtex_entry` inputs
- prefer settings that resolve within 1 or 2 tool calls
- do not broaden or disable the tool set
- keep the run short enough for iterative search

Safe edit surface:
- `TRIAL_OVERRIDES["model"]["max_tokens"]`
- `TRIAL_OVERRIDES["model"]["max_rollout_rounds"]`
- `TRIAL_OVERRIDES["training"]["num_layers"]`
- `TRIAL_OVERRIDES["training"]["learning_rate"]`
- `TRIAL_OVERRIDES["training"]["num_iterations"]`
- `TRIAL_OVERRIDES["training"]["max_seq_length"]`
- `TRIAL_OVERRIDES["training"]["grad_accumulation_steps"]`
- `TRIAL_OVERRIDES["tools"][tool]["rows"]`

Recommended ranges:
- `num_layers`: 10 to 14
- `learning_rate`: 7e-5 to 1.5e-4
- `num_iterations`: 80 to 160
- `max_seq_length`: 4096 to 6144
- `grad_accumulation_steps`: 4 to 8
- `max_tokens`: 384 to 768
- `max_rollout_rounds`: 3 to 5

Do not:
- change `TRIAL_NAME`
- change `POLICY_MODE`
- rewrite the file structure
- change the benchmark split path
- change the metric definition
- change the verdict label set
- change `temperature`, `finalization_mode`, or `force_bibtex_updater_first`
- change `seed`
- change any tool `enabled` flag
