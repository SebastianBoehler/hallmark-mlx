Optimize for the best verification frontier, not just raw quality at any cost.

Primary objective:
- maximize `frontier_score`

Important constraints:
- favor solutions that succeed within 1 or 2 tool calls
- preserve deterministic, evidence-grounded verification behavior
- do not weaken hallucination detection to gain easier `VALID` outputs
- keep BibTeX-first behavior strong for `bibtex_entry` inputs
- avoid enabling extra tools unless they improve the frontier metrics

Safe edit surface:
- `POLICY_MODE`
- values inside `TRIAL_OVERRIDES["model"]`
- values inside `TRIAL_OVERRIDES["tools"]`

Do not:
- rewrite the file structure
- change the benchmark slice path
- change the metric definition
- change the verdict label set
