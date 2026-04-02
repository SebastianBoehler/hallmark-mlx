# Training Data

The core supervised object in this repository is a verification trace.

## Why Traces

A final label alone is a weak learning target for citation verification. The model should learn:

- what is missing from the citation,
- why a citation looks suspicious,
- which query to run next,
- which tool to call,
- how to compare retrieved candidates,
- and when to abstain.

That structure is what enables tool-using policy learning instead of bibliographic memorization.

## Expected Trace Fields

Key trace components include:

- `input`
- `parsed_fields`
- `suspected_issues`
- `proposed_query`
- `next_action`
- `tool_calls`
- `tool_results`
- `candidate_ranking`
- `final_decision`

## Example

```json
{
  "policy_version": "v0",
  "input": {
    "record_id": "claim-014",
    "input_type": "claim_for_supporting_refs",
    "raw_input": "Transformers outperform recurrent models on large-scale translation benchmarks."
  },
  "parsed_fields": {
    "claim_text": "Transformers outperform recurrent models on large-scale translation benchmarks."
  },
  "suspected_issues": [
    {
      "code": "insufficient_metadata",
      "rationale": "The input is a claim, not a citation. Evidence retrieval is required."
    }
  ],
  "proposed_query": {
    "query": "transformer neural machine translation attention is all you need",
    "purpose": "find_supporting_reference"
  },
  "next_action": "query_semantic_scholar"
}
```

## Contamination Control

The dataset utilities enforce family-level split assignment rather than row-level randomization. Families are grouped primarily by DOI and secondarily by normalized title plus author signature. This matters because slightly perturbed citations can otherwise leak across train and evaluation splits.

The intended practices are:

- group by DOI when available,
- otherwise group by normalized citation family,
- keep benchmark entries and their perturbation families out of training when evaluating on them,
- store private holdouts separately,
- and avoid mixing benchmark labels with retrieval corpora used at inference time.

## Private Holdouts

`VerificationInput.private_holdout` routes examples into a dedicated holdout split. Use this for:

- internal evaluation sets,
- manually curated edge cases,
- and unreleased citation families reserved for model selection sanity checks.

## Bootstrapping Seed Traces

The repository now includes a deterministic warm-start inference path for generating first-pass traces from real inputs. Use it when you need seed supervision before a trained policy exists:

```bash
hallmark-mlx infer \
  --config configs/base.yaml \
  --raw-input "Vaswani et al. Attention Is All You Need. NeurIPS 2017." \
  --output-path artifacts/examples/attention_trace.json
```

Recommended workflow:

- run `infer` on representative citations, BibTeX entries, paragraphs, and claims,
- or run `bootstrap-traces` on a JSONL batch of `VerificationInput` records,
- review the parsed fields, suspected issues, tool calls, and final decision,
- correct any weak or misleading traces manually,
- then append approved traces into a JSONL corpus for `build-dataset`.

This warm-start backend is useful because it generates explicit tool-call chains and abstentions, but it is not a substitute for human review. The goal is to accelerate seed-dataset creation, not to silently manufacture gold labels.
