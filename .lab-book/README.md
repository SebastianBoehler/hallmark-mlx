# Lab Book

Use `.lab-book/` for short, durable experiment notes.

## Purpose

- keep a chronological record of training, evaluation, and benchmark changes
- capture exact settings, artifacts, and outcomes
- make later reruns and paper-writing easier

## File Naming

- one note per meaningful experiment batch or benchmark refresh
- use `YYYY-MM-DD-short-slug.md`
- keep entries short and factual

## What To Record

- goal of the run
- configs or code paths changed
- commands or scripts used
- output artifacts
- headline metrics
- decision: keep, reject, or follow up

## Guardrails

- do not paste benchmark examples verbatim into notes
- treat `search64` and `compare32` as internal model-selection splits only
- use official HALLMARK splits for benchmark claims
- if a fresh official rerun changes a headline number, update the reports before reusing the old claim

## Suggested Entry Shape

```md
# Title

- Date:
- Goal:
- Configs:
- Commands:
- Artifacts:
- Metrics:
- Decision:
- Next:
```
