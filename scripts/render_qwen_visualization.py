#!/usr/bin/env python3
"""Render native Qwen training and inference examples as HTML visualizations."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

MAX_TOKEN_ROWS = 220
def _load_first_example(path: Path, target_type: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("metadata", {}).get("target_type") == target_type:
                return row
    raise ValueError(f"No sample with target_type={target_type!r} in {path}")
def _load_traces(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows
def _render_chat(tokenizer: Any, sample: dict[str, Any]) -> str:
    return tokenizer.apply_chat_template(
        sample["messages"],
        tools=sample.get("tools"),
        tokenize=False,
    )
def _token_rows(tokenizer: Any, text: str, limit: int = MAX_TOKEN_ROWS) -> list[dict[str, Any]]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    rows: list[dict[str, Any]] = []
    for idx, token_id in enumerate(token_ids[:limit]):
        piece = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        rows.append(
            {
                "idx": idx,
                "token_id": token_id,
                "piece": piece,
                "kind": _token_kind(piece),
            }
        )
    return rows
def _token_kind(piece: str) -> str:
    if "<|im_start|>" in piece or "<|im_end|>" in piece or "<|endoftext|>" in piece:
        return "special"
    if "<tool_call>" in piece or "</tool_call>" in piece or "<tool_response>" in piece:
        return "protocol"
    if piece.strip().startswith("{") or piece.strip().startswith("}") or piece.strip().startswith('"'):
        return "json"
    return "plain"
def _token_table(rows: list[dict[str, Any]]) -> str:
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{row['idx']}</td>"
            f"<td>{row['token_id']}</td>"
            f"<td><span class='chip {row['kind']}'>{html.escape(repr(row['piece']))}</span></td>"
            "</tr>"
        )
    return "\n".join(body)
def _pretty_json(payload: Any) -> str:
    return html.escape(json.dumps(payload, indent=2, ensure_ascii=False))
def _trace_section(trace: dict[str, Any], index: int) -> str:
    assistant_turns = trace.get("metadata", {}).get("assistant_turns", [])
    turns_html = "\n".join(
        f"<details><summary>Assistant turn {i + 1}</summary><pre>{html.escape(turn)}</pre></details>"
        for i, turn in enumerate(assistant_turns)
    )
    return f"""
    <section class="panel">
      <h2>Finetuned Trace {index + 1}: {html.escape(trace['input']['record_id'])}</h2>
      <div class="grid two">
        <div>
          <h3>Trace Summary</h3>
          <pre>{_pretty_json({
              "input": trace["input"],
              "tool_calls": trace.get("tool_calls", []),
              "tool_results": trace.get("tool_results", []),
              "final_decision": trace.get("final_decision"),
              "metadata": {
                  "first_response_tool_call_count": trace.get("metadata", {}).get("first_response_tool_call_count"),
                  "parse_errors": trace.get("metadata", {}).get("parse_errors", []),
              },
          })}</pre>
        </div>
        <div>
          <h3>Assistant Turns</h3>
          {turns_html}
        </div>
      </div>
    </section>
    """
def build_html(
    *,
    model_name: str,
    tool_call_sample: dict[str, Any],
    final_sample: dict[str, Any],
    tool_call_rendered: str,
    final_rendered: str,
    tool_call_tokens: list[dict[str, Any]],
    final_tokens: list[dict[str, Any]],
    traces: list[dict[str, Any]],
) -> str:
    trace_html = "\n".join(_trace_section(trace, idx) for idx, trace in enumerate(traces))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>hallmark-mlx Qwen Visualization</title>
  <style>
    body {{
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f7f6f2;
      color: #1d1b17;
      margin: 0;
      padding: 32px;
      line-height: 1.4;
    }}
    h1, h2, h3 {{ margin-top: 0; }}
    .panel {{
      background: #fffdf8;
      border: 1px solid #dfd7ca;
      border-radius: 16px;
      padding: 20px;
      margin-bottom: 24px;
      box-shadow: 0 8px 20px rgba(40, 30, 10, 0.06);
    }}
    .grid {{
      display: grid;
      gap: 20px;
    }}
    .two {{
      grid-template-columns: 1fr 1fr;
    }}
    pre {{
      background: #fbfaf6;
      border: 1px solid #e4ddcf;
      border-radius: 12px;
      padding: 14px;
      overflow-x: auto;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 12px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    th, td {{
      border-bottom: 1px solid #ece5d8;
      padding: 6px 8px;
      text-align: left;
      vertical-align: top;
    }}
    .chip {{
      display: inline-block;
      padding: 4px 6px;
      border-radius: 8px;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }}
    .special {{ background: #ffe7c2; }}
    .protocol {{ background: #d8f0ff; }}
    .json {{ background: #e9f5d1; }}
    .plain {{ background: #f1ede6; }}
    .note {{
      background: #fff6de;
      border: 1px solid #f0d99b;
      border-radius: 12px;
      padding: 12px 14px;
      margin-bottom: 20px;
    }}
    details summary {{
      cursor: pointer;
      font-weight: 600;
      margin-bottom: 8px;
    }}
    @media (max-width: 1000px) {{
      .two {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <section class="panel">
    <h1>hallmark-mlx Sample / Token Visualization</h1>
    <div class="note">
      This uses the actual <strong>{html.escape(model_name)}</strong> tokenizer and chat template.
      It is more accurate for this fine-tune than the OpenAI tokenizer tool, because the model-specific
      tokens here are Qwen-native tokens like <code>&lt;|im_start|&gt;</code> and <code>&lt;|im_end|&gt;</code>.
    </div>
    <p>Artifacts shown below are taken from the round-7 native-tool training set and the finetuned model traces from held-out evaluation.</p>
  </section>

  <section class="panel">
    <h2>Training Sample: First Tool Call Step</h2>
    <div class="grid two">
      <div>
        <h3>Structured Sample JSON</h3>
        <pre>{_pretty_json(tool_call_sample)}</pre>
      </div>
      <div>
        <h3>Rendered Chat Template Text</h3>
        <pre>{html.escape(tool_call_rendered)}</pre>
      </div>
    </div>
    <h3>Token Breakdown (first {len(tool_call_tokens)} tokens)</h3>
    <table>
      <thead><tr><th>#</th><th>Token ID</th><th>Decoded Piece</th></tr></thead>
      <tbody>{_token_table(tool_call_tokens)}</tbody>
    </table>
  </section>

  <section class="panel">
    <h2>Training Sample: Final Decision Step</h2>
    <div class="grid two">
      <div>
        <h3>Structured Sample JSON</h3>
        <pre>{_pretty_json(final_sample)}</pre>
      </div>
      <div>
        <h3>Rendered Chat Template Text</h3>
        <pre>{html.escape(final_rendered)}</pre>
      </div>
    </div>
    <h3>Token Breakdown (first {len(final_tokens)} tokens)</h3>
    <table>
      <thead><tr><th>#</th><th>Token ID</th><th>Decoded Piece</th></tr></thead>
      <tbody>{_token_table(final_tokens)}</tbody>
    </table>
  </section>

  {trace_html}
</body>
</html>
"""
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--train-jsonl", default="artifacts/train_qwen_1_5b_round7/prepared_dataset/train.jsonl")
    parser.add_argument("--trace-jsonl", default="artifacts/eval_qwen15_round7_test_traces.jsonl")
    parser.add_argument("--output-html", default="artifacts/visualizations/qwen_round7_native_visualization.html")
    parser.add_argument("--trace-limit", type=int, default=2)
    args = parser.parse_args()

    train_path = Path(args.train_jsonl)
    trace_path = Path(args.trace_jsonl)
    output_path = Path(args.output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tool_call_sample = _load_first_example(train_path, "tool_call")
    final_sample = _load_first_example(train_path, "final_decision")
    traces = _load_traces(trace_path, args.trace_limit)

    tool_call_rendered = _render_chat(tokenizer, tool_call_sample)
    final_rendered = _render_chat(tokenizer, final_sample)
    tool_call_tokens = _token_rows(tokenizer, tool_call_rendered)
    final_tokens = _token_rows(tokenizer, final_rendered)

    html_doc = build_html(
        model_name=args.model,
        tool_call_sample=tool_call_sample,
        final_sample=final_sample,
        tool_call_rendered=tool_call_rendered,
        final_rendered=final_rendered,
        tool_call_tokens=tool_call_tokens,
        final_tokens=final_tokens,
        traces=traces,
    )
    output_path.write_text(html_doc, encoding="utf-8")
    print(output_path)
if __name__ == "__main__":
    main()
