Now I have full context. Here's my re-review:

---

## Re-review: PR #4 — Preserve provider reasoning on tool streams

### Previous concerns status

The previous review file was empty, so I'm reviewing from scratch against the updated diff.

### Verdict: **No blocking issues. Ship it.**

### Correctness analysis

1. **`to_llm_response` — `raw=response` (non-streaming path):** Correctly preserves the full SDK response object so Kestrel core can extract `reasoning_content` from `raw.choices[0].message`. The new docstring accurately describes the contract. Good.

2. **`stream_with_tool_calls` — reasoning accumulation (streaming path):** The `reasoning_content` accumulator follows the same `getattr` + `isinstance` guard pattern used for `content`. It only surfaces in the terminal `LLMResponse.raw` dict when tool calls are present, which matches the comment: text-only streams deliver chunks directly and don't need the replay envelope. This is correct — the orchestrator only replays assistant messages when sending them back alongside tool results.

3. **`raw` shape asymmetry (streaming vs non-streaming):** Non-streaming sets `raw` to the full SDK response object; streaming sets it to `{"reasoning_content": "..."}` (a plain dict) or `None`. This is intentional — the streaming path has no single response object to preserve. The downstream consumer (Kestrel core) needs to handle both shapes. As long as core checks for `reasoning_content` via the dict key on streaming and via `raw.choices[0].message.reasoning_content` on non-streaming, this works. The PR description and comments indicate this is the established contract.

4. **Version bumps:** openai-compat 0.1.4 → 0.1.6, leaf providers 0.1.5 → 0.1.7, meta package 0.1.3 → 0.1.5. All dependency lower bounds updated consistently. The version skip (0.1.5 → 0.1.7, not 0.1.6) is fine — likely 0.1.6 was the prior commit's bump. Lock file matches.

5. **Tests are solid:**
   - `test_to_llm_response_preserves_raw_provider_reasoning_object` — verifies `raw is response` (identity, not equality), confirming the full object is passed through.
   - Streaming test adds reasoning chunks before content/tool deltas and asserts `final.raw == {"reasoning_content": "Need health. Use lookup."}` — covers the accumulation and conditional inclusion.
   - Contract tests updated for new version strings.

### Minor observations (non-blocking)

- **No test for text-only streaming (no tool calls):** When the stream has reasoning but no tool calls, `reasoning_content` is accumulated but silently discarded since no terminal `LLMResponse` is yielded. This is by design per the comment, but a test asserting this behavior would strengthen the contract. Not blocking.

- **No test for streaming without reasoning:** The happy path where `delta` has no `reasoning_content` attribute at all relies on `getattr` returning `None`. Already implicitly covered by the `isinstance` guard, so fine.
