# Token usage and cost report

Requests in the final run: 250
Total model calls: 464 (explanations 250, messages 214)

Counts cover every model call this output depends on, including calls whose
results are served from the on-disk cache on a rerun, counted once per artifact.
Aggregate usage only. No API keys, credentials, or endpoint configuration appear here.

## Per model

| Provider | Model | Stage(s) | Calls | Input tokens | Output tokens | Total tokens | Est. cost (USD) |
|---|---|---|---:|---:|---:|---:|---:|
| openai | gpt-4o-mini-2024-07-18 | explanations, messages | 464 | 266,682 | 15,912 | 282,594 | 0.0495 |

## Overall

- Input tokens: 266,682
- Output tokens: 15,912
- Total tokens: 282,594
- Average tokens per request: 1,130.4
- Estimated total cost: USD 0.0495
- Estimated cost per request: USD 0.000198

## Cache

Extraction is cached per artifact, so a re-run makes no further calls for artifacts already read. Counts below are cache hits in this run.

- images: 16
