# Code quality gaps and incomplete areas

This note summarizes parts of the current prototype that are brittle or under-engineered so we can prioritize fixes.

## Data handling
- Data loading relies on hardcoded toy paths and auto-builds graphs without validation or train/val/test splitting, which makes evaluation impossible and hides preprocessing failures (see `DatasetLoader` and `ensure_fused_graph`).
- Interaction history is reused directly for recommendation and explanation without any time-based filtering or deduplication, which can leak target items into the history.

## Recommendation logic
- `PopularityRecommender` has no fit/config stage, no handling for ties or cold-start users/items, and simply counts historical items, which limits extensibility.
- User goals are inferred from raw counts only; there is no weighting by recency or fallback when metadata is missing, so summaries can be empty.

## Retrieval
- Vector retrieval builds bag-of-words profiles without normalization or stopword handling, and it returns all items (including seen ones) sorted by similarity, even when similarity is zero.
- Graph retrieval assigns a heuristic confidence and stops after the first few paths without ranking or deduplication; missing nodes silently yield no evidence.

## LLM/explanation
- `ExplanationGenerator.generate` swallows all exceptions, silently falls back, and parses arbitrary JSON keys without validating against the expected schema, so malformed responses are not surfaced.
- The pipeline calls the generator's private `_fallback` method directly instead of a public interface, and explanations ignore user goals when the LLM is disabled.

## CLI/configuration
- `scripts/run_demo.py` trusts config values without type validation, and it only checks for the existence of one toy file, so misconfigured datasets fail late.
- Outputs are always overwritten without versioning, and the script does not expose seeds or logging levels for reproducibility.

## Testing
- There are no unit tests for retrievers, prompt formatting, or JSON schema validation; the only check is the toy pipeline script.
