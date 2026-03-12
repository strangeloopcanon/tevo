# Campaign 001: D20 Nanochat

So what: this is the smallest collaborative TEVO pilot that still produces a meaningful result. Contributors run the same d20 nanochat base config under different assigned seeds, submit compact artifact bundles, and aggregate only comparable evidence. In v1, collaboration is artifact-based, not distributed training.

If you want the sendable version for another person, start with [COLLABORATOR_BRIEF.md](COLLABORATOR_BRIEF.md).

## Why This Shape

- The campaign is intentionally additive.
- The live TEVO runner, current CLI, and existing transfer workflows stay unchanged.
- Every lane shares one base config fingerprint so aggregation stays simple and we do not need campaign concepts inside the core DSL or orchestrator.
- Lane `focus` text is advisory only. The tracked comparison variable is the assigned seed plus the standard run artifacts.

## How To Contribute

1. Claim one lane in the campaign issue. The issue is the source of truth for lane ownership.
2. Run the shared config with that lane's assigned seed and budget.
3. Package the result into a compact bundle.
4. Open a PR with only the tracked bundle under `artifacts/campaigns/campaign-001-d20-nanochat/<lane-id>/`.

If you are packaging an older run that does not include `frontier.manifest.json`, pass
`--config <path-to-the-original-config>` to `campaign_submit.py`.

The default GitHub submission path is:

- run locally or on Modal
- package with `campaign_submit.py`
- commit only the compact bundle on a branch or fork
- open a PR back to `main`

If GitHub Discussions are enabled, use them for general questions or recruiting. Use the campaign issue for claims and releases, and use PRs for actual submissions.

Example:

Device and runner environment can vary. Comparability comes from the shared config fingerprint, budget, and submitted artifacts.

```bash
RUN="runs/campaign-001_lane-01"
python scripts/run_live.py \
  configs/exp_nanochat_gpt2grade_d20_modal_evolve_fineweb_staggered_gpt2vocab_aeff095e.yaml \
  --device cuda \
  --generations 12 \
  --steps 160 \
  --eval-batches 8 \
  --seed 11 \
  --out "$RUN/frontier.json" \
  --lineage-out "$RUN/frontier_lineage.json" \
  --state-out "$RUN/frontier.state.json" \
  --checkpoint-dir "$RUN/checkpoints"

python scripts/campaign_submit.py \
  campaigns/campaign-001-d20-nanochat/manifest.yaml \
  lane-01 \
  "$RUN"
```

## What Gets Tracked

- `manifest.json`: where the bundle came from
- `summary.json`: champion, top candidates, bridgeability, lineage stats
- `frontier_top.json`: compact frontier slice for review and aggregation
- `lineage_summary.json`: mutation and status counts
- `champion_spec.yaml`: the lane champion's TEVO spec
- `champion.train_recipe.yaml`: only when the champion is bridge-compatible

Raw `runs/`, checkpoints, and large logs stay local.

## Aggregation

```bash
python scripts/campaign_aggregate.py campaigns/campaign-001-d20-nanochat/manifest.yaml
python scripts/campaign_shortlist.py \
  artifacts/campaigns/campaign-001-d20-nanochat/_aggregate/aggregate_report.json
```

## Success Criteria

- pooled collaborative search beats a solo budget-matched control on TEVO metrics
- at least one bridgeable pooled candidate survives the downstream `autoresearch` lane
