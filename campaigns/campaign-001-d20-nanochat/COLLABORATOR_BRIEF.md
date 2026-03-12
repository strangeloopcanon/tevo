## Collaborator Brief

So what: this is the sendable version of the first TEVO campaign. It explains the goal, the exact ask, and what a useful contribution looks like without requiring anyone to understand the whole repo.

## The Pitch

We are running a small collaborative TEVO campaign around the same d20 nanochat-aligned base config.

The question is simple:

- does pooling several seed-diverse TEVO searches find better motifs than one solo budget-matched search?
- and does at least one pooled motif survive downstream validation in upstream `autoresearch`?

This is intentionally not distributed training. Each collaborator owns one complete lane, runs it normally, and submits one compact artifact bundle for aggregation.

## How Submission Works

Runs do not happen in GitHub branches. The run happens locally or on Modal, and GitHub is only used to submit the compact result bundle.

The default flow is:

1. Fork the repo if you do not already have write access.
2. Claim a lane in the campaign issue.
3. Run the lane locally or on Modal.
4. Package the result into a compact bundle.
5. Commit only that bundle on a branch.
6. Open a PR back to `main`.

The recommended default for outside collaborators is fork plus PR. If you already have write access, a branch in the main repo is also fine, but fork plus PR keeps the contribution path consistent.

## What I Am Asking For

If you join, I am asking you to do one bounded piece of work:

1. Claim one lane.
2. Run the fixed d20 campaign config with that lane's assigned seed and budget.
3. Package the result with `campaign_submit.py`.
4. Open one PR with the compact bundle only.

You do not need to:

- edit TEVO core code
- coordinate live with other runners
- upload raw checkpoints or large `runs/` directories
- do distributed training

## Lane Menu

Every lane uses the same base config fingerprint and budget. The only hard comparison variable is the assigned seed.

| Lane | Seed | Focus |
|------|------|-------|
| `lane-01` | `11` | Plain baseline basin |
| `lane-02` | `17` | Watch for sparse or window-pattern improvements |
| `lane-03` | `23` | Watch for optimizer or schedule improvements |
| `lane-04` | `29` | Watch for KV-layout or attention-geometry improvements |
| `lane-05` | `31` | Watch for dense-block depth or FFN-shape trade-offs |
| `lane-06` | `37` | Watch for surprising motifs worth replication |

## Exact Contribution Shape

Each collaborator contributes one PR containing only:

- `manifest.json`
- `summary.json`
- `frontier_top.json`
- `lineage_summary.json`
- `champion_spec.yaml`
- optional `champion.train_recipe.yaml`

That gives us something small enough to review on GitHub and strong enough to aggregate.

## Coordination

Use the campaign issue as the source of truth for lane ownership.

- issue comments are where people claim or release lanes
- PRs are where people submit compact bundles
- GitHub Discussions, if enabled, are optional and best used for recruiting, questions, and result chatter

That split keeps coordination simple:

- issues track who is doing what
- PRs track what was submitted
- discussions stay non-blocking

## Ready-To-Send Note

Use this if you want a short direct outreach message:

> I’m running a small collaborative TEVO experiment and I think you’d be a great fit for one lane. The idea is simple: we each run the same d20 nanochat-aligned TEVO campaign under different assigned seeds, then submit a compact artifact bundle instead of raw checkpoints. We’re testing whether pooled seed-diverse search finds a better motif than a solo budget-matched run, and whether at least one winning motif transfers into upstream `autoresearch`. If you’re up for it, I’d ask you to own one complete lane: claim a seed, run the fixed config and budget, package the result, and open one small PR. No core code changes, no distributed training, and no giant uploads. If that sounds interesting I can send you the exact lane list and commands.

## If They Say Yes

Send them this checklist:

1. Pick one unclaimed lane from the table above.
2. Comment on the campaign issue with:

```text
Claiming: lane-0X
Hardware: <local CUDA / Modal / other>
ETA: <date or rough time>
```

3. Run the campaign:

```bash
RUN="runs/campaign-001_lane-0X"
SEED="<assigned-seed>"

python scripts/run_live.py \
  configs/exp_nanochat_gpt2grade_d20_modal_evolve_fineweb_staggered_gpt2vocab_aeff095e.yaml \
  --device cuda \
  --generations 12 \
  --steps 160 \
  --eval-batches 8 \
  --seed "$SEED" \
  --out "$RUN/frontier.json" \
  --lineage-out "$RUN/frontier_lineage.json" \
  --state-out "$RUN/frontier.state.json" \
  --checkpoint-dir "$RUN/checkpoints"
```

4. Package the result:

```bash
python scripts/campaign_submit.py \
  campaigns/campaign-001-d20-nanochat/manifest.yaml \
  lane-0X \
  "$RUN"
```

If they are packaging an older run without `frontier.manifest.json`, they should add:

```bash
  --config configs/exp_nanochat_gpt2grade_d20_modal_evolve_fineweb_staggered_gpt2vocab_aeff095e.yaml
```

5. Open one PR containing only:

```text
artifacts/campaigns/campaign-001-d20-nanochat/lane-0X/
```

The cleanest Git flow is:

```bash
git checkout -b campaign-001-lane-0X
git add artifacts/campaigns/campaign-001-d20-nanochat/lane-0X
git commit -m "campaign: submit lane-0X"
git push origin campaign-001-lane-0X
```

If they are contributing from a fork, that PR should target `strangeloopcanon/tevo:main`.

## Recommended Framing

The best framing is:

- one fixed campaign
- one owned lane per collaborator
- one small PR per lane
- one aggregated report at the end

That is much easier to say yes to than “want to collaborate on a repo?”
