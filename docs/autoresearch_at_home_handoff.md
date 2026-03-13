# TEVO -> autoresearch@home Handoff

So what: TEVO can now contribute one bridge-compatible candidate into `autoresearch@home` without pretending the whole TEVO evolutionary loop is swarm-native. TEVO proposes the motif, and `autoresearch@home` becomes the collaborative `train.py` validation lane.

## What This Is

- **TEVO** is a search engine for motifs.
- **`autoresearch@home`** is a collaborative `train.py` experiment loop.
- The integration point is the shared CUDA `train.py` surface.

This is intentionally a contribution lane, not distributed TEVO evolution.

## Recommended Workflow

1. Run TEVO locally and inspect the frontier.
2. Choose one bridge-compatible candidate.
3. Export an `autoresearch@home` handoff workspace.
4. Run that candidate in an `autoresearch@home` fork or branch.
5. Publish the result, insight, and next hypothesis through their coordinator flow.

## Export One Candidate

Use the new handoff command when you already know which frontier candidate you want to try downstream:

```bash
evo-loop autoresearch-at-home-handoff \
  --frontier runs/<run>/frontier.json \
  --candidate-id <candidate_id> \
  --run-root runs/at_home_handoff
```

By default this stages against `https://github.com/mutable-state-inc/autoresearch-at-home.git`.

If you already have a local checkout:

```bash
evo-loop autoresearch-at-home-handoff \
  --frontier runs/<run>/frontier.json \
  --candidate-id <candidate_id> \
  --run-root runs/at_home_handoff \
  --autoresearch-repo /path/to/autoresearch-at-home
```

## What The Command Writes

The handoff bundle contains:

- `candidate.train_recipe.yaml`
- `baseline.train.py`
- `candidate.train.py`
- `candidate.diff`
- `repo/`
- `handoff_manifest.json`
- `handoff_summary.md`

The staged `repo/` directory is the practical center of the handoff: it is a runnable `autoresearch@home` workspace with the TEVO candidate already installed as `train.py`.

## What To Do Next

From the staged repo:

```bash
cd runs/at_home_handoff/repo
git checkout -b autoresearch/tevo-<candidate_id>
uv run train.py
```

If you are contributing inside the `autoresearch@home` swarm, claim the experiment through their coordinator before the run, then publish the result, insight, and follow-up hypothesis afterward.

## How To Think About This

- TEVO is doing the broader motif search.
- `TrainRecipe` carries only the renderer-safe subset of that candidate.
- `autoresearch@home` is where one chosen candidate can be discussed, run, and iterated collaboratively.

That means this workflow helps TEVO contribute cleanly to their swarm today, while leaving native TEVO swarm search as a separate future project.
