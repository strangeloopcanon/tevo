# Evolutionary Architecture Takeaways

This document summarizes what we have learned so far from multiple evolutionary sweeps over the small surrogate model. It is intentionally architecture‑agnostic and focuses on *how to set up evolution* so that specific families of architectures can emerge, rather than hard‑coding them.

## 1. Evolution is only as good as its search space

The DSL defines what can exist; the evolution config defines what is *likely* to appear. In practice:

- If the DSL does not expose a concept, evolution can never discover it.
- If the mutation set never touches a knob, that knob is effectively constant.
- If the evaluation budget is too small, complex candidates never have a chance to prove they are better.

**Implication:** Before running any sweep, make sure:

- The DSL is expressive enough for the structures you care about (MoE, selectors, memory, recurrence, sparsity, etc.).
- The mutation registry includes operators that touch those structures.
- The evaluation loop surfaces metrics that reward those structures.

## 2. Hard gates are the floor; schedules are the ramp

Without structural gates we consistently observed collapse to trivial architectures:

- Shallow stacks (3–4 layers).
- No or few MoE blocks.
- A single selector or memory module at most.

These models are cheap to evaluate and look “good enough” under tight budgets, so they dominate selection.

We fixed this by enforcing hard structural gates in live mode, e.g.:

- `min_layers` – minimum number of blocks.
- `min_moe_blocks` – minimum number of MoE FFNs.
- `min_selector_blocks` – minimum number of attention blocks with selectors enabled.

Candidates that violate these thresholds are rejected before training.

**Pattern:** To target a structural regime (e.g., “deep + sparse + multi‑expert”), first define *floors*:

- Reject candidates that are too shallow.
- Reject candidates with too few sparse/mixture components.
- Reject candidates that lack the memory/recurrence you want to explore.

For progressive runs, use `gate_schedule` to increase those floors by generation (for example, `min_layers: 2 -> 4 -> 8`). This keeps early search broad, then gradually shifts pressure toward richer structures.

This does **not** hard‑code a specific blueprint; it says “anything below this complexity band is not part of this phase of the experiment.”

## 3. Seeds matter: start in the right basin

Even with gates, starting from a trivial seed asks evolution to climb a huge hill under a tiny budget. We saw this when initial seeds had only a handful of layers and experts: complex variants appeared but were quickly out‑selected.

A much better pattern is:

- Build a **deep, structurally rich seed** that already satisfies the gates:
  - Enough layers.
  - Enough MoE blocks.
  - Enough selectors or memory modules.
  - Optional extras such as retro modules or recurrences scattered across depth.
- Use this as the base spec for live runs and resumes.

This is analogous to starting from a good baseline model: evolution then explores variations *within* a rich regime instead of having to discover that regime from scratch.

## 4. Budget and score weights decide what “good” means

We experimented with different score weights and budgets. The results were clear:

- Small budgets + generic weights → shallow models with minimal structure dominate.
- Larger budgets + structural gates + structural objectives → deeper, more structured architectures survive.
- Changing only weights, without gates, was not enough to escape the shallow basin.

Key knobs:

- **Steps / eval_batches** – more steps per candidate give complex architectures a chance to reduce perplexity and stabilize routing.
- **Score weights** – which metrics matter in parent selection:
  - Quality: `ppl_code`, `ppl_math`, `ppl_per_long_recall`.
  - Efficiency: `throughput`, `ram`, `ppl_per_param`, `ppl_per_throughput`.
  - Structure: `layers`, `moe_blocks`, `selector_blocks`, `graph_entropy`.
  - Memory/long‑term behaviour: `long_recall`, `recurrence_gain` (if present).

**Pattern:** To push evolution toward a specific family:

- Up‑weight metrics that reflect the desired behaviour (e.g. long‑range recall, expert usage, memory quality).
- Down‑weight or temporarily ignore others (e.g. throughput early on).
- Ensure the budget is large enough that complex candidates can improve these metrics.

## 5. Mutation mix is a policy, not a constant

Evolution only explores what mutations let it explore. We broadened the mutation space to include:

- **Depth / structure**
  - `duplicate_block_span`, `shuffle_block_span` – depth and ordering changes.
  - `remove_block_span` – contiguous shrink edits to counter one-way growth pressure.
  - `graph_jitter` – small neutral structural edits to increase entropy.
- **MoE / experts**
  - `dense_to_moe`, `mutate_topk`, `shift_moe`.
  - `moe_to_dense` – inverse simplification path from MoE back to dense FFN.
  - `tune_experts` – jitter expert counts and top‑k.
  - `tune_router`, `tune_router_coeffs` – router type, temperature, and load‑balance coefficients.
- **Attention**
  - `tune_rope` – RoPE theta jitter.
  - `tune_attn_gating` – on/off or change gating positions/ops.
  - `tune_attn_shape` – heads/head_dim/kv_groups while preserving model dim.
  - `tune_attn_sparsity` – sparsity mode and local/global window sizes.
  - `toggle_selector` – enable/disable selectors and retune top‑k/heads.
  - `toggle_qk_norm` – enable/disable QK norm clamping.
  - `simplify_attention` – convert specialized attention kinds back to `MHA`.
- **FFN / MLP**
  - `tune_ffn_width_activation` – jitter hidden size and activation type.
- **Memory / recurrence**
  - `insert_retro_module`, `tune_retro` – retro memory size, stride, and gating.
  - `add_recurrence`, `add_additional_recurrence`, `tune_recurrence` – recurrence spans and settings.
  - `remove_recurrence` – remove recurrence paths when they stop paying off.
- **Misc**
  - `insert_custom_module`, `toggle_gated_mix`, `toggle_ssm`, `tune_kv`, `strip_extras`, etc.

We also introduced:

- **Weighted mutation selection** – `--mutation-weight name=weight` lets us favour particular mutation types.
- **Multi‑step mutation** – `--mutation-steps N` chains N mutations per child, so each candidate can undergo a compound transformation.
- **Dynamic mutation registration** – template mutations are registered as individual `tpl::...` entries (and plugins can register new entries), so adaptive weighting can learn at per-mutation granularity.

**Pattern:** For any target regime:

- Include mutations that directly move the levers you care about.
- Include both grow and shrink operators so search can recover from over-complexification.
- Up‑weight those mutations so they occur frequently.
- Allow multi‑step mutations so richer edits can occur in one generation.

## 6. Preserve structural diversity (structural elites + novelty)

Even with gates and a rich mutation set, it is easy for the pool to collapse to a narrow set of patterns. We mitigated this via:

- **Structural elites**:
  - Maintain a small set of candidates chosen purely by structural score (e.g., weighted combination of layers, MoE blocks, selector blocks).
  - These elites are protected from trimming even if they are slower or slightly worse on short‑term perplexity.
- **Archive novelty and graph entropy objectives**:
  - Reward candidates by kNN sparseness against a novelty archive, not only parent-relative distance.
  - This captures “different from what has been seen globally,” not just “different from my parent.”
  - Encourage multiple lineages instead of a single scaffold with minor hyperparameter tweaks.
- **Complexity banding (optional MAP-Elites keying)**:
  - Add a coarse complexity band to archive keys to prevent early simple models from crowding out nascent complex candidates.

**Pattern:** To avoid premature convergence:

- Keep a few structurally rich candidates alive regardless of short‑term metrics.
- Reward structural novelty and entropy alongside quality/efficiency.

## 7. Crossover quality depends on alignment

Naive positional crossover is easy but fragile: index-based splicing often mixes non-homologous blocks and limits checkpoint transfer utility.

The current approach aligns blocks using structural similarity plus lineage IDs (`origin_id`, `parent_origin`), then:

- Inherits aligned/disjoint blocks with fitter-parent bias.
- Transfers checkpoint tensors using explicit source maps instead of pure positional index.
- Emits per-child crossover reports (`matched`, `disjoint`, transfer/dropped tensor counts) into lineage artifacts.

**Pattern:** If crossover is a core operator, treat block identity and alignment as first-class data; otherwise crossover becomes mostly random restart pressure.

## 8. What the strict deep runs validated

In the strict deep experiments, we combined:

- Deep, structurally rich seed.
- Hard gates on layers, MoE count, and selector count.
- Multi‑step, weighted mutations focused on depth, MoE, selectors, routing, and memory.
- Structural elites and novelty.
- Quality‑oriented score weights (long‑range recall, MoE, layers) with throughput de‑emphasized.

The resulting frontier consistently contained:

- Deep stacks (12–13 layers on the small surrogate).
- Several MoE blocks (5–6 per architecture).
- Several selector‑enabled attention blocks (6–7).
- Retro extras spread across depth.

This is a qualitatively different outcome from the initial shallow runs and aligns with the “multi‑branch + sparse + expert” regime we intended to explore.

## 9. General recipe for targeting a new family of architectures

Given the above, a useful mental checklist for future experiments is:

1. **Define the regime**
   - What structural features should be present? (e.g., depth, number of experts, memory modules, recurrences, sparsity patterns.)
   - Add hard gates to enforce minima for those features.

2. **Seed in the right basin**
   - Construct a seed config that already satisfies the gates and exhibits the structural motifs you want.
   - Avoid starting from an almost empty scaffold.

3. **Expose the right mutations**
   - Add mutation operators that can move all relevant knobs: structure, attention, FFN, routing, memory, etc.
   - Ensure they are registered and debuggable.

4. **Bias the mutation mix**
   - Assign higher weights to the mutations that most directly explore the target regime.
   - Use multi‑step mutations to allow compound edits.

5. **Tune budgets and score weights**
   - Choose steps/eval_batches so complex candidates can improve.
   - Align score weights with the desired properties (quality per token, long‑range behaviour, expert usage, memory quality).

6. **Preserve diversity**
   - Use structural elites and novelty/entropy objectives to keep multiple structurally distinct lineages alive.

Following this recipe, the search is not “magical,” but it becomes a controlled way to *probe a particular architectural family* under realistic resource constraints. The small surrogate then acts as a fast‑feedback environment for discovering promising motifs to scale up. 
