Tomorrow's Parameter Golf seeds are:

- [pg_tomorrow_seed_10l_bigramhash_mlp3x.yaml](/Users/rohit/Documents/Workspace/Coding/transformer-evolution-llm/configs/pg_tomorrow_seed_10l_bigramhash_mlp3x.yaml)
  This is the straight "bigger MLP plus hashed bigrams" seed. It is meant to start from a public-style strong baseline without too many moving parts.

- [pg_tomorrow_seed_11l_smear_bigram.yaml](/Users/rohit/Documents/Workspace/Coding/transformer-evolution-llm/configs/pg_tomorrow_seed_11l_smear_bigram.yaml)
  This adds the smoothing gate on top of the bigram-heavy recipe, so we can test a richer version of the public "SmearGate plus BigramHash" direction.

- [pg_tomorrow_seed_11l_xsa_partial_rope.yaml](/Users/rohit/Documents/Workspace/Coding/transformer-evolution-llm/configs/pg_tomorrow_seed_11l_xsa_partial_rope.yaml)
  This is the sparse-attention seed. The last few layers start with the selector-based attention idea already turned on, so search can refine it instead of having to rediscover it.

- [pg_tomorrow_seed_11l_hybrid_longctx.yaml](/Users/rohit/Documents/Workspace/Coding/transformer-evolution-llm/configs/pg_tomorrow_seed_11l_hybrid_longctx.yaml)
  This is the most ambitious seed. It mixes our own embedding-fed block idea with longer context, richer rotary settings, sparse tail attention, and lightweight stabilizers.

Common intent for all four:

- start from richer models instead of the undersized branch
- keep them near the top end of the allowed size range
- leave the mutation space broad so TEVO can still grow them in multiple directions
- treat them as tomorrow's first serious starting points, not final answers

All four also set an explicit edge target now, so the search is no longer relying on the old hard-coded default. The two simpler seeds aim for 98% of the limit, and the two more ambitious seeds aim for 98.5%.
