# Legacy imitation-learning files

These files predate the maintained BC, DAgger, and GAIL pipelines:

- `zcollect_expert_data.py`: superseded by `collect_long.py` and the synchronous
  BC pipeline.
- `zcollect_short_turns_data.py`: superseded by `collect_short.py`.
- `zexpert_walk_data.npz`: dataset produced by the legacy collectors.

They are retained for provenance only. They use outdated target bounds,
asynchronous loop ordering, dataset handling, and fall recovery. Do not use them
for current experiments or comparisons. See `../../PIPELINES.md` for supported
commands.
