# Attribution and modification notice

## Original software

This repository is derived from the
[FC Portugal Codebase](https://github.com/m-abr/FCPCodebase), a Python/C++ agent
framework for the RoboCup 3D Soccer Simulation League. The upstream authors and
contributors retain credit for their work, including the expert walking policy,
gait generation, inverse kinematics, robot/world interfaces, communication and
supporting skills. Upstream Git history and existing notices are preserved.

The inherited checkout's latest upstream commit is
`22d9e2f` ("Update README.md"). Local research changes build on that history;
this notice does not attribute those changes to the upstream authors.

Upstream's recommended academic citation is:

> Miguel Abreu, Luis Paulo Reis and Nuno Lau (2025).
> Designing a skilled soccer team for RoboCup: exploring skill-set-primitives
> through reinforcement learning. Neural Computing and Applications.
> DOI: [10.1007/s00521-025-11151-3](https://doi.org/10.1007/s00521-025-11151-3).

The BibTeX entry is retained in the root README and the paper bibliography.

## Research additions and modifications

The modified research version is maintained in
[TafaraC/Walker](https://github.com/TafaraC/Walker). This notice was added on
29 October 2026. The project's experimental phases and modification chronology
are documented in [EXPERIMENT_PHASES.md](EXPERIMENT_PHASES.md),
[CHALLENGES.md](CHALLENGES.md) and the Git history as changes are committed.

Research contributions include:

- BC, mixed DAgger and GAIL collection, fitting and rollout pipelines.
- Simulator ownership, synchronization checks, fall/reset handling and
  controlled slip experiment orchestration.
- Standardized training allocations, initialization/context comparisons and
  additional optimization controls.
- Evaluation metrics, representation probes, analysis, figures and research documentation.
- Adaptations to the original walking and communication code supporting these experiments.

These are research extensions; they do not imply endorsement by FC Portugal.
Academic author details will be supplied by the researcher before submission.

## Licence and dependencies

The derived software is distributed under the existing
[GNU GPL version 3](LICENSE). Preserve that licence and original copyright/legal
notices when redistributing this version. The licence text has not been replaced
or edited. This attribution supplements the licence and does not replace it.

External software such as SimSpark/rcssserver3d, RoboViz, PyTorch and scientific
Python packages retains its own authorship and licensing. Refer to those projects'
distributions for their notices. Citing a paper does not imply that its code was
incorporated into this repository.
