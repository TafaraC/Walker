# Walker: imitation learning for RoboCup locomotion

Research repository: [TafaraC/Walker](https://github.com/TafaraC/Walker).
This project studies behavioural cloning, mixed DAgger and GAIL using the native
FC Portugal walking controller as an expert, with controlled surface experiments,
mechanical evaluation and representation probing.

## Credit and licence

Built on the [FC Portugal Codebase](https://github.com/m-abr/FCPCodebase).
Credit for the original agent, walking controller, gait/IK machinery and supporting
code belongs to the FC Portugal authors and contributors. Their Git history,
copyright notices and original academic citation below are retained.

The derived software is distributed under the **GNU General Public License,
version 3**, provided in [LICENSE](LICENSE). Research additions and modifications
are described in [ATTRIBUTION.md](ATTRIBUTION.md); this repository is a modified
research version of the upstream project.

## Research entry points

- [Paper draft](evaluation/FINAL_PAPER_DRAFT.md) and [LaTeX package instructions](evaluation/paper/README.md).
- [Presentation](evaluation/presentation/CONSOLIDATED_PAPER_SLIDES.pdf) and [slide explanations](evaluation/presentation/CONSOLIDATED_PAPER_NOTES.md).
- [Findings and numerical results](evaluation/CONSOLIDATED_RESULTS_REPORT_2026-10-15.md) and [optimization controls](evaluation/SUBMISSION_FEEDBACK_RESPONSE_2026-10-28.md).
- [Setup workbook](docs/learning/README.md), [commands](COMMANDS.md) and [repository/artifact guide](REPOSITORY.md).

Large datasets, experiment outputs and learned checkpoints are distributed
separately from source control; artifact download details are pending.

![](https://s5.gifyu.com/images/Siov6.gif)

## About the upstream FC Portugal codebase

The FC Portugal Codebase was mainly written in Python, with some C++ modules. It was created to simplify and speed up the development of a team for participating in the RoboCup 3D Soccer Simulation League. We hope this release helps existing teams transition to Python more easily, and provides new teams with a robust and modern foundation upon which they can build new features.


## Documentation

The new standardized-budget study has two owned-server entry points:
`run_training.sh` and `run_evaluation_all.sh`. See
[consolidated/README.md](consolidated/README.md) for budgets, model lineage,
context variants, evaluation schedules and smoke commands.

For executable training, evaluation, probing and RoboViz demo commands, see
[COMMANDS.md](COMMANDS.md).

The documentation is available [here](https://docs.google.com/document/d/1aJhwK2iJtU-ri_2JOB8iYvxzbPskJ8kbk_4rb3IK3yc/edit)

The local BC, DAgger, and GAIL technical and execution guide is available in
[PIPELINES.md](PIPELINES.md).

Experiment progress, completed work, and the staged robustness plan are tracked
in [EXPERIMENT_PHASES.md](EXPERIMENT_PHASES.md).

For a new-session summary and the exact next commands, read
[SESSION_HANDOFF.md](SESSION_HANDOFF.md).

The implementation challenge log and unresolved research risks are maintained in
[CHALLENGES.md](CHALLENGES.md).

## Live walking demo

With SimSpark and RoboViz running, execute `./run_demo.sh` and choose an agent
by number. Each demo runs 1,500 active steps at 50 Hz (30 seconds, plus startup
and any recovery). See [PIPELINES.md](PIPELINES.md) for setup and menu options.

## Features

- The team is ready to play!
    - Sample Agent - the active agent attempts to score with a kick, while the others maintain a basic formation
        - Launch team with: **start.sh**
    - Sample Agent supports [Fat Proxy](https://github.com/magmaOffenburg/magmaFatProxy) 
        - Launch team with: **start_fat_proxy.sh**
    - Sample Agent Penalty - a striker performs a basic kick and a goalkeeper dives to defend
        - Launch team with: **start_penalty.sh**
- Skills
    - Get Ups (latest version)
    - Walk (latest version)
    - Dribble v1 (version used in RoboCup 2022)
    - Step (skill-set-primitive used by Walk and Dribble)
    - Basic kick
    - Basic goalkeeper dive
- Features
    - Accurate localization based on probabilistic 6D pose estimation [algorithm](https://doi.org/10.1007/s10846-021-01385-3) and IMU
    - Automatic head orientation
    - Automatic communication with teammates to share location of all visible players and ball
    - Basics: common math ops, server communication, RoboViz drawings (with arrows and preset colors)
    - Behavior manager that internally resets skills when necessary
    - Bundle script to generate a binary and the corresponding start/kill scripts
    - C++ modules are automatically built into shared libraries when changes are detected
    - Central arguments specification for all scripts
    - Custom A* pathfinding implementation in C++, optimized for the soccer environment
    - Easy integration of neural-network-based behaviors
    - Integration with Open AI Gym to train models with reinforcement learning
        - User interface to train, retrain, test & export trained models
        - Common features from Stable Baselines were automated, added evaluation graphs in the terminal
        - Interactive FPS control during model testing, along with logging of statistics
    - Interactive demonstrations, tests and utilities showcasing key features of the team/agents
    - Inverse Kinematics
    - Multiple agents can be launched on a single thread, or one agent per thread
    - Predictor for rolling ball position and velocity
    - Relative/absolute position & orientation of every body part & joint through forward kinematics and vision
    - Sample train environments
    - User-friendly interface to check active arguments and launch utilities & gyms

## Citing the Project

```
@article{abreu2025designing,
  title={Designing a skilled soccer team for RoboCup: exploring skill-set-primitives through reinforcement learning},
  author={Abreu, Miguel and Reis, Luis Paulo and Lau, Nuno},
  journal={Neural Computing and Applications},
  year={2025},
  doi={10.1007/s00521-025-11151-3}
}
```

![](https://github.com/user-attachments/assets/0893c6c7-ad88-46dc-bcd6-cc7a2c49c719)
RoboCup 3DSSL aggregated locomotion statistics for periods of notable change (2008-2013, 2017-2023). Metrics include average speed, average maximum speed, and maximum speed (all in m/s), plotted radially to show variation by movement direction, relative to the robot. Values at the top represent forward movement, values on the left indicate sideways movement to the left, and so on.
