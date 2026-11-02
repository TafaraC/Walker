# Walker: imitation learning for RoboCup locomotion

[![Code DOI](https://zenodo.org/badge/1438268704.svg)](https://doi.org/10.5281/zenodo.23884771)
[![Data DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23894016.svg)](https://doi.org/10.5281/zenodo.23894016)

- **Code:** archived releases, [doi:10.5281/zenodo.23884771](https://doi.org/10.5281/zenodo.23884771)
- **Data:** experiment outputs, datasets and checkpoints (about 32 GB), [doi:10.5281/zenodo.23894016](https://doi.org/10.5281/zenodo.23894016)
- **Video:** walking demos of the expert, learners and controls, [youtu.be/5fa8AXPmhqg](https://youtu.be/5fa8AXPmhqg)

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

- [Research report (PDF)](Behavioural_Cloning_DAgger_GAIL_RoboCup_Walking.pdf): *Behavioural Cloning, DAgger and GAIL for RoboCup Walking: Comparisons Across Training Resource Allocations*, T. Chibanda, B. Ingram and P. Ranchod, University of the Witwatersrand, 2026.
- [Paper figure script](evaluation/figures_2026-10-31_paper/README.md), which draws all six paper figures, and the [extended-fitting presentation figures](evaluation/figures_2026-10-30/README.md).
- [Presentation](evaluation/presentation/CONSOLIDATED_PAPER_SLIDES.pdf) and [slide explanations](evaluation/presentation/CONSOLIDATED_PAPER_NOTES.md).
- [Findings and numerical results](evaluation/CONSOLIDATED_RESULTS_REPORT_2026-10-15.md) and [optimization controls](evaluation/SUBMISSION_FEEDBACK_RESPONSE_2026-10-28.md).
- [Exploratory analysis and measurement audit](evaluation/exploratory_2026-10-29/REPORT.md) and [12-figure atlas](evaluation/exploratory_2026-10-29/EDA_FIGURES.pdf).
- [Extended-fitting results, engine controls and new probes](evaluation/convergence_2026-10-30/REPORT.md) and [11-figure atlas](evaluation/convergence_2026-10-30/EDA_FIGURES.pdf).
- [Setup](#setup) (below), the [setup workbook](docs/learning/README.md), [commands](COMMANDS.md) and [repository/artifact guide](REPOSITORY.md).

Large datasets, experiment outputs and learned checkpoints are distributed
separately from source control; artifact download details are pending.
The older [Markdown paper draft](evaluation/FINAL_PAPER_DRAFT.md) is retained as a historical draft.
The presentation and October 15 report predate the final extended-fitting analysis.

![](https://s5.gifyu.com/images/Siov6.gif)

## Setup

The code runs on Linux; it was developed on Ubuntu 24.04 and also runs in WSL 2 on Windows. There are two levels of setup: regenerating the paper's figures needs only Python, while running the experiments needs the RoboCup 3D simulator.

### 1. Clone the repository

```bash
git clone https://github.com/TafaraC/Walker.git
cd Walker
```

### 2. Quick start: regenerate the paper's figures (no simulator)

The evaluation records behind the figures are in `evaluation/paper_inputs/`, so this works on a fresh clone:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib
python evaluation/figures_2026-10-31_paper/make_paper_figures.py --out paper_figures
```

The six figures are written to `paper_figures/`. See the [figure script README](evaluation/figures_2026-10-31_paper/README.md) for details.

### 3. Full setup: run the experiments

**System packages.** Install a compiler, Git and Conda (or Miniconda):

```bash
sudo apt update
sudo apt install build-essential git
```

**Simulator.** Install SimSpark and the RoboCup 3D server `rcssserver3d` following the official [SimSpark instructions](https://gitlab.com/robocup-sim/SimSpark). The paper used SimSpark 0.3.8 and rcssserver3d 0.7.9, installed under `/usr/share/rcssserver3d` (the path the managed launchers expect). To watch the robot, also install the [RoboViz](https://github.com/magmaOffenburg/RoboViz) viewer (requires Java); it is optional for running experiments.

**Simulator settings.** The experiments need synchronous, non-real-time simulation. Set these two lines (Ruby syntax) and restart any running server:

```ruby
$agentSyncMode = true          # in ~/.simspark/spark.rb
$enableRealTimeMode = false    # in rcssserver3d.rb (e.g. /usr/share/rcssserver3d/rcssserver3d.rb)
```

**Python environment.** Create a Python 3.9 environment and install the dependencies. For a GPU, install the matching PyTorch build from [pytorch.org](https://pytorch.org) before the last command.

```bash
conda create -n walker python=3.9
conda activate walker
pip install -r requirements.txt
```

The launchers (`run_*.sh`) use the Python in `PYTHON_BIN`. Point it at this environment (add the line to `~/.bashrc` to keep it):

```bash
export PYTHON_BIN="$(command -v python3)"
```

**Build and check.** Build the C++ modules, then check the imports and the simulator settings:

```bash
python -c "from scripts.commons.Script import Script; Script.build_cpp_modules()"
python -c "from agent.Base_Agent import Base_Agent; print('agent imports OK')"
python check_simulator_config.py
```

**First run.** A small smoke run trains and evaluates tiny models end to end. The launchers start and stop their own simulator, so leave ports 3100 and 3200 free and do not start `rcssserver3d` yourself:

```bash
./run_training.sh run --quick --output experiments/learning/smoke_01
./run_evaluation_all.sh run --quick --output experiments/learning/smoke_01
```

A smoke run checks that everything works; its tiny budgets do not reproduce the paper's results. Datasets, checkpoints and full experiment outputs are not in Git (see [REPOSITORY.md](REPOSITORY.md)), so commands that need the completed study's models will not run on a fresh clone.

**Next steps.** [COMMANDS.md](COMMANDS.md) lists the training, evaluation and viewer commands (written for the original machine: replace `/home/pamnt/...` paths and the `fcportugal` environment name with your own). [PIPELINES.md](PIPELINES.md) describes each pipeline, and the [setup workbook](docs/learning/README.md) explains the environment, simulator settings and common problems in depth.

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

[![Walking demos](https://img.youtube.com/vi/5fa8AXPmhqg/0.jpg)](https://youtu.be/5fa8AXPmhqg)

The [demo video](https://youtu.be/5fa8AXPmhqg) shows 30-second runs of the expert,
matched-update BC, DAgger, GAIL-BC, the zero-action, untrained-network and
uniform-random controls on the original field, and BC and the expert at a fixed
slip of 10, all with the same targets (seed 101). The video description lists the
exact model running in each segment, with its timestamp, targets reached and falls.
It was recorded with
`./record_demos.sh --auto`, which runs the viewer for each clip in turn and saves
each clip's model, conditions, targets reached and falls under `logs/demos/`.
The slip-10 runs use a fixed setting, not the paper's sampled evaluation rounds.

To watch a model yourself, start RoboViz and run `./run_viewer.sh` (or
`./run_viewer.sh --list` to see the available models), or `./run_demo.sh` for the
original menu. Each demo runs 1,500 active steps at 50 Hz (30 seconds, plus startup
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
