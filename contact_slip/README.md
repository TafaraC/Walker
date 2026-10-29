# Independent contact-slip experiments

Latest alternative: `./run_slip_sensitivity.sh smoke --suite scalar` tests a
single-argument local scene setter while auditing contacts passively. Five oracle
settings, 3,000 steps each; results in experiments/slip_sensitivity/oracle_scalar_v1.
This is not Gaussian contact noise or a solver override. See PIPELINES.md for
the method, provenance and validation requirements; live effectiveness is pending.

For the current **passive physics sensitivity pilot**, use
`./run_slip_sensitivity.sh smoke`, then `./run_slip_sensitivity.sh evaluate`.
This separate workflow does not randomize or override contacts. It logs
solver-bound slip/enable flags at the existing field slips .025/.100/.200.
See PIPELINES.md for scope and artifacts. Offline checks:
`./run_slip_sensitivity.sh selftest`. Live verification remains user-run.
The independent contact-noise workflow below is a different experiment.

This is a separate frozen-policy robustness batch for FC Portugal (unchanged),
BC, mixed DAgger, GAIL-BC and GAIL-DAgger. It does not alter or train the policies.
It does not replace the whole-run Gaussian training/evaluation pipeline.

## Exact definition

Every qualifying **ODE contact point/joint creation** gets a fresh independent
truncated-Gaussian draw. A planted foot can generate several contact points per
physics step, and new draws occur again in subsequent collision steps. This is
not one draw per foot strike, per agent step, per episode or per spatial location.
All body/ball contacts with the field qualify, including startup and fall recovery.

Default parent mean 0.100, std 0.035, bounds [0.025,0.200]. Rejection sampling,
not clipping; one scalar is applied to both tangential slip directions. Slip
stays hidden from the policy. Repeated contact at the same position need not have
the same value. Thus this is stochastic contact dynamics, not a fixed Gaussian
surface map. The default batch uses five task seeds and five paired contact RNG
seeds; a policy's contact history determines how many values it consumes. Equal
seeds do not guarantee matched physical events or maps between policies.

## Physics implementation and controls

A workspace-local C++ shared library intercepts ODE `dJointCreateContact` using
LD_PRELOAD only for the owned server subprocess. It copies the contact, changes
only the two slip values and enables their mode bits, then calls real ODE. It
does not overwrite system libraries or change the oracle, Python observation,
action or fall-recovery logic. The installed ODE headers define the contact layout;
the loaded runtime precision is checked before use. Build/runtime hashes are saved.

Selection: a horizontal PlaneCollider at z=0, with contact position inside the
standard 30m × 20m field. Wall contacts and contacts outside that rectangle are
excluded. Other field sizes/ground geometries require an explicit implementation
revision. Live audit must show nonzero qualifying contacts or the run fails.

The sampled value is the **effective solver contact slip**, after SimSpark's
surface-combination logic, not the field's `setContactSlip` input. Therefore
numerically equal bounds are not directly equivalent to the preceding field-slip
study. This distinction matters when robot and field parameters are combined.

Three controls, all using fresh servers and the same task seeds:

- `scene`: unmodified nominal scene; no interposer.
- `constant`: interposer supplies effective slip 0.100 for every qualifying contact.
- `gaussian`: interposer supplies a new bounded Gaussian value for every contact.

Constant-versus-Gaussian isolates added contact randomness. Scene-versus-constant
reveals the difference between ordinary surface mixing and overriding effective
slip. These are new controls for this mechanism, not a repeat of the three fixed
field-slip settings. Do not pool these results with prior fixed/whole-run Gaussian
results as if the physical manipulation were identical.

ODE contact representation is defined in installed `/usr/include/ode/contact.h`.
SimSpark's [contact handler documentation](https://robocup-sim.gitlab.io/SimSpark/doc/simspark/classContactJointHandlerImp.html)
describes combination of the two objects' surface parameters.

## Commands

From Walker, after any existing simulator experiment finishes:

```bash
./run_contact_slip.sh selftest
./run_contact_slip.sh plan
./run_contact_slip.sh smoke
./run_contact_slip.sh evaluate
./run_contact_slip.sh report
```

`selftest` compiles the local library and runs a small standalone ODE integration
probe, without SimSpark/network. It checks 1,000 sampled contacts, bounds/moments,
same-seed replay, and exclusion of wall/out-of-field contacts. Requires g++ and
libode-dev matching the installed ODE runtime. It never installs system packages.

`smoke` runs the oracle for 3,000 active steps in the Gaussian contact environment.
Review contact audit statistics, progress and falls before full evaluation.
`evaluate` runs 3 modes × 5 agents × 5 seeds = 75 runs of 3,000 active steps.
Scripts run to completion without prompts, stopping on errors. A shared lock
prevents concurrency with fixed/whole-run Gaussian pipelines. Each run owns a
fresh verified server, watchdog and cleanup. Rerun the same command to validate
and skip completed artifacts; failed attempts are preserved before retry.

Default outputs: `experiments/contact_slip/gaussian_v1/`. Use a new directory when
changing parameters:

```bash
./run_contact_slip.sh plan --output experiments/contact_slip/gaussian_v2 \
  --low 0.025 --high 0.200 --mean 0.100 --std 0.035
```

Supply that `--output` on all subsequent commands. No distributions are revised
in place. Native binary changes require compatible new experiments; no live
server is started by build/plan/report commands.

## Evidence and limitations

Each run retains scene/config manifests, native/library/source fingerprints,
server/evaluation logs, normal evaluation traces, metrics and a completion receipt.
`contacts.csv` logs each qualifying contact's sequence number, position, original
slip components and applied scalar immediately before passing it to ODE. Logging
is flushed per contact to preserve native-crash evidence. This can increase wall
time/disk use; synchronous simulated-time metrics remain the primary comparison.
Contact statistics include warm-up/recovery, whereas locomotion metrics cover only
active evaluation. Existing `field_contact_slip=0.1` metadata denotes the static
scene, not the randomized effective coefficient; summaries state this explicitly.

Reports provide raw progress and falls plus all ordinary evaluator metrics in
JSON. Random control is outside scope; do not claim random-normalized return.
The native probe verifies interception with the installed ODE library, but does
not validate SimSpark's geometry selection, plugin symbol loading or robot stability.
The user-run smoke test must validate that integration. Absence of contact evidence
is an error, not permission to treat the run as a successful Gaussian experiment.

Native selftest passed during implementation; live SimSpark runs have not started.
