# Gate reports (session 1, 1 Nov 2026)

Mode agreed with Chaitra at session start: run each gate's tests, write the report here, and
continue past passed gates; stop for failures or decisions only she can make. Every number
below comes from files in `docs/results/` or `docs/phase0/`. All CFD is **coarse
smoke-test data, not validated**.

## Gate 0 (Phase 0): PARTIAL

See `docs/phase0_report.md`. Passed: OpenFOAM v2512 runs (Docker), cases converge
iteratively, runtimes recorded, literature note written (from abstracts only). Not passed:
**validation pending** (sources blocked; you chose to allow the domains, but they were still
blocked at the end of the session). Finding: coarse meshes **not mesh-converged** (CL varies
by up to ~35 % across 12k/25k/52k cells at h/c = 0.2).

## Gate 1 (Phase 1): PASSED

- 20 random Sobol cases from the full 9-parameter space (`configs/ranges/front3.yaml`, seed 101):
  **20/20 meshes pass checkMesh** (`docs/results/gate1/meshcheck_summary.csv`), 14.5k-19.4k cells.
- Unit tests cover geometry edge cases: overlapping elements, self-intersection, tiny clearance,
  bad parameter counts/values, extreme overlap, orientation and ride-height definitions
  (`tests/test_wing.py`, `tests/test_polygon.py`).
- 64/64 Sobol samples in the full space are geometrically valid (gap solve + checks).
- Caveat: 6/20 meshes have 2 faces with non-orthogonality 86-88 deg, located on the ground
  boundary layer ~2 chords upstream of the wing (checkMesh warning, not failure).
- Bug found and fixed on the way: collinear-but-separate edges were reported as intersecting.

## Gate 2 (Phase 2): PASSED, with an important finding

- End to end: the nominal 3-element case (`configs/demo_case_params.yaml`) ran gmsh, gmshToFoam,
  checkMesh, simpleFoam and post-processing, and produced an HTML report
  (`docs/results/gate2/g2_nominal_report.html`).
- Failure handling: a deliberately wrong turbulence model was caught as `solver_failed` with
  OpenFOAM's message; a 0.001c gap was caught as `geometry_invalid`. Both rows are in
  `docs/results/gate2/gate2_cases.csv` with `converged = False` and a reason: not dropped.
- Local/Docker runner, force-convergence stop rule, divergence/timeout detection: tested.
  **SSH and SLURM runners: UNTESTED** (no remote host/cluster; only command/script generation tested).
- **Finding:** the nominal 3-element S1223 wing (AoA 4, flaps 20/20 deg, gaps 0.015, overlaps
  0.02, h/c 0.2, Re 5e5) does **not converge** and shows separated flaps (CD 0.62; flaps carry
  CD 0.39 and 0.28). A milder setting (flaps 8/8, AoA 2) still separates (CD 0.12-0.14) on both
  the wall-function mesh and a low-Re (y+ ~ 1, 17 layers, potentialFoam start) mesh. Slot
  refinement (~10 cells across) did not change this. Cause **not established**: possibly
  physical (S1223 as a flap section is extremely cambered), possibly the placement convention,
  possibly numerics. **This needs your input: what sections, chords and angles does the
  Formula Manipal front wing actually use?**

## Gate 3 (Phase 3): PASSED on a substitute dataset (single element)

Because of the Gate 2 finding, the workspace dataset is **single-element in ground effect**
(`configs/ranges/single_ge.yaml`: AoA 0-8, h/c 0.1-0.6, Re 2e5-1e6; ls417 stand-in), not the
SPEC's 3-element wing. 32 Sobol cases: 26 converged, 6 not converged (kept, flagged, excluded
from training). Low-fidelity rows for all 32 (`source = neuralfoil`). Five random case-wise
splits, 25 % test (6-7 cases), ~20 training cases.

| model | CL RMSE | CL R2 | CD RMSE | CM RMSE | 2-sigma coverage (CL) |
| --- | --- | --- | --- | --- | --- |
| NeuralFoil/AeroSandbox only (baseline) | 0.583 +/- 0.211 | -5.74 | 0.0225 | 0.0654 | - |
| baseline + linear correction fitted on training cases | 0.187 +/- 0.107 | 0.36 | 0.0090 | 0.0170 | - |
| ensemble surrogate, CFD only | 0.103 +/- 0.046 | 0.80 | 0.0036 | 0.0128 | 0.77 |
| ensemble surrogate, multi-fidelity (low-fi as inputs) | **0.069 +/- 0.028** | **0.90** | 0.0036 | 0.0102 | 0.90 |

(+/- = sd over the 5 splits; `docs/results/phase3/metrics_summary.csv`, parity plot
`parity_split0.png`.) Honest caveats: tiny data; CM is poorly predicted (R2 < 0 for all
models: small spread, noisy CFD); the ensemble's uncertainty is too narrow (coverage
0.73-0.90 vs ~0.95 expected); coarse-mesh CFD error (Phase 0) is far larger than these
surrogate errors, so these numbers describe the surrogate, not the physics.

## Gate 4 (Phase 4): REPORTED NEGATIVE: active learning did NOT beat space filling

- Synthetic function (`active/synthetic.py`, 5 parameters, labelled SYNTHETIC; pool 256 Sobol,
  start 8, batches of 4, budget 48, 3 seeds): neither reached CL RMSE 0.05. Active learning was
  **worse** at every budget (CL RMSE at 48 runs: 0.63 +/- 0.06 vs 0.53 +/- 0.04 for Sobol).
  `docs/results/phase4/al_synthetic.png`.
- Retrospective on the Phase 3 CFD cases (pool = training cases, 3 seeds): both reached CL RMSE
  0.1 after 14 runs on the mean curve; at the full pool (20 runs) Sobol was better (0.068 vs
  0.107). `docs/results/phase4/al_cfd_retrospective.png`.
- Likely reasons (hypotheses, untested): pure ensemble-variance acquisition chases box
  corners where the ensemble extrapolates; the variance is a poor error proxy at this data size
  (see coverage above). Candidates to try: integrated-variance-reduction acquisition, mixing
  Sobol and variance picks, larger initial design, a GP baseline.

## Phase 5 (no formal gate): done on the substitute dataset

- DE on the CFD-only surrogate, pessimistic objective (-CL - 1 sigma): all 4 starts agree on
  AoA = 8.0 deg (**upper bound**, flagged), h/c = 0.286, Re = 7.1e5; predicted CL = -1.730 +/- 0.109.
- CFD verification of that point: **CL = -1.601**, CD = 0.0625 (predicted 0.0587). The surrogate
  over-predicted downforce by 8 % (1.2 sigma) at its own optimum, the expected optimiser bias.
- Aero map exported (`docs/results/phase5/aero_map.csv/.json`, format in
  `docs/coefficient_format.md`); the h/c = 0.05 row is correctly flagged `extrapolated`.

## Phase 6: written, UNTESTED

Java macro template + importer (`src/aerosurrogate/starccm/`). Only the Python side is tested,
against a hand-written synthetic fixture. Needs a licence run.

## Not verified anywhere in this session

Validation against experiments; mesh independence; the 3-element flow physics; laptop
performance; SSH/SLURM execution; STAR-CCM+ macro and real CSV layout; native (non-Docker)
OpenFOAM launcher.
