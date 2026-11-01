# aero-surrogate

Open-source Python toolkit that automates 2D OpenFOAM cases for multi-element wings in
ground effect (FSAE front wing) and trains an ensemble neural surrogate that chooses its own
next CFD cases (active learning). See `SPEC.md`, `PLAN.md` and `NEXT_STEPS.md`.

> **Status (session 1, Nov 2026): pipeline built and tested end to end; results NOT validated.**
> All CFD so far uses coarse smoke-test meshes, which Phase 0 showed are not mesh-independent
> for separated ground-effect flow. Read `docs/phase0_report.md` and `docs/gates.md` before
> using any number.

## Phases and gates

| Phase | What | Gate status (details in docs/gates.md) |
| --- | --- | --- |
| 0 | OpenFOAM v2512 (Docker), single-element ground-effect sweep, mesh study, literature note | **Partial**: runs and converges; validation **pending** (sources blocked); not mesh-converged |
| 1 | Parametric 1-3 element geometry, gmsh BL meshing, Jinja2 OpenFOAM templates | **Passed**: 20/20 random cases pass checkMesh; geometry edge-case tests |
| 2 | Local/Docker runner with stop rules + failure logging, SSH/SLURM (UNTESTED), HTML reports, GCI | **Passed**, with a finding: 3-element S1223 cases separate on the flaps (see gates) |
| 3 | Sobol sampling, shared schema, NeuralFoil/AeroSandbox low-fi, PyTorch ensemble | Pipeline demonstrated on a **single-element** smoke dataset (see gates) |
| 4 | Active learning (ensemble variance) vs space filling | Benchmarked on a **synthetic** function and retrospectively on the smoke CFD data |
| 5 | Optimiser on the surrogate + aero-map export | Code + tests; demo export |
| 6 | STAR-CCM+ macro + importer | **UNTESTED** (no licence) |
| 7 | Real campaign, validation, paper | Not started (laptop, after exams) |

## Quick start

```bash
make install        # venv + pinned dependencies (requirements-lock.txt has the full lock)
make test           # unit tests, no OpenFOAM needed (~2-3 min)
make demo           # synthetic end-to-end demo, no OpenFOAM needed
```

With OpenFOAM (Docker image `opencfd/openfoam-default:2512`, or a native install via
`configs/openfoam/native_example.yaml`):

```bash
make demo-cfd       # one 3-element case end to end with an HTML report
make phase0         # Phase 0 sweep + mesh study
make gate1          # 20 random meshes through checkMesh
python -m aerosurrogate campaign --n 32 --out runs/campaign   # Sobol campaign -> runs/campaign/cases.csv
python -m aerosurrogate lowfi --dataset runs/campaign
python -m aerosurrogate train --dataset runs/campaign --ranges configs/ranges/front3.yaml
python -m aerosurrogate active --dataset runs/campaign --ranges configs/ranges/front3.yaml
python -m aerosurrogate optimise --surrogate runs/phase3/surrogate_cfd_only --ranges configs/ranges/front3.yaml
```

On a laptop without root Docker, set `kind: native` and your `bashrc` path in a launcher
config. Use `configs/case_lowre.yaml` (y+ ~ 1) as the starting point for real data, after a
mesh study (see `docs/phase0_report.md`).

## Layout

```
src/aerosurrogate/
  geometry/   airfoil sections (AeroSandbox database), wing assembly, validity checks
  mesh/       gmsh 2D hybrid mesh with boundary layers, extruded for OpenFOAM
  foam/       Jinja2 case templates, output parsers, post-processing (Cp, y+)
  runner/     launchers (docker/native), local runner + stop rules, campaigns, SSH/SLURM (UNTESTED)
  report/     HTML case reports, plots with provenance, GCI (Celik et al. 2008)
  data/       shared schema (OpenFOAM / STAR-CCM+ / NeuralFoil / synthetic) and datasets
  sampling/   parameter spaces from YAML, scrambled Sobol
  baseline/   low-fidelity: AeroSandbox inviscid panel (ground images) + NeuralFoil drag
  surrogate/  PyTorch deep ensemble, case-wise splits, metrics
  active/     active learning loop, synthetic test function
  optimise/   differential evolution on the surrogate, aero-map export
  starccm/    Java macro template + importer (UNTESTED)
configs/      ranges, case settings, OpenFOAM launchers
docs/         conventions, Phase 0 report, gate reports, literature note, export format
tests/        unit tests (fixtures taken from real OpenFOAM v2512 output)
```

## Data schema

One CSV row per case (`cases.csv`) + `metadata.json` + one folder per case. Required columns
(SPEC.md): `case_id, solver, solver_version, turbulence_model, mesh_cells, y_plus_max,
geometry_params (JSON), aoa_deg, re, ride_height_over_chord, cl, cd, cm, converged,
residual_final, wall_clock_s, source`. Failed cases stay in the file with `status` and
`failure_reason`; training uses only converged rows. Conventions: `docs/conventions.md`.

## Known risks (carried into every report)

1. RANS accuracy on separated multi-element flow near a moving ground is uncertain; a surrogate inherits CFD error.
2. Validation needs published ground-effect data and STAR-CCM+ runs; none has been done yet.
3. The training campaign runs on a laptop whose specs are not yet known.
4. Novelty is unproven: the closest 2025 FSAE paper has not been read (docs/literature_note.md).

## Tools used, not rebuilt

OpenFOAM (ESI/OpenCFD), gmsh, AeroSandbox and NeuralFoil (Sharpe & Hansman, arXiv:2503.16323),
PyTorch, SciPy, pandas, Jinja2. Versions pinned in `pyproject.toml` and `requirements-lock.txt`.

## Licence

MIT (to confirm with the team before publishing).
