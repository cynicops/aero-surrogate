# PLAN: aero-surrogate

Target budget for the credit window: about $19 (a target, not a measurement). Stop at each gate.

## Phase 0: feasibility spike and literature note
Tasks:
1. Install OpenFOAM in the workspace (try the ESI repository; record the exact version). If it fails, record why.
2. Run one 2D single-element-in-ground-effect case on a coarse mesh. Record cells, wall-clock time on 2 cores, convergence.
3. Compare lift against published ground-effect wing data (find and cite the reference; do not invent values).
4. Write `docs/literature_note.md`: NeuralFoil, the F1 front-wing PINN paper, the active multi-fidelity airfoil paper, plus the two papers not opened before (read or flag).
Gate 0: the case converges; runtime per case is known; lift is within a tolerance we agree on from the published data; the literature note exists. If not: stop and propose a fallback (XFOIL/VLM-only surrogate).

## Phase 1: case generator
- Parametric geometry for main + 1 to 2 flaps (gap, overlap, angle, ride height, AoA, Re).
- gmsh meshing with boundary-layer layers; templated OpenFOAM dictionaries (Jinja2); config files for every case.
Gate 1: 20 random valid cases generate meshes that pass checkMesh; unit tests cover geometry edge cases (overlaps, self-intersection).

## Phase 2: runner and reports
- Local / SSH / SLURM runner; residual and force-convergence stop rules; failure handling.
- Reports: Cp, CL/CD/CM, y+ histogram, mesh quality, GCI; HTML first, PDF optional.
Gate 2: one case runs end to end and produces a report; a failed case is detected and logged, not silently dropped.

## Phase 3: dataset and baseline surrogate
- Sobol sampling over the parameter ranges; data stored in the shared schema.
- Cheap baseline data from NeuralFoil/AeroSandbox (marked `source=neuralfoil`).
- Ensemble neural net (PyTorch) for CL, CD, CM; train/validation/test split by case, not by row.
Gate 3: baseline surrogate trains; metrics are reported on a held-out test set; comparison against a NeuralFoil/AeroSandbox-only baseline is included.

## Phase 4: active-learning loop
- Ensemble-variance acquisition; retraining; stop rule; plot of CFD runs saved versus accuracy.
Gate 4: on synthetic and (small) CFD data, the loop reaches the target error with fewer runs than space-filling sampling, or the report says it did not.

## Phase 5: optimiser and export
- Bayesian or gradient optimisation on the surrogate; verify top candidates with CFD; export CL/CD/CM maps in a documented format for a lap-time simulator.

## Phase 6: STAR-CCM+ adapter (UNTESTED here)
- Java macro templates and an importer into the shared schema. Mark every file UNTESTED until I run it on my licence.

## Phase 7: paper pass (after exams, on my laptop)
- The real CFD campaign, validation against my STAR-CCM+ runs, ablations, cleaned repo and dataset.

## Definition of done for the 5-day window
Phases 0 to 3 complete with gates passed, Phase 4 started if budget remains, `NEXT_STEPS.md` lists everything UNTESTED, and the repo is on GitHub.
