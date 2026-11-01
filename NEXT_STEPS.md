# NEXT_STEPS: aero-surrogate

Last updated: 1 Nov 2026 (end of session 1)

## Resume here
1. Repo: https://github.com/cynicops/aero-surrogate (pushed 1 Nov 2026). Commit and push after every work block.
2. **Decide the real 3-element geometry** (sections, chord split, flap angle ranges). With S1223
   on all three elements, 2D RANS separates on the flaps even at mild settings (docs/gates.md, Gate 2).
   Then re-run Gate 2's nominal case and a small 3-element campaign.
3. **Validation (Gate 0)**: arxiv.org, core.ac.uk, eprints.soton.ac.uk and www.sciencedirect.com
   were still blocked at the end of session 1 even after you chose to allow them; network changes
   may only apply to a new session. Or upload the Zerihan & Zhang paper / Zerihan thesis PDF.
   Then fill `docs/validation/reference_lift.csv` and set the section/AoA/Re in `phase0.py`.
4. **Production mesh study** before generating training data: `configs/case_lowre.yaml`
   (y+ ~ 1) at 3 refinement levels and 2 ride heights; consider `div_turb: bounded Gauss limitedLinear 1`.

## Done (details: docs/gates.md, docs/phase0_report.md)
- Phase 0 (partial): OpenFOAM **v2512** via the official Docker image (build `_87ed40d2-20251219`);
  single-element ride-height sweep; mesh study (not mesh-converged); NeuralFoil sanity check;
  literature note from abstracts. Validation PENDING.
- Phase 1 (gate passed): parametric 1-3 element geometry, gmsh BL meshing, Jinja2 templates;
  20/20 random meshes pass checkMesh.
- Phase 2 (gate passed, with finding): Docker/native runner, stop rules, failure logging, HTML
  reports, GCI; SSH/SLURM written (UNTESTED).
- Phase 3 (gate passed on a **single-element substitute** dataset): 32 CFD cases, low-fi rows,
  PyTorch ensemble; multi-fidelity surrogate CL RMSE 0.069 vs 0.583 for the NeuralFoil/AeroSandbox baseline.
- Phase 4 (reported negative): active learning did NOT beat Sobol sampling, on synthetic or retrospective CFD.
- Phase 5: optimiser + aero-map export; top candidate verified by CFD (surrogate over-predicted downforce by 8 %).
- Phase 6: STAR-CCM+ macro + importer written (UNTESTED).
- 88 unit tests (`make test`, ~35 s); `make demo` (~1 min, synthetic, no OpenFOAM).

## UNTESTED (cannot be verified in the cloud workspace)
- Any STAR-CCM+ Java macro or importer (`src/aerosurrogate/starccm/`): only parsing of a synthetic fixture is tested.
- SSH runner and SLURM runner (`src/aerosurrogate/runner/remote.py`): only command/script generation tested.
- Native (non-Docker) OpenFOAM launcher (`configs/openfoam/native_example.yaml`).
- Docker launcher as a non-root user (uid:gid mapping) on your laptop.
- Performance of the pipeline on your laptop.

## Facts confirmed in session 1
- OpenFOAM version: **v2512** (ESI), Docker image digest `sha256:33fb575a...c622f319`.
  ESI apt repo blocked here; Ubuntu package is v1912 (not used).
- Workspace: 4 vCPUs / 15 GB RAM reported, but CPU pressure ~90 %: about 2 effective cores.
- Per-case wall clock (solver only): single element 11.5k-21.5k cells, 2 MPI ranks: 45-70 s;
  1 rank, 2 cases at once: 16-83 s (mean 36 s). 3-element 15k-33k cells: 67-141 s for 3000
  iterations (not converged). Low-Re 3-element (44k cells): 371 s for 4000 iterations.

## Facts to confirm
- Your laptop's cores, RAM, OS, and whether Docker or a native OpenFOAM is available.

## Open questions (for Chaitra)
1. SPEC Q2: laptop specs and OpenFOAM install.
2. SPEC Q3: which STAR-CCM+ results can be used for validation, and may they be published?
3. NEW: actual front-wing sections, chords and angle ranges (see Resume step 2).
4. NEW: the "Airfoil2Vec" link: web search found nothing (docs/literature_note.md).
5. NEW: licence for the repo (README says MIT, to confirm).
Answered in session 1: front wing, 3 elements, draft ranges (Re 2e5-1e6, h/c 0.05-0.5, AoA 0-10,
flaps up to ~35-40 deg); fully turbulent k-omega SST for v1; continue past passed gates;
Gate 0 tolerance CL +/-10 % for h/c >= 0.2.

## Known issues / ideas
- Coarse meshes not mesh-independent (Phase 0). TE thickness changes CL by ~16 % (0.1 % c default).
- Ensemble uncertainty too narrow (2-sigma coverage 0.73-0.90); CM poorly predicted.
- AL acquisition: try integrated-variance reduction, Sobol/variance mixing, larger initial design, GP baseline.
- 2 faces/mesh with non-orthogonality > 70 deg on the upstream ground BL (harmless so far).
- Literature: read Suvanjumrat et al. 2025 (closest prior work) before any novelty statement.

## Budget
Target was ~$19 for the project. Spend was not measurable from inside the session; this
session was long (full build of Phases 0-6). Check your balance before the next session.
