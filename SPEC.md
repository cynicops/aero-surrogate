# SPEC: aero-surrogate (ML + OpenFOAM aero toolkit that improves itself)

## One-line goal
An open-source Python toolkit that automates 2D OpenFOAM cases for multi-element wings in ground effect and trains an ensemble neural surrogate that chooses its own next CFD cases (active learning).

## Why this is worth building
- The automation pipeline generates the surrogate's training data; the surrogate's uncertainty decides which case to run next.
- Existing tools cover plain airfoils well (NeuralFoil: single airfoils, Re 10^2 to 10^10, 8x to 1,000x faster than XFoil at 0.37% to 2.0% mean relative drag error versus XFoil). Its abstract does not mention multi-element wings or ground effect.
- Active multi-fidelity surrogate methods exist for single-element airfoil optimisation (XFOIL + RANS; RANS needed for 9.5% to 14.8% of individuals in the reported cases). The defensible niche is low-Re, multi-element, ground-effect FSAE wings with an open pipeline.

## Scope
In scope:
- 2D, 2 to 3 element wing (main plane + one or two flaps) near a moving ground.
- Parameters: AoA, flap angle(s), gap(s), overlap(s), ride height, Reynolds number. Ranges: TBD (open question).
- Outputs: CL, CD, CM; optionally Cp distribution (reduced with PCA).
- Automated reports: Cp, force coefficients, y+ histogram, mesh quality, mesh-convergence and GCI.
- Ensemble NN surrogate, active-learning loop, optimiser on the surrogate, export of coefficients for a later lap-time simulator.

Out of scope (for now):
- 3D full-car or full-wing CFD, unsteady or LES runs.
- Running STAR-CCM+ (adapter code only, UNTESTED).
- Building a GUI.

## Known risks (carry these into every report)
1. RANS accuracy on separated multi-element flow near a moving ground is uncertain; a surrogate inherits CFD error.
2. Validation data is required: published ground-effect wing data (verify the exact references) and my own STAR-CCM+ runs.
3. Compute: the training campaign runs on my laptop, specs unknown. Plan for per-case wall-clock time measured in Phase 0.
4. Novelty is unproven until the literature note is written.

## Data schema (shared by OpenFOAM and STAR-CCM+ results)
One row per case in a Parquet or CSV file plus one folder per case. Required columns:
`case_id, solver, solver_version, turbulence_model, mesh_cells, y_plus_max, geometry_params (JSON), aoa_deg, re, ride_height_over_chord, cl, cd, cm, converged (bool), residual_final, wall_clock_s, source (openfoam|starccm|neuralfoil|synthetic)`.
Units and reference length/area are stored with the dataset metadata, never assumed.

## Success criteria
- A validated 2D surrogate with an uncertainty estimate.
- A measured reduction in CFD runs needed versus space-filling sampling at equal accuracy.
- A reproducible repo another team could run (`make test`, `make demo`).
- A Phase 0 note stating honestly what was and was not verified.

## Open questions (ask me in the first session)
1. Which wing: front or rear, how many elements, and what Re and ride-height ranges?
2. Laptop specs (cores, RAM) and OS, and whether OpenFOAM is installed there (which version).
3. Which STAR-CCM+ results can I use for validation, and may they be published?
4. Is a transition model needed, or is fully turbulent RANS acceptable for the first version?
