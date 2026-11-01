# Aero-map export format (`aero-map/1`)

Written by `aerosurrogate optimise` (`aerosurrogate.optimise.export.export_aero_map`) for a
later lap-time simulator. Two files with the same stem:

## `<name>.csv`: one row per grid point

| column | meaning |
| --- | --- |
| grid axes (e.g. `ride_height_over_chord`, `aoa_deg`, `re`) | the swept inputs; names listed in the JSON `grid_axes` |
| `cl`, `cd`, `cm` | ensemble-mean surrogate prediction; 2D, per unit span, conventions below |
| `cl_std`, `cd_std`, `cm_std` | ensemble standard deviation (model/epistemic only; it does **not** include CFD modelling error) |
| `downforce_coeff` | `-cl` |
| `extrapolated` | `True` if the point is outside the surrogate's training ranges: do not trust it |

## `<name>.json`: metadata (read it; never assume)

- `format`: `"aero-map/1"`
- `fixed_design`: all wing parameters held constant for the map
- `training_ranges`: `{parameter: [low, high, scale]}`
- `surrogate_inputs`, `source` (which surrogate/data produced the map), `generator`, `created_utc`
- `conventions`: full copy of the dataset metadata (axes, signs, c_ref, reference area,
  moment reference, parameter definitions)
- `use_in_3d`: a reminder that these are **2D sectional** coefficients

## Conventions (summary of docs/conventions.md)

- `cl` positive up (downforce wing: `cl < 0`), `cd` positive downstream, `cm` positive nose-up
  about the main-element quarter chord.
- Coefficients are based on c_ref = sum of element chords and q_inf = 0.5 rho U_inf^2, per unit span.
- `ride_height_over_chord` = ground to lowest point / c_ref; `re` = U_inf c_ref / nu.

## Using it in a lap-time model

Per-unit-span 2D coefficients are not car coefficients. You need at least: wing span and
area, 3D/endplate corrections (or calibration against your STAR-CCM+ 3D runs), and a
mapping from car state (front ride height, pitch) to `ride_height_over_chord` and `aoa_deg`.
Interpolate inside the grid only; refuse rows with `extrapolated = True`.
