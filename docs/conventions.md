# Conventions

These are stored with every dataset in `metadata.json` (see `aerosurrogate.data.schema`).

- **Axes**: x downstream (freestream direction), y up, ground plane at y = 0. 2D, per unit span.
- **Reference length** c_ref = sum of element chords (geometry is built with c_ref = 1).
  **Reference area** = c_ref x unit span. **Re** = U_inf c_ref / nu. q_inf = 0.5 rho U_inf^2.
- **CL** positive up. A downforce wing has CL < 0; the *downforce coefficient* is -CL.
- **CD** positive downstream. **CM** positive nose-up (OpenFOAM `pitchAxis (0 0 -1)`),
  about the main-element quarter chord (on its chord line).
- **aoa_deg**: main-element incidence; positive = leading edge down (more downforce). The
  assembly rotates about the main-element leading edge.
- **flapN_deg**: deflection of element N relative to the chord of element N-1; positive =
  trailing edge towards the suction side (upwards in the ground frame).
- **gapN**: minimum surface-to-surface distance between elements N-1 and N, / c_ref.
- **overlapN**: along the chord of element N-1, distance from the flap leading edge to the
  trailing-edge midpoint of element N-1, / c_ref; positive = flap leading edge upstream.
- **ride_height_over_chord**: ground to the lowest point of the assembly, / c_ref.
- **Trailing edges** are blunt with thickness `te_thickness` x element chord (default 0.1 %).
- **Moving ground**: belt speed = U_inf (no-slip wall moving with the flow); `ground: slip`
  removes the ground boundary layer.
- **Sources**: `openfoam`, `starccm`, `neuralfoil` (AeroSandbox panel + NeuralFoil
  low-fidelity), `synthetic` (analytic test functions; never physical data).
