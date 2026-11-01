# Phase 0 report: feasibility spike (session 1, 1 Nov 2026)

**Gate 0 verdict: PARTIAL.** The OpenFOAM pipeline runs, cases converge iteratively,
runtimes are measured and the literature note exists. **Validation against published data
was not possible** (sources blocked by the workspace network), and the coarse meshes are
**not mesh-converged** for this separated ground-effect flow (CL varies by up to ~35 %
between meshes). No fallback to an XFOIL/VLM-only surrogate is proposed: CFD is feasible;
what is missing is validation data and a production-quality mesh.

Data: `docs/phase0/phase0_cases.csv` (schema rows), `docs/phase0/phase0_summary.json`,
figure `docs/phase0/phase0_ride_height_sweep.png`. Per-case HTML reports were generated in
`runs/phase0/dataset/cases/*/report.html` (not committed: large, regenerable with `make phase0`).

## 1. OpenFOAM install

| Route | Result |
| --- | --- |
| ESI Debian repository (dl.openfoam.com) | **Blocked** by the workspace network policy (HTTP 403 at the proxy). |
| Ubuntu package | Available, but it is v1912 (2019); not used. |
| Official ESI Docker image `opencfd/openfoam-default:2512` | **Used.** OpenFOAM **v2512**, build `_87ed40d2-20251219`, Ubuntu 24.04 base, image digest `sha256:33fb575aa9980d2bc42fd58c75ae698c489293ba30c991380fe3f899c622f319`. The first pull hit a Docker Hub rate limit (429) and succeeded on retry. |

The toolkit runs OpenFOAM through a launcher config (`configs/openfoam/*.yaml`): `docker`
(used here) or `native` (for your laptop; UNTESTED here).

## 2. Single-element case in ground effect

Setup (all in `src/aerosurrogate/phase0.py`):

- Section: **stand-in** `ls417` (LS(1)-0417 = GA(W)-1) from AeroSandbox's database, TE thickness
  0.1 % c. Zerihan & Zhang used a *modified* GA(W)-1 / Tyrrell 026 section whose coordinates we
  could not obtain, so this is **not** their wing.
- AoA 4 deg (positive = leading edge down), Re = 5e5 on chord (a placeholder, not the paper's Re),
  moving ground (belt speed = U_inf), k-omega SST fully turbulent, steady simpleFoam (SIMPLE),
  wall functions (nutUSpaldingWallFunction), y+ about 25-57 on the wing.
- Coarse 2D hybrid mesh: ~11.5k-21.5k cells, 8 boundary-layer layers, domain 5c upstream,
  10c downstream, 5c high.

### Results (coarse mesh)

| h/c | status | CL | -CL (downforce) | CD | CM | iterations | solver wall clock (s) | cells |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.10 | **not converged** (limit cycle) | -0.865 | 0.865 | 0.0782 | 0.151 | 3000 | 153.5 | 11,513 |
| 0.15 | ok | -1.018 | 1.018 | 0.0658 | 0.156 | 707 | 46.6 | 11,483 |
| 0.20 | ok | -1.161 | 1.161 | 0.0536 | 0.163 | 640 | 44.5 | 12,080 |
| 0.30 | ok | -1.311 | 1.311 | 0.0345 | 0.172 | 720 | 54.4 | 13,228 |
| 0.50 | ok | -1.159 | 1.159 | 0.0242 | 0.148 | 819 | 60.6 | 15,666 |
| 1.00 | ok | -0.990 | 0.990 | 0.0199 | 0.124 | 803 | 68.7 | 21,482 |

Trend: downforce rises as the wing approaches the ground, peaks near h/c = 0.3 and then falls,
while drag rises steeply. Surface pressures show a trailing-edge separation plateau on the
suction side at h/c = 0.2 (x/c > ~0.75). This is the qualitative "force reduction" behaviour
Zerihan & Zhang describe (they report it below h/c ~ 0.1 for their wing and incidence; ours
starts higher, which may be the different section/incidence or the CFD: **unknown until validated**).

### Runtime (2 MPI ranks)

- Single element, 11.5k-21.5k cells: **45-70 s** of solver wall clock per converged case
  (640-820 iterations), plus ~5-10 s for meshing, conversion and checks.
- Caveat: these were measured with two cases running at once, on a VM whose CPU-pressure
  stall counter read ~90 % (the host gave us about 2 effective cores, though `nproc` = 4).
  Treat them as upper bounds. Your laptop timings are still to be measured.

### Mesh study (h/c = 0.2; first-cell height and y+ held fixed)

| cells | CL | CD | CM | solver wall clock (s) |
| --- | --- | --- | --- | --- |
| 12,080 | -1.161 | 0.0536 | 0.163 | 44.5 |
| 24,982 | -1.325 | 0.0572 | 0.191 | 124.9 |
| 52,326 | -0.984 | 0.0494 | 0.133 | 184.8 |

Convergence is **oscillatory**, so the GCI (Celik et al. 2008) is not a reliable error bar;
formally GCI_fine = 41 % for CL. Each mesh is iteratively converged (residuals 2e-5 to 9e-5).
The separation extent changes with the mesh. **Conclusion: these coarse wall-function meshes
are pipeline smoke tests, not data you can design with.** (A first attempt that also refined
the wall cell pushed y+ into the buffer layer; that was fixed so refinement keeps y+.)

Recommendation for the laptop campaign (to test, not yet verified): low-Re mesh (y+ ~ 1,
20-30 layers), second-order convection for k and omega (`bounded Gauss limitedLinear 1`),
and a 3-level GCI study at two ride heights before generating training data.

### Trailing-edge thickness sensitivity

With a 0.4 % c blunt trailing edge, the h/c = 0.2 case never converged: it oscillated
(CL about -1.00 +/- 0.01, residuals stuck at 1e-2), consistent with vortex shedding that
steady RANS cannot settle. With 0.1 % c it converged to CL = -1.161. **TE thickness alone
moved CL by ~16 %.** Default is now 0.1 % of each element's chord; this is a modelling
choice to document in any paper.

## 3. Sanity checks (NOT validation)

Near-freestream case (slip ground at h/c = 5, 68k cells) vs NeuralFoil 0.3.3 on the same
section, alpha = 4 deg, Re = 5e5 (NeuralFoil is in the lift-up frame, so compare -CL and -CM):

| | CL (lift-up) | CD | CM (nose-up +) | note |
| --- | --- | --- | --- | --- |
| OpenFOAM (coarse, fully turbulent) | 0.893 | 0.0178 | -0.110 | |
| NeuralFoil, free transition (n_crit 9) | 0.907 | 0.0098 | -0.107 | confidence 0.99 |
| NeuralFoil, transition forced at LE | 0.775 | 0.0164 | -0.085 | confidence **0.27** |

The magnitudes are consistent. CFD drag sits close to NeuralFoil's fully-turbulent value, as
expected for a fully turbulent model. This checks signs, references and units of the
pipeline; it says nothing about ground-effect accuracy.

The inviscid panel method with ground images (AeroSandbox, used as our low-fidelity baseline)
over-predicts downforce strongly near the ground (h/c = 0.1: -CL = 4.66 vs 0.87 in CFD),
because it has no boundary layer and cannot separate.

## 4. Validation against published data: PENDING

- Reference identified: Zerihan & Zhang (2000), J. Aircraft 37(6) 1058-1064, and Zerihan's PhD
  thesis (Southampton). Neither could be opened (network policy). No reference values were
  typed in from memory.
- Agreed tolerance (session 1): **CL within +/-10 % for h/c >= 0.2**; below that, report the
  error but do not gate on it.
- To finish Gate 0: put the paper's section coordinates in a `.dat` file, put its
  CL(h/c) into `docs/validation/reference_lift.csv` (format in `docs/validation/README.md`),
  set `SECTION`, `AOA` and `RE` in `phase0.py` to the experiment's, and run `make phase0`. The
  comparison table is written to `runs/phase0/phase0_summary.json` automatically.
- Given the mesh study above, expect to need the production mesh before the +/-10 % test is meaningful.

## 5. What was not verified

- Agreement with any experiment (no data).
- Mesh independence (shown **not** to hold on coarse meshes).
- Runtime on 2 *uncontended* cores, and anything about your laptop.
- That `ls417` behaves like the Zerihan & Zhang wing.
- The NeuralFoil forced-transition numbers (low confidence 0.27).
