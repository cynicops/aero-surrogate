# Literature note (Phase 0)

Status: **draft from abstracts and search-result summaries only.** The workspace network
blocks arxiv.org, sciencedirect.com, core.ac.uk and eprints.soton.ac.uk, so **no full
text was opened** in session 1 (1 Nov 2026). Every number below is copied from an
abstract, a repository record or a secondary summary, and is marked as such. Re-check
each one against the full text before quoting it in a paper.

This note does not claim novelty. It lists what the closest work appears to do, so that
the gap can be stated after the full texts are read.

## 1. NeuralFoil (baseline we use, not rebuild)

- Sharpe, P. and Hansman, R. J., "NeuralFoil: An Airfoil Aerodynamics Analysis Tool Using
  Physics-Informed Machine Learning", arXiv:2503.16323 (March 2025).
- From the abstract: open-source Python tool, a hybrid of physics-informed machine learning
  and analytical models; 18-dimensional airfoil shape space, 360 deg angle of attack,
  Re 10^2 to 10^10, up to transonic drag rise; 8x to 1,000x faster than XFoil; mean
  relative drag error vs XFoil 0.37 % (simple cases) to 2.0 % (test set with post-stall
  and transitional cases); includes a surrogate uncertainty method. Secondary summaries say
  the training data came from XFoil.
- Scope: single, isolated airfoils. The abstract does not mention multi-element wings or
  ground effect, and the Python API (`get_aero_from_airfoil`) takes one airfoil.
- What we measured ourselves (workspace, neuralfoil 0.3.3): for the Phase 0 section
  (`ls417`, TE 0.1 % c, alpha 4 deg, Re 5e5) its `analysis_confidence` is 0.99 with free
  transition but only 0.27 with transition forced at the leading edge. So "fully
  turbulent NeuralFoil" is outside its comfortable range. See docs/phase0_report.md.
- How we use it: profile drag per element in the low-fidelity baseline
  (`aerosurrogate.baseline.lowfi`), and as the "NeuralFoil/AeroSandbox-only" baseline the
  surrogate must beat (Gate 3).

## 2. F1 front-wing PINN

- Shah, N., "Computational Fluid Dynamics Optimization of F1 Front Wing using Physics
  Informed Neural Networks", arXiv:2509.01963 (September 2025). Single-author preprint;
  no sign of peer review in the search results.
- From the abstract / summaries: CFD data generated with SimScale for an F1 front wing;
  feed-forward network (summaries say 12 inputs -> 128 -> 64 -> 32 -> 2 outputs) with a
  Navier-Stokes-based physics term in the loss; reported R^2 of 0.968 (CD) and 0.981 (CL);
  the author notes the dataset and design variations are limited.
- Not confirmed: whether ground effect is modelled, how the physics term is evaluated
  from force data only, and how train/test were split.
- Relevance: shows interest in ML surrogates for front wings; small-data regime similar to
  ours. Its reported R^2 is not comparable to our metrics until the split is known.

## 3. Active multi-fidelity surrogate for airfoil optimisation

- Robledo et al., "Optimization-Embedded Active Multi-Fidelity Surrogate Learning for
  Multi-Condition Airfoil Shape Optimization", arXiv:2603.17057 (March 2026).
- From the abstract / aggregators: XFOIL evaluations act as cheap low-fidelity features;
  sparse RANS runs are requested when predictive uncertainty is high; CST parametrisation
  with 12 parameters; two operating points (cruise, take-off) at Re = 6e6. Aggregator pages
  (CatalyzeX, newx.sg) report RANS needed for about 14.78 % (cruise) and 9.5 % (take-off)
  of evaluated individuals, and gains of 41.05 % (cruise efficiency) and 20.75 % (take-off
  lift) vs the best first-generation individual. These match the SPEC's "9.5 % to 14.8 %"
  but come from secondary pages.
- Scope: single-element airfoils, high Re, no ground. Closest methodological template for
  our Phase 4 (uncertainty-driven CFD selection with a cheap-model feature).

## 4. AI-assisted multi-element FSAE wing optimisation (2025, ScienceDirect), "not opened" paper 1

- Suvanjumrat et al., "AI-assisted CFD optimisation of multi-element wing angle of attack for
  enhanced formula SAE aerodynamic performance", International Journal of Thermofluids
  30 (2025) 101440, DOI 10.1016/j.ijft.2025.101440. Found via the Mahidol University
  repository record, not the ScienceDirect page.
- From the repository abstract: a multi-level framework combining CFD and AI for Formula SAE;
  a GA-ANN stage reported +14.8 % front-wing downforce and +28.4 % rear-wing downforce, with
  reduced drag.
- **Status: NOT READ (blocked).** This is the closest prior work to our niche (multi-element
  FSAE wings + CFD + neural surrogate). Until it is read we do not know: 2D or 3D, whether
  ground effect / ride height is a variable, which parameters besides angle of attack,
  how many CFD runs, whether the pipeline or data are open, or whether sampling is
  adaptive. **Our novelty statement depends on this paper.**

## 5. "Airfoil2Vec" downforce dataset, "not opened" paper 2

- **Could not be found.** Web search for "Airfoil2Vec downforce dataset" returned no matching
  paper, code or dataset. Need the original link or citation.
- Closest results (not the same thing):
  - Agarwal et al., "A Comprehensive Dataset of the Aerodynamic and Geometric Coefficients of
    Airfoils in the Public Domain", *Data* 9(5):64 (2024): ~2,900 airfoils at Re = 1e5,
    OpenFOAM + CST, MIT-licensed on GitHub (kanakaero/Dataset-of-Aerodynamic-and-Geometric-Coefficients-of-Airfoils).
    Listed on researcher.manipal.edu, so possibly from your own university: worth asking.
    Single airfoils, freestream, no ground effect.
  - A NASA-style "airfoil CFD 9k shapes, 2 AoAs" RANS dataset on data.gov (HAM2D).

## 6. Validation data for Gate 0 (identified, not obtained)

- Zerihan, J. and Zhang, X., "Aerodynamics of a single element wing in ground effect",
  Journal of Aircraft 37(6):1058-1064 (2000). Repository record: eprints.soton.ac.uk/21339.
  From the abstract: highly cambered single-element wing, suction surface nearest the ground,
  ride height and incidence varied; downforce reduces below about h/c = 0.1 because of
  trailing-edge separation. Search results describe the profile as a modified NASA GA(W)-1
  section and the model as an 80 %-scale 1998 Tyrrell F1 front wing (secondary sources).
- Zerihan, J., PhD thesis "An Investigation into the Aerodynamics of Wings in Ground Effect",
  University of Southampton (eprints.soton.ac.uk/426058): contains force and pressure
  chapters; most likely source of tabulated values and coordinates.
- Zhang, X. and Zerihan, J., AIAA paper 2001-0423, comparison of experiments and computation
  for the single-element wing (DOI 10.2514/6.2001-423 per search result).
- A double-element follow-up by the same group is likely the best multi-element ground-effect
  validation case. **Citation not verified in this session; look it up.**

## Working gap hypothesis (to test, not to claim)

Based only on abstracts: tools for single airfoils are mature (NeuralFoil); active
multi-fidelity learning is shown for single-element, high-Re airfoils without ground; FSAE
multi-element CFD+ANN optimisation exists (item 4), with unknown openness and scope. The
candidate niche is still the one in SPEC.md: an **open, reproducible 2D pipeline for
low-Re, 2-3 element wings in ground effect, where the surrogate's uncertainty picks the next
CFD case**, plus an honest comparison against NeuralFoil/AeroSandbox-only estimates.
Item 4 must be read before this is written anywhere.

## Actions

1. Allow the domains above (or upload PDFs) and re-run this note against full texts.
2. Get the Airfoil2Vec link from whoever mentioned it.
3. Extract Zerihan & Zhang geometry + CL(h/c) into `docs/validation/reference_lift.csv`
   (format in docs/phase0_report.md) and re-run `make phase0`.
