# aero-surrogate. All targets run from the repo root.
PY ?= .venv/bin/python
CONFIG ?= configs/openfoam/docker_v2512.yaml

.PHONY: install test test-openfoam demo demo-cfd phase0 gate1 lock

install:
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

lock:
	$(PY) -m pip freeze --exclude-editable > requirements-lock.txt

test:            ## unit tests (no OpenFOAM needed)
	$(PY) -m pytest -q

test-openfoam:   ## tests that run OpenFOAM through the configured launcher
	$(PY) -m pytest -q -m openfoam

demo:            ## synthetic end-to-end demo: no OpenFOAM, a few minutes on a laptop CPU
	$(PY) -m aerosurrogate demo --out runs/demo

demo-cfd:        ## one coarse 3-element CFD case end to end, with HTML report
	$(PY) -m aerosurrogate case --launcher $(CONFIG) --case configs/case_coarse.yaml --params configs/demo_case_params.yaml --out runs/demo_cfd

phase0:          ## Phase 0 single-element ground-effect sweep
	$(PY) -m aerosurrogate phase0 --launcher $(CONFIG) --out runs/phase0

gate1:           ## Gate 1: 20 random 3-element cases must pass checkMesh
	$(PY) -m aerosurrogate meshcheck --launcher $(CONFIG) --n 20 --out runs/gate1_meshcheck
