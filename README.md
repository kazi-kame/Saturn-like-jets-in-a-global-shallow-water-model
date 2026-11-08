# Saturn shallow-water dynamics

Global shallow-water simulation and linear stability analysis of prescribed
Saturn-like jets.

## Code

- `saturn_poly/`: spherical equations, balanced jets, spectral operators,
  eigenmode comparisons and response diagnostics.
- `saturn_poly_l1/src/`: Dedalus solver, forcing, field output and checkpoints.
- `saturn_poly_l1/scripts/`: experiment preparation and saved-output analysis.
- `tests/`, `saturn_poly_l1/tests/`: analytical and regression checks.

## Configurations

- `configs/stability_manufactured.json`: polynomial-jet numerical test.
- `configs/reference_convergence.json`: nonrotating rest-state reference.
- `configs/stability_refinement.json`: intermediate spectral refinement.
- `configs/stability_wide.json`: wider mode range and higher truncations.
- `configs/reference_source_config.json`: nonlinear input used to construct the smooth jets.
- `saturn_poly_l1/configs/exploratory_proxy.json`: smaller example grid.
- `saturn_poly_l1/configs/saturn_working_1024x512.json`: six-hour working configuration.

Names ending in `_m`, `_s`, or `_rad` indicate SI units. Each `smooth_jets` row
contains the amplitude, center and width of W(mu), where U = cos(latitude) W(mu)
and mu = sin(latitude). These are prescribed profiles, not fitted wind measurements.

## Setup

For analysis:

```bash
python -m pip install -r requirements.txt
```

For the nonlinear solver, including MPI/FFTW dependencies:

```bash
conda env create -f environment.yml
conda activate saturn-shallow-water
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

`requirements-solver.txt` is an alternative for an existing MPI/FFTW toolchain.

## Check the code

From the repository root:

```bash
python -m unittest discover -s tests -v
python -m pytest -q saturn_poly_l1/tests
python -m saturn_poly.stability --config configs/stability_wide.json --validate-only
python saturn_poly_l1/src/run_l1.py --check-environment
```

Tests include analytical operators and static solver builds; they do not evolve
the fluid. Inputs and code hashes are recorded for newly created runs. Checkpoints
from the original research workspace are not compatible with this cleaned copy.

The nonlinear experiments characterize short-time forced response. The linear
analysis uses a separately maintained background. Neither alone establishes
long-term polygon formation. Manuscripts, research notes and simulation outputs
are not included.
