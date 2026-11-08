#============
#Imports and module setup
#============
from pathlib import Path
import sys
import numpy as np
import pytest
d3 = pytest.importorskip('dedalus.public')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from grid_io import grid_copy, set_grid, SI_SOLVER_OPTIONS
from run_l1 import main

#============
#Test saturn si poisson recovers degree one harmonic
#============
def test_saturn_si_poisson_recovers_degree_one_harmonic():
    radius = 60268000.0
    g = 11.19
    coords = d3.S2Coordinates('phi', 'theta')
    dist = d3.Distributor(coords, dtype=np.float64)
    basis = d3.SphereBasis(coords, (16, 8), radius=radius, dtype=np.float64)
    h = dist.Field(name='h', bases=basis)
    c = dist.Field(name='c')
    rhs = dist.Field(name='rhs', bases=basis)
    phi, theta = dist.local_grids(basis, scales=1)
    exact = np.cos(theta) + np.zeros_like(phi)
    set_grid(rhs, -2 * g / radius ** 2 * exact)
    problem = d3.LBVP([h, c], namespace=dict(h=h, c=c, rhs=rhs, g=g))
    problem.add_equation('g*lap(h)+c=rhs')
    problem.add_equation('ave(h)=0')
    with pytest.raises(RuntimeError, match='exactly singular'):
        problem.build_solver(entry_cutoff=1e-12).solve()
    problem.build_solver(**SI_SOLVER_OPTIONS).solve()
    np.testing.assert_allclose(grid_copy(h), exact, rtol=1e-12, atol=1e-12)

#============
#Test actual build only never steps
#============
def test_actual_build_only_never_steps(tmp_path, monkeypatch):
    from dedalus.core.solvers import InitialValueSolver
    original_init = InitialValueSolver.__init__

    #============
    #Inspect options
    #============
    def inspect_options(self, *args, **kwargs):
        assert kwargs['entry_cutoff'] == 0.0
        original_init(self, *args, **kwargs)

    #============
    #Forbidden step
    #============
    def forbidden_step(*args, **kwargs):
        raise AssertionError('Build-only must never evolve the fluid')
    monkeypatch.setattr(InitialValueSolver, '__init__', inspect_options)
    monkeypatch.setattr(InitialValueSolver, 'step', forbidden_step)
    output = tmp_path / 'build'
    assert main(['--config', str(Path(__file__).resolve().parents[1] / 'configs/exploratory_proxy.json'), '--build-only', '--out', str(output)]) == 0
    import json
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['status'] == 'built_not_evolved'
    assert manifest['steps_evolved'] == 0
    assert (output / 'states/step00000000.nc').is_file()
