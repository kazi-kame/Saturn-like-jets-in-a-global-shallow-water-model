#============
#Imports and module setup
#============
import numpy as np
SI_SOLVER_OPTIONS = {'entry_cutoff': 0.0}

#============
#Grid copy
#============
def grid_copy(field):
    field.change_scales(1)
    return np.array(field['g'], copy=True)

#============
#Set grid
#============
def set_grid(field, values):
    field.change_scales(1)
    values = np.asarray(values)
    target = field['g']
    if target.shape != values.shape:
        raise ValueError(f'Grid shape mismatch: expected {target.shape}, got {values.shape}')
    if not np.all(np.isfinite(values)):
        raise ValueError('Nonfinite field assignment')
    target[...] = values

#============
#Equation namespace
#============
def equation_namespace(u, h, F, g, omega, nu, H, zcross):
    return dict(u=u, h=h, F=F, g=g, omega=omega, nu=nu, H=H, zcross=zcross)
IVP_EQUATIONS = ('dt(u) + nu*lap(lap(u)) + g*grad(h) + 2*omega*zcross(u) = -u@grad(u) + F', 'dt(h) + nu*lap(lap(h)) + H*div(u) = -div((h-H)*u)')
