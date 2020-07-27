""" Transient single-phase flow """

from time import time
import numpy as np
import ressim
from spatial_expcov import batch_generate

nx,ny=128,128

def test_case_scipy():

    np.random.seed(42)  # for reproducibility

    grid = ressim.Grid(nx=nx, ny=ny, lx=1.0, ly=1.0)  # unit square, 128x128 grid
    k = batch_generate(nx, nx, 0.1, 1.0, 1.0, 1.0, 1)
    k=np.exp(k[0])
    q = np.zeros(grid.shape); q[0,0]=1; q[-1,-1]=-1  # source term: corner-to-corner flow (a.k.a. quarter-five spot)

    phi = np.ones(grid.shape)*0.2  # uniform porosity
    s0 = np.zeros(grid.shape)  # initial water saturation
    dt = 1e-2  # timestep

    def f_fn(s): return s  # water fractional flow; for single-phase flow, f(s) = s

    # (Optional) derivative of water fractional flow; in single-phase flow, f'(s) = 1
    # This is to compute the jacobian of the residual to accelerate the
    # saturation solver. If not provided, the jacobian is approximated in the
    # solver.
    def df_fn(s): return np.ones(len(s))

    # instantiate solvers
    solverP = ressim.PressureEquation(grid, q=q, k=k)
    solverS = ressim.SaturationEquation(grid, q=q, phi=phi, s=s0, f_fn=f_fn, df_fn=df_fn)

    # solve for 25 timesteps
    nstep=25
    # solve pressure; in single-phase, we only need to solve it once
    solverP.step()
    solverS.v = solverP.v
    s_list = []
    t_list = []
    for _ in range(nstep):
        before = time()
        # solve saturation
        solverS.step(dt)
        after = time()
        s_list.append(solverS.s)
        t_list.append(after-before)
    return s_list, t_list

def test_case_implicit():

    np.random.seed(42)  # for reproducibility

    grid = ressim.Grid(nx=nx, ny=ny, lx=1.0, ly=1.0)  # unit square, 128x128 grid
    k = batch_generate(nx, nx, 0.1, 1.0, 1.0, 1.0, 1)
    k=np.exp(k[0])
    q = np.zeros(grid.shape); q[0,0]=1; q[-1,-1]=-1  # source term: corner-to-corner flow (a.k.a. quarter-five spot)

    phi = np.ones(grid.shape)*0.2  # uniform porosity
    s0 = np.zeros(grid.shape)  # initial water saturation
    dt = 1e-2  # timestep

    def f_fn(s): return s  # water fractional flow; for single-phase flow, f(s) = s

    # (Optional) derivative of water fractional flow; in single-phase flow, f'(s) = 1
    # This is to compute the jacobian of the residual to accelerate the
    # saturation solver. If not provided, the jacobian is approximated in the
    # solver.
    def df_fn(s): return np.ones(len(s))

    # instantiate solvers
    solverP = ressim.PressureEquation(grid, q=q, k=k)
    solverS = ressim.SaturationEquation(grid, q=q, phi=phi, s=s0, f_fn=f_fn, df_fn=df_fn)

    # solve for 25 timesteps
    nstep=25
    # solve pressure; in single-phase, we only need to solve it once
    solverP.step()
    solverS.v = solverP.v
    s_list = []
    t_list = []
    for _ in range(nstep):
        before = time()
        # solve saturation
        solverS.step_mrst(dt)
        after = time()
        s_list.append(solverS.s)
        t_list.append(after-before)
    return s_list, t_list

def test_case_explicit():

    np.random.seed(42)  # for reproducibility

    grid = ressim.Grid(nx=nx, ny=ny, lx=1.0, ly=1.0)  # unit square, 128x128 grid
    k = batch_generate(nx, nx, 0.1, 1.0, 1.0, 1.0, 1)
    k=np.exp(k[0])
    q = np.zeros(grid.shape); q[0,0]=1; q[-1,-1]=-1  # source term: corner-to-corner flow (a.k.a. quarter-five spot)

    phi = np.ones(grid.shape)*0.2  # uniform porosity
    s0 = np.zeros(grid.shape)  # initial water saturation
    dt = 1e-2  # timestep

    def f_fn(s): return s  # water fractional flow; for single-phase flow, f(s) = s

    # (Optional) derivative of water fractional flow; in single-phase flow, f'(s) = 1
    # This is to compute the jacobian of the residual to accelerate the
    # saturation solver. If not provided, the jacobian is approximated in the
    # solver.
    def df_fn(s): return np.ones(len(s))

    # instantiate solvers
    solverP = ressim.PressureEquation(grid, q=q, k=k)
    solverS = ressim.SaturationEquation(grid, q=q, phi=phi, s=s0, f_fn=f_fn, df_fn=df_fn)

    # solve for 25 timesteps
    nstep=25
    # solve pressure; in single-phase, we only need to solve it once
    solverP.step()
    solverS.v = solverP.v
    s_list = []
    t_list = []
    for _ in range(nstep):
        before = time()
        # solve saturation
        solverS.step_explicit(dt,0,0)
        after = time()
        s_list.append(solverS.s)
        t_list.append(after-before)
    return s_list, t_list