""" Transient single-phase flow """

from time import time

import numpy as np
import scipy.optimize

import ressim

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['image.cmap'] = 'jet'
import matplotlib.pyplot as plt

from spatial_expcov import batch_generate

np.random.seed(42)  # for reproducibility
nx,ny=64,64
grid = ressim.Grid(nx=nx, ny=ny, lx=1.0, ly=1.0)  # unit square, 64x64 grid
# k = batch_generate(nx, nx, 0.1, 1.0, 1.0, 1.0, 1)
# k=np.exp(k[0])

k = np.exp(np.load('perm.npy').reshape(grid.shape))  # load log-permeability, convert to absolute with exp()
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
for i in range(nstep):
    before = time()
    # solve saturation
    solverS.step_explicit(dt,0.0,0.0)
    # solverS.step(dt)
    # solverS.step_mrst(dt)
    after = time()
    print('[{}/{}]: this loop took {} secs'.format(i+1, nstep, after - before))
    s_list.append(solverS.s)

# # visualize
# fig, axs = plt.subplots(5,5, figsize=(8,8))
# fig.subplots_adjust(wspace=.1, hspace=.1, left=0, right=1, bottom=0, top=1)
# for ax, s in zip(axs.ravel(), s_list):
#     ax.imshow(s)
#     ax.axis('off')
# fig.savefig('saturations.png', bbox_inches=0, pad_inches=0)

# visualize
fig, axs = plt.subplots(5,5, figsize=(8,8))
fig.subplots_adjust(wspace=.1, hspace=.1, left=0.15, right=0.85, bottom=0.1, top=0.9)
for ax, s in zip(axs.ravel(), s_list):
    im = ax.imshow(s, vmin=0, vmax=1)
    ax.axis('off')
cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
fig.savefig('saturations.png', bbox_inches=0, pad_inches=0)
