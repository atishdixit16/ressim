""" Transient two-phase (oil-water) flow """

from time import time, sleep

import numpy as np
import functools
from tqdm import tqdm, trange

import ressim
import utils

import matplotlib
# matplotlib.use('Agg')
matplotlib.rcParams['image.cmap'] = 'jet'
import matplotlib.pyplot as plt

# case = 'upperness' 
case = 'tarbert' 

# domain properties
nx = 60
ny = 220
lx = 60*0.3048*20
ly = 220*0.3048*10
grid = ressim.Grid(nx=nx, ny=ny, lx=lx, ly=ly)
k=np.loadtxt(case+'_perm.csv', delimiter=',')

phi = np.loadtxt(case+'_por.csv', delimiter=',')
s_wir = 0.2
s_oir = 0.2

# fluid properties
mu_w = 3e-4
mu_o = 3e-3
mobility='quadratic'

# time steps
dt_s = 5
dt_p = 100
days = 2000

# initial conditions
Q = 795./85  # total injection
q = np.zeros(grid.shape)
q[0,0] = Q
q[-1,-1] = -Q
s = np.zeros(grid.shape)

np.random.seed(42)  # for reproducibility
s0 = np.ones(grid.shape) * s_wir  # initial water saturation equals s_wir

mobi_fn = functools.partial(utils.quadratic_mobility, mu_w=mu_w, mu_o=mu_o, s_wir=s_wir, s_oir=s_oir)  # quadratic mobility model
lamb_fn = functools.partial(utils.lamb_fn, mobi_fn=mobi_fn)  # total mobility function
f_fn = functools.partial(utils.f_fn, mobi_fn=mobi_fn)  # water fractional flow function
df_fn = functools.partial(utils.df_fn, mobi_fn=mobi_fn)

# instantiate solvers
solverP = ressim.PressureEquation(grid, q=q, k=k, lamb_fn=lamb_fn)
solverS = ressim.SaturationEquation(grid, q=q, phi=phi, s=s0, f_fn=f_fn, df_fn=df_fn)

s_list = []
for tp in trange( int(days/dt_p) ):
    # solve pressure
    solverP.s = solverS.s
    solverP.step()
    for ts in range( int(dt_p/dt_s) ):
        # solve saturation
        solverS.v = solverP.v
        solverS.step_implicit(dt_s)
    s_list.append(solverS.s)


# visualize
fig, axs = plt.subplots(1,1,figsize=(1.5,3))
axs.axis('off')
for s in s_list:
    axs.imshow(s, vmin=0, vmax=1)
    plt.pause(0.2)
plt.show()