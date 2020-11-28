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

np.random.seed(42)  # for reproducibility

case = 'upperness' 
# case = 'tarbert' 

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
dt_s = 1 
dt_p = 100
days = 2000

def solve(solver='adaptive'):

    # initial conditions
    Q = 795./85  # total injection
    q = np.zeros(grid.shape)
    q[0,0] = Q
    q[-1,-1] = -Q
    s0 = np.ones(grid.shape) * s_wir  # initial water saturation equals s_wir

    mobi_fn = functools.partial(utils.quadratic_mobility, mu_w=mu_w, mu_o=mu_o, s_wir=s_wir, s_oir=s_oir)  # quadratic mobility model
    lamb_fn = functools.partial(utils.lamb_fn, mobi_fn=mobi_fn)  # total mobility function
    f_fn = functools.partial(utils.f_fn, mobi_fn=mobi_fn)  # water fractional flow function
    df_fn = functools.partial(utils.df_fn, mobi_fn=mobi_fn)

    # instantiate solvers
    solverP = ressim.PressureEquation(grid, q=q, k=k, lamb_fn=lamb_fn)
    solverS = ressim.SaturationEquation(grid, q=q, phi=phi, s=s0, f_fn=f_fn, df_fn=df_fn)

    s_compute = []
    p_compute = []

    for index, tp in enumerate( trange( int(days/dt_p) ) ):
        # solve pressure
        solverP.s = solverS.s
        solverP.step()
        solverS.v = solverP.v
        for ts in range( int(dt_p/dt_s) ):
            # solve saturation
            if solver=='adaptive':
                solverS.step_implicit(dt_s)
            if solver=='fixed':
                solverS.step(dt_s)

        p_compute.append(solverP.p)
        s_compute.append(solverS.s)
    
    return s_compute, p_compute

print('adaptive solver...')
before = time()
s_adaptive, p_adaptive = solve(solver='adaptive')
after = time()
time_adaptive = after - before
print('adaptive ',time_adaptive)

print('fixed solver...')
before = time()
s_fixed, p_fixed = solve(solver='fixed')
after = time()
time_fixed = after - before
print('fixed ',time_fixed)

time_benchmark = 28 # approximate value taken from matlab calculations (without plotting)
s_benchmark, p_benchmark = [],[]
for tp in range(int(days/dt_p)):
    p_benchmark.append(np.loadtxt('benchmark_saturations/{}/P_{}.csv'.format(case,int(tp+1)) , delimiter=',' )  )
    s_benchmark.append(np.loadtxt('benchmark_saturations/{}/S_{}.csv'.format(case,int(tp+1)) , delimiter=',' )  )

plt.rcParams.update({'font.size': 8})
points = [[10, 20],[50, 200],[10, 200],[50, 20]]
fig, axs = plt.subplots(2,2,figsize=(8,8))
fig.subplots_adjust(wspace=.3, hspace=.3)
fig.suptitle('{} case solvers:\n fixed (~{} secs), adaptive (~{} secs), benchmark (~{} secs)'.format(case, round(time_fixed), round(time_adaptive), round(time_benchmark) ))
for p, ax in zip(points, axs.ravel()):
    sf, sa, sb = [],[],[]
    for s_f, s_a, s_b in zip(s_fixed, s_adaptive, s_benchmark):
        sf.append(s_f[p[1], p[0]])
        sa.append(s_a[p[1], p[0]])
        sb.append(s_b[p[1], p[0]])
    ax.plot( np.arange(dt_p, days+1, dt_p), sf, '.--' )
    ax.plot( np.arange(dt_p, days+1, dt_p), sa, '.--' )
    ax.plot( np.arange(dt_p, days+1, dt_p), sb, '.--' )
    ax.legend(['fixed', 'adaptive', 'benchmark'])
    ax.set_xlabel('days')
    ax.set_ylabel('water saturation')
    ax.set_title('saturation at [{}, {}]'.format(p[1], p[0]))

fig.savefig('{}_solver_comparison.png'.format(case), bbox_inches=0, pad_inches=0, dpi=300)
# plt.show()

# # visualize
# fig, axs = plt.subplots(2,3,figsize=(6,18))
# for ax in axs.ravel():
#     ax.axis('off')
# fig.suptitle(case+' benchmarking')
# for i, (s_c, s_b, p_c, p_b) in enumerate(zip(s_compute, s_benchmark, p_compute, p_benchmark)):

#     # plot benchmark data
#     im = axs[0,0].imshow(s_b, vmin=0, vmax=1)
#     axs[0,0].set_title('Benchmark S')
#     if i==19:
#         fig.colorbar(im, ax=axs[0,0])

#     im = axs[1,0].imshow(p_b)
#     axs[1,0].set_title('Benchmark P')
#     if i==19:
#         fig.colorbar(im, ax=axs[1,0])


#     # plot computed data
#     im = axs[0,1].imshow(s_c, vmin=0, vmax=1)
#     axs[0,1].set_title('Computed S')
#     if i==19:
#         fig.colorbar(im, ax=axs[0,1])

#     im = axs[1,1].imshow(p_c)
#     axs[1,1].set_title('Computed P')
#     if i==19:
#         fig.colorbar(im, ax=axs[1,1])

#     # plot L2 norm
#     l2_norm = np.sqrt(np.abs(s_b-s_c)**2)
#     im = axs[0,2].imshow( l2_norm )
#     axs[0,2].set_title('mean L2 norm: {}'.format( round( np.mean(l2_norm) , 4) ))
#     if i==19:
#         fig.colorbar(im, ax=axs[0,2])

#     l2_norm = np.sqrt(np.abs(p_b-p_c)**2)
#     im = axs[1,2].imshow( l2_norm )
#     axs[1,2].set_title('mean L2 norm: {}'.format( round( np.mean(l2_norm) ) ))
#     if i==19:
#         fig.colorbar(im, ax=axs[1,2])

#     plt.pause(0.2)

# fig.savefig('{}_benchmarking.png'.format(case), bbox_inches=0, pad_inches=0, dpi=300)