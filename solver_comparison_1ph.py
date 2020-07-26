
import numpy as np
import matplotlib.pyplot as plt

from test_cases_1ph import test_case_scipy, test_case_implicit, test_case_explicit

s_explicit, t_explicit = test_case_explicit()
s_scipy, t_scipy = test_case_scipy()
s_implicit, t_implicit = test_case_implicit()

s_explicit = np.array(s_explicit)
t_explicit = np.array(t_explicit)
s_scipy = np.array(s_scipy)
t_scipy = np.array(t_scipy)
s_implicit = np.array(s_implicit)
t_implicit = np.array(t_implicit)


# visualize

# solution comparson

fig, axs = plt.subplots(2,2, figsize=(8,8))
fig.subplots_adjust(wspace=0.5, hspace=0.5, left=0.15, right=0.85, bottom=0.1, top=0.9)

# saturations at point [16,16]
indices = [1,16,32,48]
fig.suptitle('Comparison of Saturation Equation Solvers', fontsize=16)
for ax, ind in zip(axs.ravel(), indices):
    ax.plot(s_scipy[:,ind,ind], 'o--')
    ax.plot(s_implicit[:,ind,ind], '.--')
    ax.plot(s_explicit[:,ind,ind], 'x--')
    ax.set_xticks(range(0,26,5))
    ax.set_xlabel('timesteps')
    ax.set_ylabel('water saturation')
    ax.set_ylim([0.0,1.0])
    ax.grid()
    ax.legend(['SciPy', 'Implicit', 'Explicit'])
    ax.set_title('S[{},{}]'.format(ind,ind))
fig.savefig('solution_compare_1ph.png', bbox_inches=0, pad_inches=0)

# time comparison
plt.clf()
plt.plot()
plt.plot(t_scipy, 'o--')
plt.plot(t_implicit, '.--')
plt.plot(t_explicit, 'x--')
plt.grid()
plt.legend(['SciPy', 'Implicit', 'Explicit'])
plt.title('Computational Time for each Solver')
plt.xlabel('simulation timestep')
plt.ylabel('computational time (sec)')
plt.ylim([0.0,0.2])
plt.savefig('time_comparison_1ph.png',  bbox_inches=0, pad_inches=0)
