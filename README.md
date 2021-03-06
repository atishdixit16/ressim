# ressim: reservoir simulation in python
Reservoir simulation in Python. It currently supports 2D rectangular grids and isotropic permeability.

## Modules
- `ressim.py`: main module containing classes to define the grid, and to model and solve the pressure and saturation equations.

- `utils.py`: module containing useful function definitions, e.g. linear and quadratic mobility functions, fractional flow function, etc.

## Usage
See `main_*.py` for examples.

# `ressim.py`
A module for reservoir simulation in Python.
## Grid
```python
Grid(self, nx, ny, lx=1.0, ly=1.0)
```
A class to define a rectangular grid. Given `nx, ny, lx, ly`, it maintains an update of attributes `vol, dx, dy, ncell, shape`.

__Attributes__

- `nx, ny`: `int`
        Grid resolution.

- `lx, ly`: `float`
        Grid physical dimensions. Default is `lx=1.0, ly=1.0`, i.e. unit square.

- `vol`: `float`
        Cell volume.

- `dx, dy`: `float`
        Cell dimensions.

- `ncell`: `int`
        Number of cells.

- `shape`: `int`
        Grid shape, i.e. `(ny, nx)`.

## PressureEquation
```python
PressureEquation(self, grid=None, q=None, k=None, diri=None, lamb_fn=None, s=None)
```

A class to model and solve the pressure equation,

![pressure](https://i.imgur.com/PLoJ0bj.gif)

with no-flow boundary conditions (closed reservoir).

__Attributes__

- `grid`: `Grid`
    Simulation grid.

- `q`: `ndarray, shape (ny, nx) | (ny*nx,)`
    Integrated source term.

- `k`: `ndarray, shape (ny, nx)`
    Permeability

- `diri`: `list` of `(int, float)` tuples
    Dirichlet boundary conditions, e.g. `[(i1, val1), (i2, val2), ...]`
    means pressure values `val1` at cell `i1`, `val2` at cell `i2`, etc. Defaults
    to `[(ny*nx/2, 0.0)]`, i.e. zero pressure at center of the grid.

- `lamb_fn`: `callable`
    Total mobility function `lamb_fn(s)`

- `s`: `ndarray, (ny, nx) | (ny*nx,)`
    Water saturation

- `p`: `ndarray, (ny, nx)`
    Pressure

- `v`: `dict` of `ndarray`
    - `'x' `: `ndarray, shape (ny, nx+1)`
        Flux in x-direction
    - `'y' `: `ndarray, shape (ny+1, nx)`
        Flux in y-direction

__Methods__

- `step()`:
    Solve the pressure equation to obtain pressure and flux. Update
    `self.p` and `self.v`.

- `solve(mat, q)`:
    Method to solve the system of linear equations. Default is
    `scipy.sparse.linalg.spsolve(mat, q)`. You can override this method to
    use a different solver.

## SaturationEquation
```python
SaturationEquation(self, grid=None, q=None, phi=None, s=None, f_fn=None, v=None, df_fn=None)
```
A class to model and solve the (water) saturation equation under water injection,

![Water saturation](https://i.imgur.com/qswqrcK.gif)

__Attributes__
- `grid`: `Grid`
    Simulation grid

- `q`: `ndarray, (ny, nx) | (ny*nx,)`
    Integrated source term.

- `phi`: `ndarray, (ny, nx) | (ny*nx,)`
    Porosity

- `f_fn`: `callable`
    Water fractional flow function `f_fn(s)`

- `v`: `dict` of `ndarray`
    - `'x'` : `ndarray, (ny, nx+1)`
        Flux in x-direction
    - `'y'` : `ndarray, (ny+1, nx)`
        Flux in y-direction

- `df_fn`: `callable` (optional)
    Derivative (element-wise) of water fractional flow function `df_fn(s)`.
    It is used to compute the jacobian of the residual function. If `None`,
    the jacobian is approximated by the solver (which can be slow).

- `s` : `ndarray, (ny, nx) | (ny*nx,)`
    Water saturation

__Methods__
- `step(dt)`:
    Solve saturation forward in time by `dt`. Update `self.s`.

- `solve(residual, s0, residual_jac=None)`:
    Method to perform the minimization of the residual. Default is
    `scipy.optimize.nonlin.nonlin_solve(residual, s0, jacobian=residual_jac)`.
    If `residual_jac` is `None`, defaults to `'krylov'`.
    You can override this method to use a different solver.

# `utils.py`
Useful functions for reservoir simulation tasks.

## linear_mobility
```python
linear_mobility(s, mu_w, mu_o, s_wir, s_oir, deriv=False)
```
Function to compute water and oil mobility with a *linear* model.

![](https://i.imgur.com/kTSiUd4.gif)

__Parameters__

- `s`: `ndarray, (ny, nx) | (ny*nx,)`
    Water saturation

- `mu_w`: `float`
    Viscosity of water

- `mu_o`: `float`
    Viscosity of oil

- `s_wir`: `float`
    Irreducible water saturation

- `s_oir`: `float`
    Irreducible oil saturation

- `deriv`: `bool`
    If `True`, also return derivatives

__Returns__

`if deriv=False:`
- `lamb_w, lamb_o`: `ndarray, (ny, nx) | (ny*nx,)`
    - `lamb_w`: water mobility
    - `lamb_o`: oil mobility

`if deriv=True:`
- `lamb_w, lamb_o, dlamb_w, dlamb_o`: `ndarray, (ny, nx) | (ny*nx,)`
    - `lamb_w`: water mobility
    - `lamb_o`: oil mobility
    - `dlamb_w`: derivative of water mobility
    - `dlamb_o`: derivative of oil mobility

## quadratic_mobility
```python
quadratic_mobility(s, mu_w, mu_o, s_wir, s_oir, deriv=False)
```
Function to compute water and oil mobility with a *quadratic* model.

![](https://i.imgur.com/DbPnjSk.gif)

__Parameters__
- `s`: `ndarray, (ny, nx) | (ny*nx,)`
    Water saturation

- `mu_w`: `float`
    Viscosity of water

- `mu_o`: `float`
    Viscosity of oil

- `s_wir`: `float`
    Irreducible water saturation

- `s_oir`: `float`
    Irreducible oil saturation

- `deriv`: `bool`
    If `True`, also return derivatives

__Returns__

`if deriv=False:`
- `lamb_w, lamb_o`: `ndarray, (ny, nx) | (ny*nx,)`
    - `lamb_w`: water mobility
    - `lamb_o`: oil mobility

`if deriv=True:`
- `lamb_w, lamb_o, dlamb_w, dlamb_o`: `ndarray, (ny, nx) | (ny*nx,)`
    - `lamb_w`: water mobility
    - `lamb_o`: oil mobility
    - `dlamb_w`: derivative of water mobility
    - `dlamb_o`: derivative of oil mobility

## f_fn
```python
f_fn(s, mobi_fn)
```
Water fractional flow function. 

![](https://i.imgur.com/2grtPNR.gif)

__Parameters__
- `s`: `ndarray, (ny, nx) | (ny*nx,)`
    Water saturation

- `mobi_fn`: `callable`
    Mobility function `lamb_w, lamb_o = mobi_fn(s)` where:
    - `lamb_w`: water mobility
    - `lamb_o`: oil mobility

__Returns__
- `ndarray, (ny, nx) | (ny*nx,)`
    Fractional flow

## df_fn
```python
df_fn(s, mobi_fn)
```
Derivative function (element-wise) of water fractional flow.

![](https://i.imgur.com/bwz8RQ1.gif)

__Parameters__
- `s`: `ndarray, (ny, nx) | (ny*nx,)`
    Water saturation

- `mobi_fn`: `callable`
    Mobility function `lamb_w, lamb_o, dlamb_w, dlamb_o = mobi_fn(s, deriv=True)` where:
    - `lamb_w`: water mobility
    - `lamb_o`: oil mobility
    - `dlamb_w`: derivative of water mobility
    - `dlamb_o`: derivative of oil mobility

__Returns__
- `ndarray, (ny, nx) | (ny*nx,)`
    Fractional flow derivative

## lamb_fn
```python
lamb_fn(s, mobi_fn)
```
Total mobility function.

![](https://i.imgur.com/YCKjZ8N.gif)

__Parameters__
- `s`: `ndarray, (ny, nx) | (ny*nx,)`
    Water saturation

- `mobi_fn`: `callable`
    Mobility function `lamb_w, lamb_o = mobi_fn(s)` where:
    - `lamb_w`: water mobility
    - `lamb_o`: oil mobility

__Returns__
- `ndarray, (ny, nx) | (ny*nx,)` Total mobility

# References
Aarnes J.E., Gimse T., Lie KA. (2007) An Introduction to the Numerics of Flow in Porous Media using Matlab. In: Hasle G., Lie KA., Quak E. (eds) Geometric Modelling, Numerical Simulation, and Optimization. Springer, Berlin, Heidelberg