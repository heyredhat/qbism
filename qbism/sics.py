# AUTOGENERATED! DO NOT EDIT! File to edit: 02sics.ipynb (unless otherwise specified).

__all__ = ['load_fiducial', 'sic_states', 'sic_povm', 'sic_gram', 'hoggar_fiducial', 'hoggar_indices', 'hoggar_povm']

# Cell
import numpy as np
import qutip as qt
from itertools import product
import pkg_resources

from .weyl_heisenberg import *

# Cell
def load_fiducial(d):
    r"""
    Loads a Weyl-Heisenberg covariant SIC-POVM fiducial state of dimension $d$ from the repository provided here: http://www.physics.umb.edu/Research/QBism/solutions.html.
    """
    f = pkg_resources.resource_stream(__name__, "sic_povms/d%d.txt" % d)
    fiducial = []
    for line in f:
        if line.strip() != "":
            re, im = [float(v) for v in line.split()]
            fiducial.append(re + 1j*im)
    return qt.Qobj(np.array(fiducial)).unit()

# Cell
def sic_states(d):
    r"""
    Returns the $d^2$ states constructed by applying the Weyl-Heisenberg displacement operators to the SIC-POVM fiducial state of dimension $d$.
    """
    return weyl_heisenberg_states(load_fiducial(d))

# Cell
def sic_povm(d):
    r"""
    Returns a SIC-POVM of dimension $d$.
    """
    return weyl_heisenberg_povm(load_fiducial(d))

# Cell
def sic_gram(d):
    r"""
    The Gram matrix is the matrix of inner products: $G_{i,j} = \langle v_{i} \mid v_{j} \rangle$. For a SIC, this matrix should consist of 1's along the diagonal, and all other entries $\frac{1}{d}$:

    $$ \begin{pmatrix} 1 & \frac{1}{d} & \frac{1}{d} & \dots \\
                       \frac{1}{d} & 1 & \frac{1}{d} & \dots \\
                       \frac{1}{d} & \frac{1}{d} & 1 & \dots \\
                       \vdots & \vdots & \vdots & \ddots \end{pmatrix}$$

    """
    return np.array([[1 if i == j else 1/(d+1) for j in range(d**2)] for i in range(d**2)])

# Cell
def hoggar_fiducial():
    r"""
    Returns a fiducial state for the exceptional SIC in dimension $8$, the Hoggar SIC.

    Unnormalized: $\begin{pmatrix} -1 + 2i \\ 1 \\ 1 \\ 1 \\ 1 \\ 1 \\ 1 \\ 1 \end{pmatrix}$.
    """
    fiducial = qt.Qobj(np.array([-1 + 2j, 1, 1, 1, 1, 1, 1, 1])).unit()
    fiducial.dims = [[2,2,2],[1,1,1]]
    return fiducial

# Cell
def hoggar_indices():
    r"""
    Returns a list with entries $(a, b, c, d, e, f)$ for $a, b, c, d, e, f \in [0, 1]$.
    """
    return list(product([0,1], repeat=6))

# Cell
def hoggar_povm():
    r"""
    Constructs the Hoggar POVM, which is covariant under the tensor product of three copies of the $d=2$ Weyl-Heisenberg group. In other words, we apply the 64 displacement operators:

    $$ \hat{D}_{a, b, c, d, e, f} = X^{a}Z^{b} \otimes X^{c}Z^{d} \otimes X^{e}Z^{f} $$

    To the Hoggar fiducial state, form the corresponding projectors, and rescale by $\frac{1}{8}$.
    """
    Z, X = clock(2), shift(2)
    indices = hoggar_indices()
    D = dict([(I, qt.tensor(X**I[0]*Z**I[1],\
                            X**I[2]*Z**I[3],\
                            X**I[4]*Z**I[5])) for I in indices])
    fiducial = hoggar_fiducial()
    hoggar_states = [D[I]*fiducial for I in indices]
    return [(1/8)*state*state.dag() for state in hoggar_states]