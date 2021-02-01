# AUTOGENERATED! DO NOT EDIT! File to edit: 06rics.ipynb (unless otherwise specified).

__all__ = ['real_rand_ket', 'real_rand_dm', 'rand_symmetric', 'rand_orthogonal', 'petersen_povm', 'petersen', 'rho',
           'circular_shifts', 'icosahedron_vertices', 'icosahedron_povm', 'icosahedron', 'rho']

# Cell
import numpy as np
import qutip as qt
from scipy.stats import ortho_group

from .povm import *
from .random import *
from .kraus import *

# Cell
def real_rand_ket(d):
    r"""
    Generates a random ket in real Hilbert space of dimension $d$.
    """
    return qt.Qobj(np.random.randn(d)).unit()

# Cell
def real_rand_dm(d):
    r"""
    Generates a random density matrix for a real Hilbert space of dimension $d$.
    """
    return qt.Qobj(qt.rand_dm(d).full().real)

# Cell
def rand_symmetric(d):
    r"""
    Generates a random $d \times d$ symmetric matrix. These matrices correspond to observables in real quantum mechanics, being the real analogue of Hermitian matrices: $\hat{S} = \hat{S}^{T}$.
    """
    M = qt.Qobj(np.random.randn(d,d))
    return M*M.trans() + M.trans()*M

# Cell
def rand_orthogonal(d):
    r"""
    Generates a random $d \times d$ orthogonal matrix. These matrices correspond to time evolution in real quantum mechanics, being the real analogue of unitary matrices: $\hat{S}\hat{S}^{T} = \hat{I}$.
    """
    return qt.Qobj(ortho_group.rvs(d))

# Cell
def petersen_povm():
    petersen_vertices = ["u1", "u2", "u3", "u4", "u5", "v1", "v2", "v3", "v4", "v5"]
    petersen_graph = \
        {"u1": ["v1", "u2", "u5"],
        "u2": ["u1", "v2", "u3"],
        "u3": ["u2", "v3", "u4"],
        "u4": ["u3", "v4", "u5"],
        "u5": ["u4", "v5", "u1"],
        "v1": ["u1", "v4", "v3"],
        "v2": ["u2", "v4", "v5"],
        "v3": ["v5", "v1", "u3"],
        "v4": ["u4", "v1", "v2"],
        "v5": ["u5", "v3", "v2"]}
    petersen_gram = np.array([[1 if a == b else (\
                               -2/3 if b in petersen_graph[a] else \
                               1/6) for b in petersen_vertices]\
                                        for a in petersen_vertices])
    U, D, V = np.linalg.svd(petersen_gram)
    petersen_states = [qt.Qobj(state) for state in V[:4].T @ np.sqrt(np.diag(D[:4]))]
    return [(2/5)*v*v.dag() for v in petersen_states]

petersen = petersen_povm()
assert np.allclose(sum(petersen), qt.identity(4))

rho = real_rand_dm(4)
assert np.allclose(rho, probs_dm(dm_probs(rho, petersen), petersen))
print("petersen gram:\n %s" % np.round(povm_gram(petersen, normalized=False), decimals=3))
print("quantumness: %f" % quantumness(petersen))

# Cell
def circular_shifts(v):
    shifts = [v]
    for i in range(len(v)-1):
        u = shifts[-1][:]
        u.insert(0, u.pop())
        shifts.append(u)
    return shifts

def icosahedron_vertices():
    phi = (1+np.sqrt(5))/2
    return [np.array(v) for v in
               circular_shifts([0, 1, phi]) + \
               circular_shifts([0, -1, -phi]) + \
               circular_shifts([0, 1, -phi]) + \
               circular_shifts([0, -1, phi])]

def icosahedron_povm():
    vertices = icosahedron_vertices()
    keep = []
    for i, a in enumerate(vertices):
        for j, b in enumerate(vertices):
            if i != j and np.allclose(a, -b) and j not in keep:
                keep.append(i)
    vertices = [qt.Qobj(e).unit() for i, e in enumerate(vertices) if i in keep]
    return [(1/2)*v*v.dag() for v in vertices]

icosahedron = icosahedron_povm()
assert np.allclose(sum(icosahedron), qt.identity(3))

rho = real_rand_dm(3)
assert np.allclose(rho, probs_dm(dm_probs(rho, icosahedron), icosahedron))
print("icosahedron gram:\n %s" % np.round(povm_gram(icosahedron, normalized=False), decimals=3))
print("quantumness: %f" % quantumness(icosahedron))