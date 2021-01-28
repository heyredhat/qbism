# QBism 



[![PyPI version](https://badge.fury.io/py/qbism.svg)](https://badge.fury.io/py/qbism)

> python tools for the budding quantum bettabilitarian

## Installation

`pip install qbism`

Note that `qbism` relies on `qutip`!

## Usage

Let's start off with a random density matrix:

```
from qbism import *
import qutip as qt
import numpy as np

d = 2
rho = qt.rand_dm(d)
```

We construct a random Weyl-Heisenberg IC-POVM, and get the magical quantum coherence matrix. We find the probabilities with respect to this POVM.

```
povm = weyl_heisenberg_povm(qt.rand_ket(d))
phi = povm_phi(povm)
p = dm_probs(rho, povm)
print("probs: %s" % p)
```

    probs: [0.20215649 0.20215649 0.29784351 0.29784351]


We can compare the classical probabilities (for a Von Neumann measurement after a POVM measurement whose outcome we are ignorant of) to the the quantum probabilities (in the case where we go directly to the Von Neumann measurement):

```
H = qt.rand_herm(d)
vn = [v*v.dag() for v in H.eigenstates()[1]]

classical_probs = conditional_probs(vn, povm) @ p
quantum_probs = conditional_probs(vn, povm) @ phi @ p

print("classsical probs: %s" % classical_probs)
print("quantum probs: %s" % quantum_probs)

post_povm_rho = sum([(e*rho).tr()*(e/e.tr()) for e in povm])
assert np.allclose(classical_probs, [(v*post_povm_rho).tr() for v in vn])
assert np.allclose(quantum_probs, [(v*rho).tr() for v in vn])
```

    classsical probs: [0.55802905 0.44197095]
    quantum probs: [0.65778315 0.34221685]


Now let's get a SIC-POVM and explore time evolution:

```
sic = sic_povm(d)
sic_phi = povm_phi(sic)
sic_p = dm_probs(rho, sic)

U = qt.rand_unitary(d)
evolved_sic = [U*e*U.dag() for e in sic]
R = conditional_probs(evolved_sic, sic).T

time_evolved_sic_p = R @ sic_phi @ sic_p
print("time evolved probs: %s" % time_evolved_sic_p)
assert np.allclose(dm_probs(U*rho*U.dag(), sic), time_evolved_sic_p)
```

    time evolved probs: [0.20445193 0.20445193 0.29554807 0.29554807]


We could also use:

```
time_evolved_sic_p2 = povm_map([U], sic) @ sic_phi @ sic_p
assert np.allclose(time_evolved_sic_p, time_evolved_sic_p2)
```

Finally, let's check out partial traces:

```
entangled = qt.rand_dm(4)
entangled.dims = [[2,2],[2,2]]

povm2 = weyl_heisenberg_povm(qt.rand_ket(2))
povm4 = apply_dims(weyl_heisenberg_povm(qt.rand_ket(4)), [2,2])
phi = povm_phi(povm4)
p = dm_probs(entangled, povm4)

ptrA = povm_map(partial_trace_kraus(0, [2,2]), povm4, povm2)
ptrB = povm_map(partial_trace_kraus(1, [2,2]), povm4, povm2)

assert np.allclose(dm_probs(entangled.ptrace(0), povm2), ptrA @ phi @ p)
assert np.allclose(dm_probs(entangled.ptrace(1), povm2), ptrB @ phi @ p)
```

Check out the tutorial for the full story!

Thanks to [nbdev](https://nbdev.fast.ai/)!
