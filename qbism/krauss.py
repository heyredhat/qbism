# AUTOGENERATED! DO NOT EDIT! File to edit: 03krauss.ipynb (unless otherwise specified).

__all__ = ['apply_krauss', 'partial_trace_krauss', 'povm_map']

# Cell
import numpy as np
import qutip as qt

# Cell
def apply_krauss(dm, krauss):
	r"""
	Applies a Krauss map to a density matrix $\nho$. The Krauss map consists in some number of operators
	satisfying $\sum_{i} \hat{K}_{i}^{\dagger}\hat{K}_{i} = \hat{I}$. $\nho$ is transformed via:

	$$\nho \nightarrow \sum_{i} \hat{K}_{i}\nho\hat{K}_{i}^{\dagger} $$
	"""
	return sum([krauss[j]*dm*krauss[j].dag() for j in range(len(krauss))])

# Cell
def partial_trace_krauss(keep, dims):
    r"""
    Constructs the Krauss map corresponding to the partial trace. Takes `keep` which is a single index or list of indices denoting
    subsystems to keep, and a list `dims` of dimensions of the overall tensor product Hilbert space.

    For illustration, to trace over the $i^{th}$ subsystem of $n$, one would construct Krauss operators:

    $$ \hat{K}_{i} = I^{\otimes i - 1} \otimes \langle i \mid \otimes I^{\otimes n - i}$$.
    """
    if type(keep) == int:
        keep = [keep]
    trace_over = [i for i in range(len(dims)) if i not in keep]
    indices = [{trace_over[0]:t} for t in range(dims[trace_over[0]])]
    for i in trace_over[1:]:
        new_indices = []
        for t in range(dims[i]):
            new_indices.extend([{**j, **{i: t}} for j in indices])
        indices = new_indices
    return [qt.tensor(*[qt.identity(d) if i in keep else qt.basis(d, index[i]).dag() for i, d in enumerate(dims)]) for index in indices]

# Cell
def povm_map(krauss, A, B=None):
    r"""
    Represents a Krauss map on Qbist probability vectors. Takes a list of Krauss operators, a POVM $A$ on the initial Hilbert space,
    and a POVM $B$ on the final Hilbert space. If $B$ isn't provided, it's assumed to be the same as $A$. Then the matrix elements of the map are:

    $$K_{j, i} = tr( \mathbb{K}(\frac{\hat{A}_{i}}{tr \hat{A}_{i}})\hat{B}_{j} ) $$

    Where $\mathbb{K}(\hat{O})$ denotes the Krauss map applied to $O$.
    """
    B = B if type(B) != type(None) else A
    return np.array([[(apply_krauss(a/a.tr(), krauss)*b).tr() for a in A] for b in B]).real