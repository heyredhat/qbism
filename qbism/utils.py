# AUTOGENERATED! DO NOT EDIT! File to edit: 04utils.ipynb (unless otherwise specified).

__all__ = ['flatten']

# Cell
def flatten(to_flatten):
    return [item for sublist in to_flatten for item in sublist]