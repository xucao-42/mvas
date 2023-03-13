import numpy as np
import torch

if torch.cuda.is_available():
    device = "cuda"
elif torch.has_mps:
    device = torch.device('mps')
else:
    device = "cpu"

print(f"Using {device} device")


def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m


def boundary_excluded_mask(mask):
    top_mask = np.pad(mask, ((1, 0), (0, 0)), "constant", constant_values=0)[:-1, :]
    bottom_mask = np.pad(mask, ((0, 1), (0, 0)), "constant", constant_values=0)[1:, :]
    left_mask = np.pad(mask, ((0, 0), (1, 0)), "constant", constant_values=0)[:, :-1]
    right_mask = np.pad(mask, ((0, 0), (0, 1)), "constant", constant_values=0)[:, 1:]
    be_mask = np.logical_and.reduce((top_mask, bottom_mask, left_mask, right_mask, mask))

    # discard single point
    top_mask = np.pad(be_mask, ((1, 0), (0, 0)), "constant", constant_values=0)[:-1, :]
    bottom_mask = np.pad(be_mask, ((0, 1), (0, 0)), "constant", constant_values=0)[1:, :]
    left_mask = np.pad(be_mask, ((0, 0), (1, 0)), "constant", constant_values=0)[:, :-1]
    right_mask = np.pad(be_mask, ((0, 0), (0, 1)), "constant", constant_values=0)[:, 1:]
    bes_mask = np.logical_or.reduce((top_mask, bottom_mask, left_mask, right_mask))
    be_mask = np.logical_and(be_mask, bes_mask)
    return be_mask


def boundary_expansion_mask(mask):
    left_mask = np.pad(mask, ((0, 0), (0, 1)), "constant", constant_values=0)[:, 1:]
    right_mask = np.pad(mask, ((0, 0), (1, 0)), "constant", constant_values=0)[:, :-1]
    top_mask = np.pad(mask, ((0, 1), (0, 0)), "constant", constant_values=0)[1:, :]
    bottom_mask = np.pad(mask, ((1, 0), (0, 0)), "constant", constant_values=0)[:-1, :]

    be_mask = np.logical_or.reduce((left_mask, right_mask, top_mask, bottom_mask, mask))
    return be_mask
