import torch
from torch import nn


def preprocess_entities(es, tanh_map=True):
    prep_entites = []

    if tanh_map:
        for e in es:
            prep_entites.append(torch.tanh(e))
        return prep_entites
    else:
        return es


def positive_sign(x):
    s = torch.sign(x)
    s[s == 0] = 1
    return s


def preprocess_relations(r, tanh_map=True, min_denom=0):
    d_h, d_t, c_h, c_t, s_h, s_t = r.tensor_split(6, dim=-1)

    ReLU = nn.ReLU()

    d_h = torch.abs(d_h)
    d_t = torch.abs(d_t)

    if tanh_map:
        d_h = torch.tanh(d_h)
        d_t = torch.tanh(d_t)
        c_h = torch.tanh(c_h)
        c_t = torch.tanh(c_t)
    else:
        d_h = ReLU(d_h)
        d_t = ReLU(d_t)

    s_t = s_t + positive_sign(s_t) * 1e-4
    s_h_c = s_h.clone()

    diag_denominator = 1 - s_h.mul(s_t)
    slope_update_mask = torch.abs(diag_denominator) < min_denom

    adjusted_min_denom = diag_denominator - positive_sign(diag_denominator) * min_denom
    s_h_c[slope_update_mask] = s_h[slope_update_mask] + adjusted_min_denom[slope_update_mask] / s_t[slope_update_mask]

    return d_h, d_t, c_h, c_t, s_h_c, s_t