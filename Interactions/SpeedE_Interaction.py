import torch
from pykeen.nn.modules import Interaction
from pykeen.utils import broadcast_cat

from Utils import preprocess_entities


class SpeedE_Interaction(Interaction):
    relation_shape = (
        'e'
    )
    entity_shape = (
        'd',
    )

    def __init__(self, p: int, tanh_map: bool = True, number_inequations: int = 6,
                 mode: str = "Min_SpeedE",
                 fixed_d: float = 0):
        super().__init__()
        self.p = p  # Norm that shall be used (either 1 or 2)
        self.tanh_map = tanh_map
        self.n_ineq = number_inequations
        self.mode = mode
        self.fixed_d = fixed_d

    def distance(self, c, s_dict, h, t):
        # Calculate the distance of the triple

        if self.n_ineq % 2 == 0:
            ht = broadcast_cat([h, t] * int(self.n_ineq / 2), dim=-1)
            th = broadcast_cat([t, h] * int(self.n_ineq / 2), dim=-1)
        else:
            ht = broadcast_cat([h, t] * int((self.n_ineq - 1) / 2) + [h], dim=-1)
            th = broadcast_cat([t, h] * int((self.n_ineq - 1) / 2) + [t], dim=-1)

        contextualized_pos = torch.abs(ht - c - s_dict["s"] * th)

        is_entity_pair_within_para = torch.le(contextualized_pos, s_dict["d"]).all(dim=-1)

        if self.mode in ["Diff_SpeedE"]:
            w_i = (2 * s_dict["d"]) * s_dict["s_lin_i"] + 1
            w_o = (2 * s_dict["d"]) * s_dict["s_lin_o"] + 1

            k = s_dict["s_lin_o"] * s_dict["d"] * w_o - s_dict["s_lin_i"] * s_dict["d"] / w_i

            # Case 1: Triple outside of Para
            dist = torch.mul(contextualized_pos, w_o) - k

            # Case 2: Triple within Para
            dist[is_entity_pair_within_para] = torch.div(contextualized_pos, w_i)[is_entity_pair_within_para]
        else:
            if self.mode in ["Eq_SpeedE"]:
                w = (2 * s_dict["d"]) * s_dict["s_lin"] + 1
            else:
                w = (2 * s_dict["d"]) + 1

            k = torch.mul(0.5 * (w - 1), (w - 1 / w))

            # Case 1: Triple outside of Para
            dist = torch.mul(contextualized_pos, w) - k

            # Case 2: Triple within Para
            dist[is_entity_pair_within_para] = torch.div(contextualized_pos, w)[is_entity_pair_within_para]

        return -dist.norm(p=self.p, dim=-1)

    def prepare_global_s(self, global_s):
        return torch.abs(global_s)

    def prepare_s_nonLin(self, s_lin):
        return torch.tanh(torch.abs(s_lin))

    def prepare_s_nonLin_io(self, s_lin_i, s_lin_o):
        s_lin_o = torch.tanh(torch.abs(s_lin_o))
        s_lin_i = s_lin_o + torch.abs(s_lin_i)
        return s_lin_i, s_lin_o

    def prepare_s_nonLin_i(self, s_lin_i):
        s_lin_i = 1 + torch.abs(s_lin_i)
        return s_lin_i

    def preprocess_relations(self, r, tanh_map=True):
        d = self.fixed_d

        if self.mode == "Diff_SpeedE":
            # Diff_SpeedE
            s_lin_i, s_lin_o, r = torch.split(r, [1, 1, r.size()[-1] - 2], dim=-1)
            c, s = r.tensor_split(2, dim=-1)
            s_lin_i, s_lin_o = self.prepare_s_nonLin_io(s_lin_i, s_lin_o)
            s_dict = {"s": s, "s_lin_i": s_lin_i, "s_lin_o": s_lin_o}
        elif self.mode == "Eq_SpeedE":
            # Eq_SpeedE
            s_lin, r = torch.split(r, [1, r.size()[-1] - 1], dim=-1)
            c, s = r.tensor_split(2, dim=-1)
            s_lin = self.prepare_s_nonLin(s_lin)
            s_dict = {"s": s, "s_lin": s_lin}
        else:
            # Min_ExpressivE
            c, s = r.tensor_split(2, dim=-1)
            s_dict = {"s": s}

        if tanh_map:
            c = torch.tanh(c)

        s_dict["d"] = d
        return c, s_dict

    def forward(self, h, r, t):
        c, s_dict = self.preprocess_relations(r, tanh_map=self.tanh_map)

        h, t = preprocess_entities([h, t], tanh_map=self.tanh_map)

        return self.distance(c, s_dict, h, t)
