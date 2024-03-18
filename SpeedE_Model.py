import re
from typing import ClassVar, Mapping, Any

from pykeen.losses import NSSALoss
from pykeen.models import ERModel
from pykeen.nn import EmbeddingSpecification

from Interactions import ExpressivE_Interaction, SpeedE_Interaction


class SpeedE(ERModel):
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=dict(type=int, low=9, high=11, scale='power', base=2),
        p=dict(type=int, low=1, high=2),
        min_denom=dict(type=float, low=2e-1, high=8e-1, step=1e-1)
    )

    loss_default = NSSALoss
    loss_default_kwargs = dict(margin=3, adversarial_temperature=2.0, reduction="sum")

    def __init__(
            self,
            embedding_dim: int = 50,
            p: int = 2,
            min_denom=0.5,
            tanh_map=True,
            interactionMode="baseExpressivE",
            **kwargs,
    ) -> None:
        if interactionMode == "baseExpressivE":
            # <<<<<< ExpressivE >>>>>>

            super().__init__(
                interaction=ExpressivE_Interaction(p=p, min_denom=min_denom, tanh_map=tanh_map),
                entity_representations=EmbeddingSpecification(
                    embedding_dim=embedding_dim,
                ),
                relation_representations=EmbeddingSpecification(
                    embedding_dim=6 * embedding_dim,
                ),  # d_h, d_t, c_h, c_t, s_h, s_t
                **kwargs,
            )
        elif re.search("SpeedE", interactionMode):
            if min_denom > 0:
                raise Exception(interactionMode + " SpeedE does not use the <min_denom> argument.\\"
                                                  "Please set <min_denom>=0.")

            number_inequations = int(re.search("SpeedE\_(\d)", interactionMode)[1])

            n_paras = 2
            n_single_paras = 0

            mode = "Min_SpeedE"  # <<<<<< Min_SpeedE >>>>>>

            if re.search("\_Eq", interactionMode):
                # <<<<<< Eq_SpeedE >>>>>>

                n_single_paras = n_single_paras + 1
                mode = "Eq_SpeedE"
            elif re.search("\_Diff", interactionMode):
                # <<<<<< Diff_SpeedE >>>>>>

                n_single_paras = n_single_paras + 2
                mode = "Diff_SpeedE"
            if re.search("\_dIs(\d+\.?\d*)", interactionMode):
                fixed_d = float(re.search("\_dIs(\d+\.?\d*)", interactionMode)[1])
            else:
                raise Exception("No d specified!")

            super().__init__(
                interaction=SpeedE_Interaction(p=p, tanh_map=tanh_map,
                                               number_inequations=number_inequations,
                                               mode=mode, fixed_d=fixed_d),
                entity_representations=EmbeddingSpecification(
                    embedding_dim=embedding_dim,
                ),
                relation_representations=EmbeddingSpecification(
                    embedding_dim=(n_paras) * number_inequations * embedding_dim + n_single_paras,
                ),
                **kwargs,
            )

        else:
            raise Exception("<<< Interaction Mode unkown! >>>")
