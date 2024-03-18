import json

import torch
from pykeen.datasets import FB15k237, WN18RR
from pykeen.triples import TriplesFactory

from SpeedE_Model import SpeedE


def load_checkpoint(config_path, checkpoint_path):
    with open(config_path, "r") as f:
        config = json.loads(f.read())

    checkpoint = torch.load(checkpoint_path)

    if config["dataset"] == "FB15k237":
        dataset = FB15k237()
    elif config["dataset"] == "WN18RR":
        dataset = WN18RR()
    else:
        raise Exception("Dataset %s unknown!" % config["dataset"])

    triples_factory = TriplesFactory(
        mapped_triples=dataset.training.mapped_triples,
        relation_to_id=checkpoint["relation_to_id_dict"],
        entity_to_id=checkpoint["entity_to_id_dict"],
        create_inverse_triples=config["dataset_kwargs"]["create_inverse_triples"],
    )

    if config["loss"] == "NSALoss" or config["loss"] == "NSSALoss":
        loss_str = "nssa"
    elif config["loss"] == "CrossEntropyLoss" or config["loss"] == "BCEWithLogitsLoss":
        loss_str = config["loss"]
    else:
        raise Exception("Unknown loss \"%s\"" % config["loss"])

    if config["model"] == "Model":

        if "interactionMode" in config["model_kwargs"]:
            trained_model = SpeedE(triples_factory=triples_factory,
                                            embedding_dim=config["model_kwargs"]["embedding_dim"],
                                            p=config["model_kwargs"]["p"],
                                            min_denom=config["model_kwargs"]["min_denom"],
                                            tanh_map=config["model_kwargs"]['tanh_map'],
                                            interactionMode=config["model_kwargs"]["interactionMode"],
                                            loss=loss_str,
                                            loss_kwargs=config["loss_kwargs"]
                                            )
        else:
            trained_model = SpeedE(triples_factory=triples_factory,
                                            embedding_dim=config["model_kwargs"]["embedding_dim"],
                                            p=config["model_kwargs"]["p"],
                                            min_denom=config["model_kwargs"]["min_denom"],
                                            tanh_map=config["model_kwargs"]['tanh_map'],
                                            loss=loss_str,
                                            loss_kwargs=config["loss_kwargs"]
                                            )

    trained_model.load_state_dict(checkpoint['model_state_dict'])

    return config, dataset, trained_model
