from typing import Any

import lightning.pytorch as pl
import torch
from torch.nn import Module

import wandb

from pydentification.models.modules.feedforward import TimeSeriesLinear  # isort:skip
from pydentification.models.networks.fsnn.model import FrequencyLinear, TimeFrequencyLinear  # isort:skip
from pydentification.training.lightning.simulation import LightningSimulationTrainingModule  # isort:skip


def _build_activation(name: str, parameters: dict[str, Any]) -> Module:
    activations = {
        "elu": torch.nn.ELU,
        "identity": torch.nn.Identity,
        "hardshrink": torch.nn.Hardshrink,
        "hardsigmoid": torch.nn.Hardsigmoid,
        "hardtanh": torch.nn.Hardtanh,
        "hardswish": torch.nn.Hardswish,
        "leaky_relu": torch.nn.LeakyReLU,
        "log_sigmoid": torch.nn.LogSigmoid,
        "prelu": torch.nn.PReLU,
        "relu": torch.nn.ReLU,
        "relu6": torch.nn.ReLU6,
        "rrelu": torch.nn.RReLU,
        "selu": torch.nn.SELU,
        "celu": torch.nn.CELU,
        "gelu": torch.nn.GELU,
        "sigmoid": torch.nn.Sigmoid,
        "silu": torch.nn.SiLU,
        "mish": torch.nn.Mish,
        "softplus": torch.nn.Softplus,
        "softshrink": torch.nn.Softshrink,
        "softsign": torch.nn.Softsign,
        "tanh": torch.nn.Tanh,
        "tanhsrink": torch.nn.Tanhshrink,
        "threshold": torch.nn.Threshold,
        "GLU": torch.nn.GLU,
    }

    activation = activations[name]
    return activation(**parameters)


def model_from_parameter(module: Module, parameters: dict[str, Any]) -> Module:
    layers = []

    layers.append(
        # first layer maps from input to hidden
        module(
            n_input_time_steps=parameters["n_input_time_steps"],
            n_output_time_steps=parameters["n_hidden_time_steps"],
            n_input_state_variables=parameters["n_input_state_variables"],
            n_output_state_variables=parameters["n_hidden_state_variables"],
        )
    )

    layers.append(
        _build_activation(name=parameters["activation"], parameters=parameters.get("activation_parameters", {}))
    )

    for _ in range(parameters["n_hidden_layers"]):
        layers.append(
            # intermediate layers map from hidden to hidden
            module(
                n_input_time_steps=parameters["n_hidden_time_steps"],
                n_output_time_steps=parameters["n_hidden_time_steps"],
                n_input_state_variables=parameters["n_hidden_state_variables"],
                n_output_state_variables=parameters["n_hidden_state_variables"],
            )
        )

        layers.append(
            _build_activation(name=parameters["activation"], parameters=parameters.get("activation_parameters", {}))
        )

    layers.append(
        # last layer maps from hidden to output
        # no activation, since only regression problems are considered
        module(
            n_input_time_steps=parameters["n_hidden_time_steps"],
            n_output_time_steps=parameters["n_output_time_steps"],
            n_input_state_variables=parameters["n_hidden_state_variables"],
            n_output_state_variables=parameters["n_output_state_variables"],
        )
    )

    return torch.nn.Sequential(*layers)


def model_fn(
    project_name: str, config: dict[str, Any], parameters: dict[str, Any]
) -> tuple[pl.LightningModule, pl.Trainer]:
    module_mapping = {
        "MLP": TimeSeriesLinear,  # regular MLP for time-series
        "FMLP": FrequencyLinear,  # MLP using Fourier Transform
        "FSNN": TimeFrequencyLinear,  # MLP and FMLP merged into 2 branch network
    }

    model = model_from_parameter(module=module_mapping[parameters["model_type"]], parameters=parameters)

    optimizer = torch.optim.Adam(model.parameters())  # fixed initial LR
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=config["lr_patience"], verbose=True)

    timer = pl.callbacks.Timer(duration=config["timeout"], interval="epoch")
    stopping = pl.callbacks.EarlyStopping(
        monitor="training/validation_loss", patience=config["patience"], mode="min", verbose=True
    )

    path = f"models/{wandb.run.id}"

    model = LightningSimulationTrainingModule(
        model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )

    trainer = pl.Trainer(
        max_epochs=config["n_epochs"],
        accelerator="auto",  # use GPUs if available
        default_root_dir=path,
        callbacks=[timer, stopping],
        logger=pl.loggers.WandbLogger(project=project_name),
    )

    return model, trainer
