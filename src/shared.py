from typing import Any

import lightning.pytorch as pl
import torch

from pydentification.data.datamodules.simulation import SimulationDataModule  # isort:skip
from pydentification.experiment import reporters  # isort:skip
from pydentification.metrics import regression_metrics  # isort:skip


def input_fn(data_config: dict[str, Any], parameters: dict[str, Any]) -> pl.LightningDataModule:
    """
    Creates pl.LightningDataMo from data_config and training parameters

    :param data_config: static dataset values, such as path and test size
    :param parameters: dynamic training parameters, such as batch size or input and output lengths in samples

    :return: pl.LightningDataModule supporting selected training
    """
    return SimulationDataModule.from_csv(  # type: ignore
        dataset_path=data_config["path"],
        input_columns=data_config["input_columns"],
        output_columns=data_config["output_columns"],
        test_size=data_config["test_size"],
        batch_size=parameters["batch_size"],
        validation_size=parameters["validation_size"],
        shift=parameters["shift"],
        forward_input_window_size=parameters["n_input_time_steps"],
        forward_output_window_size=parameters["n_input_time_steps"],
        # always predict one-step ahead
        forward_output_mask=parameters["n_input_time_steps"] - parameters["n_output_time_steps"],
    )


def report_fn(model: pl.LightningModule, trainer: pl.Trainer, dm: pl.LightningDataModule):  # noqa: F811
    y_pred = []
    dm.setup("test")  # make sure all data is prepared

    # short manual prediction loop, since pl.LightningModule (2.2.0) requires dict input
    for x, y in dm.test_dataloader():
        with torch.no_grad():
            y_hat = model(x)
            y_pred.append(y_hat)

    # detach to make sure all tensors are on CPU before numpy conversion
    y_pred = torch.cat(y_pred).detach().cpu().numpy()
    y_true = torch.cat([y for _, y in dm.test_dataloader()]).detach().cpu().numpy()

    metrics = regression_metrics(y_pred=y_pred.flatten(), y_true=y_true.flatten())  # type: ignore

    reporters.report_metrics(metrics, prefix="test")  # type: ignore
    reporters.report_trainable_parameters(model, prefix="config")
    reporters.report_prediction_plot(predictions=y_pred, targets=y_true, prefix="test")
