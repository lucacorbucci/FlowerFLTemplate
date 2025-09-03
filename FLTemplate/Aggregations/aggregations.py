import logging
from logging import INFO

import numpy as np
import wandb
from flwr.common.logger import log


class Aggregation:
    def agg_metrics_test(metrics: list, server_round: int, wandb_run) -> dict:
        total_examples = sum([n_examples for n_examples, _ in metrics])

        loss_test = sum([n_examples * metric["loss"] for n_examples, metric in metrics]) / total_examples

        agg_metrics = {}

        if metrics[0][1].get("accuracy"):
            accuracy_test = sum([n_examples * metric["accuracy"] for n_examples, metric in metrics]) / total_examples
            log(
                INFO,
                f"Test Accuracy: {accuracy_test} - Test Loss {loss_test}",
            )

            agg_metrics["Test Loss"] = loss_test
            agg_metrics["Test_Accuracy"] = accuracy_test
            agg_metrics["FL Round"] = server_round

        if metrics[0][1].get("rmse"):
            rmse_test = sum([n_examples * metric["rmse"] for n_examples, metric in metrics]) / total_examples
            mae_test = sum([n_examples * metric["mae"] for n_examples, metric in metrics]) / total_examples
            r2_test = sum([n_examples * metric["r2"] for n_examples, metric in metrics]) / total_examples
            mse_test = sum([n_examples * metric["mse"] for n_examples, metric in metrics]) / total_examples

            agg_metrics["Test Loss"] = loss_test
            agg_metrics["rmse_test"] = rmse_test
            agg_metrics["mae_test"] = mae_test
            agg_metrics["r2_test"] = r2_test
            agg_metrics["mse_test"] = mse_test
            agg_metrics["FL Round"] = server_round

        if wandb_run:
            wandb_run.log(agg_metrics)

        return agg_metrics

    def agg_metrics_evaluation(metrics: list, server_round: int, wandb_run) -> dict:
        total_examples = sum([n_examples for n_examples, _ in metrics])
        agg_metrics = {}
        loss_evaluation = sum([n_examples * metric["loss"] for n_examples, metric in metrics]) / total_examples
        if metrics[0][1].get("accuracy"):
            accuracy_evaluation = sum([n_examples * metric["accuracy"] for n_examples, metric in metrics]) / total_examples

            agg_metrics["Validation Loss"] = loss_evaluation
            agg_metrics["Validation_Accuracy"] = accuracy_evaluation
            agg_metrics["FL Round"] = server_round
        if metrics[0][1].get("rmse"):
            rmse_evaluation = sum([n_examples * metric["rmse"] for n_examples, metric in metrics]) / total_examples
            mae_evaluation = sum([n_examples * metric["mae"] for n_examples, metric in metrics]) / total_examples
            r2_evaluation = sum([n_examples * metric["r2"] for n_examples, metric in metrics]) / total_examples
            mse_evaluation = sum([n_examples * metric["mse"] for n_examples, metric in metrics]) / total_examples

            agg_metrics["Validation Loss"] = loss_evaluation
            agg_metrics["rmse_evaluation"] = rmse_evaluation
            agg_metrics["mae_evaluation"] = mae_evaluation
            agg_metrics["r2_evaluation"] = r2_evaluation
            agg_metrics["mse_evaluation"] = mse_evaluation
            agg_metrics["FL Round"] = server_round

        if wandb_run:
            wandb_run.log(agg_metrics)

        return agg_metrics

    def agg_metrics_train(metrics: list, server_round: int, fed_dir, wandb_run) -> dict:
        # Collect the losses logged during each epoch in each client
        total_examples = sum([n_examples for n_examples, _ in metrics])
        losses = []
        accuracies = []
        accuracy_log = False
        for n_examples, node_metrics in metrics:
            losses.append(n_examples * node_metrics["loss"])
            if node_metrics.get("accuracy"):
                accuracies.append(n_examples * node_metrics["accuracy"])
                accuracy_log = True

            # Create the dictionary we want to log. For some metrics we want to log
            # we have to check if they are present or not.
            to_be_logged = {
                "FL Round": server_round,
            }

            if wandb_run:
                wandb_run.log(
                    to_be_logged,
                )

        if accuracy_log:
            log(
                INFO,
                f"Train Accuracy: {sum(accuracies) / total_examples} - Train Loss {sum(losses) / total_examples}",
            )

            agg_metrics = {
                "Train Loss": sum(losses) / total_examples,
                "Train Accuracy": sum(accuracies) / total_examples,
                # "Train Loss with Regularization": sum(losses_with_regularization) / total_examples,
                "FL Round": server_round,
            }
        else: 
            log(
                INFO,
                f"Train Loss {sum(losses) / total_examples}",
            )

            agg_metrics = {
                "Train Loss": sum(losses) / total_examples,
                "FL Round": server_round,
            }
        if wandb_run:
            wandb_run.log(agg_metrics)

        return agg_metrics
