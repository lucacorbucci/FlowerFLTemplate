import logging
from logging import INFO

import numpy as np
import wandb
from flwr.common.logger import log


class Aggregation:
    def agg_metrics_test(metrics: list, server_round: int, wandb_run) -> dict:
        total_examples = sum([n_examples for n_examples, _ in metrics])

        loss_test = (
            sum([n_examples * metric["loss"] for n_examples, metric in metrics])
            / total_examples
        )
        accuracy_test = (
            sum([n_examples * metric["accuracy"] for n_examples, metric in metrics])
            / total_examples
        )
        f1_test = (
            sum([n_examples * metric["f1_score"] for n_examples, metric in metrics])
            / total_examples
        )

        log(
            INFO,
            f"Test Accuracy: {accuracy_test} - Test Loss {loss_test}",
        )

        agg_metrics = {
            "Test Loss": loss_test,
            "Test Accuracy": accuracy_test,
            "FL Round": server_round,
            "Test F1": f1_test,
        }
        if wandb_run:
            wandb_run.log(agg_metrics)

        return agg_metrics

    def agg_metrics_evaluation(metrics: list, server_round: int, wandb_run) -> dict:
        total_examples = sum([n_examples for n_examples, _ in metrics])

        loss_evaluation = (
            sum([n_examples * metric["loss"] for n_examples, metric in metrics])
            / total_examples
        )
        accuracy_evaluation = (
            sum([n_examples * metric["accuracy"] for n_examples, metric in metrics])
            / total_examples
        )
        f1_validation = (
            sum([n_examples * metric["f1_score"] for n_examples, metric in metrics])
            / total_examples
        )

        agg_metrics = {
            "Validation Loss": loss_evaluation,
            "Validation_Accuracy": accuracy_evaluation,
            "FL Round": server_round,
            "Validation F1": f1_validation,
        }
        if wandb_run:
            wandb_run.log(agg_metrics)

        return agg_metrics

    def agg_metrics_train(metrics: list, server_round: int, fed_dir, wandb_run) -> dict:
        # Collect the losses logged during each epoch in each client
        total_examples = sum([n_examples for n_examples, _ in metrics])
        losses = []
        losses_with_regularization = []
        accuracies = []

        for n_examples, node_metrics in metrics:
            losses.append(n_examples * node_metrics["train_loss"])
            accuracies.append(n_examples * node_metrics["train_accuracy"])
            client_id = node_metrics["cid"]

            # Create the dictionary we want to log. For some metrics we want to log
            # we have to check if they are present or not.
            to_be_logged = {
                "FL Round": server_round,
            }

            if wandb_run:
                wandb_run.log(
                    to_be_logged,
                )

        log(
            INFO,
            f"Train Accuracy: {sum(accuracies) / total_examples} - Train Loss {sum(losses) / total_examples}",
        )

        agg_metrics = {
            "Train Loss": sum(losses) / total_examples,
            "Train Accuracy": sum(accuracies) / total_examples,
            "Train Loss with Regularization": sum(losses_with_regularization)
            / total_examples,
            "FL Round": server_round,
        }
        if wandb_run:
            wandb_run.log(agg_metrics)

        return agg_metrics
