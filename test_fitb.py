# file: test_fitb.py
"""
This script loads a trained model and tests it for the FITB task.
"""

import json
import tensorflow as tf
import argparse
import numpy as np
from collections import namedtuple
import os

from utils import get_degree_supports, normalize_nonsym_adj, to_tf_sparse_tensor
from model.CompatibilityGAE import CompatibilityGAE
from dataloaders import DataLoaderPolyvore, DataLoaderFashionGen


def test_fitb(args):
    args_dict = args
    args = namedtuple("Args", args_dict.keys())(*args_dict.values())

    load_from = args.load_from
    config_file = os.path.join(load_from, "results.json")
    log_file = os.path.join(load_from, "log.json")
    checkpoint_path = os.path.join(load_from, "best_epoch")

    with open(config_file) as f:
        config = json.load(f)
    with open(log_file) as f:
        log = json.load(f)

    DATASET = config["dataset"]
    BN_AS_TRAIN = False
    ADJ_SELF_CONNECTIONS = True

    def norm_adj(adj_to_norm):
        return normalize_nonsym_adj(adj_to_norm)

    # Dataloader
    if DATASET == "fashiongen":
        dl = DataLoaderFashionGen()
    elif DATASET == "polyvore":
        dl = DataLoaderPolyvore()
    else:
        raise NotImplementedError(f"Dataloader for {DATASET} not implemented.")

    train_features, _, _, _, _ = dl.get_phase("train")
    val_features, adj_val, val_labels, val_r_indices, val_c_indices = dl.get_phase(
        "valid"
    )
    test_features, _, _, _, _ = dl.get_phase("test")

    _, mean, std = dl.normalize_features(train_features, get_moments=True)
    val_features = dl.normalize_features(val_features, mean=mean, std=std)
    test_features = dl.normalize_features(test_features, mean=mean, std=std)

    val_support = get_degree_supports(
        adj_val, config["degree"], adj_self_con=ADJ_SELF_CONNECTIONS
    )
    for i in range(1, len(val_support)):
        val_support[i] = norm_adj(val_support[i])

    num_support = len(val_support)

    model = CompatibilityGAE(
        input_dim=train_features.shape[1],
        hidden=config["hidden"],
        num_support=num_support,
        batch_norm=config["batch_norm"],
        dropout_rate=0.0,
    )

    # Build the model by calling it
    _ = model(
        {
            "node_features": tf.convert_to_tensor(val_features, dtype=tf.float32),
            "support": [to_tf_sparse_tensor(s) for s in val_support],
            "row_indices": tf.convert_to_tensor(val_r_indices, dtype=tf.int32),
            "col_indices": tf.convert_to_tensor(val_c_indices, dtype=tf.int32),
        },
        training=False,
    )

    model.load_weights(checkpoint_path).expect_partial()
    print("Model weights restored from:", checkpoint_path)

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    acc_metric = tf.keras.metrics.BinaryAccuracy()

    # --- Validation check ---
    val_inputs = {
        "node_features": tf.convert_to_tensor(val_features, dtype=tf.float32),
        "support": [to_tf_sparse_tensor(s) for s in val_support],
        "row_indices": tf.convert_to_tensor(val_r_indices, dtype=tf.int32),
        "col_indices": tf.convert_to_tensor(val_c_indices, dtype=tf.int32),
    }
    val_labels_tensor = tf.convert_to_tensor(val_labels, dtype=tf.float32)

    val_preds = model(val_inputs, training=BN_AS_TRAIN)
    val_loss = loss_fn(val_labels_tensor, val_preds)
    acc_metric.update_state(val_labels_tensor, val_preds)
    val_acc = acc_metric.result().numpy()
    acc_metric.reset_states()

    print(
        "val_loss=",
        "{:.5f}".format(val_loss.numpy()),
        "val_acc=",
        "{:.5f}".format(val_acc),
    )

    @tf.function
    def predict_step(inputs):
        return tf.nn.sigmoid(model(inputs, training=BN_AS_TRAIN))

    num_processed = 0
    correct = 0

    kwargs = {
        "K": args.k,
        "subset": args.subset,
        "resampled": args.resampled,
        "expand_outfit": args.expand_outfit,
    }

    test_features_tensor = tf.convert_to_tensor(test_features, dtype=tf.float32)

    for (
        question_adj,
        out_ids,
        choices_ids,
        labels,
        valid,
    ) in dl.yield_test_questions_K_edges(**kwargs):
        q_support = get_degree_supports(
            question_adj,
            config["degree"],
            adj_self_con=ADJ_SELF_CONNECTIONS,
            verbose=False,
        )
        for i in range(1, len(q_support)):
            q_support[i] = norm_adj(q_support[i])

        q_support_tensors = [to_tf_sparse_tensor(s) for s in q_support]

        q_inputs = {
            "node_features": test_features_tensor,
            "support": q_support_tensors,
            "row_indices": tf.convert_to_tensor(out_ids, dtype=tf.int32),
            "col_indices": tf.convert_to_tensor(choices_ids, dtype=tf.int32),
        }

        preds = predict_step(q_inputs)

        outs = tf.reshape(preds, (-1, 4))
        # average probability across all edges for each of the 4 choices
        mean_outs = tf.reduce_mean(outs, axis=0)

        predicted_choice = tf.argmax(mean_outs).numpy()

        gt_choice = tf.argmax(
            tf.reduce_mean(tf.reshape(labels, (-1, 4)), axis=0)
        ).numpy()

        num_processed += 1
        if predicted_choice == gt_choice:
            correct += 1

        print(f"[{num_processed}] Accuracy: {correct / num_processed:.4f}", end="\r")

    print(f"\nFinal FITB Accuracy: {correct / num_processed:.4f}")
    print(f"Best val score saved in log: {config['best_val_score']}")
    print(f"Last val score saved in log: {log['val']['acc'][-1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-k", type=int, default=1, help="K used for the variable number of edges case"
    )
    parser.add_argument(
        "-eo",
        "--expand_outfit",
        dest="expand_outfit",
        action="store_true",
        help="Expand the outfit nodes as well, rather than using them by default",
    )
    parser.add_argument(
        "-resampled",
        "--resampled",
        dest="resampled",
        action="store_true",
        help="Runs the test with the resampled FITB tasks (harder)",
    )
    parser.add_argument(
        "-subset",
        "--subset",
        dest="subset",
        action="store_true",
        help="Use only a subset of the nodes that form the outfit (3 of them) and use the others as connections",
    )
    parser.add_argument(
        "-lf", "--load_from", type=str, required=True, help="Model directory used."
    )
    args = parser.parse_args()
    test_fitb(vars(args))
