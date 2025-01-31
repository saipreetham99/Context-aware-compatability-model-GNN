import json
import time
import tensorflow as tf
import argparse
import numpy as np
import scipy.sparse as sp
from collections import namedtuple
import os

from utils import get_degree_supports, normalize_nonsym_adj, Graph, to_tf_sparse_tensor
from model.CompatibilityGAE import CompatibilityGAE
from dataloaders import DataLoaderAmazon


def test_amazon(args):
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

    NUMCLASSES = 2
    BN_AS_TRAIN = False
    ADJ_SELF_CONNECTIONS = True

    # evaluate in the specified version
    print(
        "Trained with {}, evaluating with {}".format(config["amz_data"], args.amz_data)
    )
    cat_rel = args.amz_data
    dp = DataLoaderAmazon(cat_rel=cat_rel)
    train_features, _, _, _, _ = dp.get_phase("train")
    _, adj_val, val_labels, val_r_indices, val_c_indices = dp.get_phase("valid")
    _, adj_test, test_labels, test_r_indices, test_c_indices = dp.get_phase("test")
    full_adj = dp.adj

    def norm_adj(adj_to_norm):
        return normalize_nonsym_adj(adj_to_norm)

    train_features, mean, std = dp.normalize_features(train_features, get_moments=True)

    val_support = get_degree_supports(
        adj_val, config["degree"], adj_self_con=ADJ_SELF_CONNECTIONS
    )
    test_support = get_degree_supports(
        adj_test, config["degree"], adj_self_con=ADJ_SELF_CONNECTIONS
    )

    for i in range(1, len(val_support)):
        val_support[i] = norm_adj(val_support[i])
        test_support[i] = norm_adj(test_support[i])

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
            "node_features": tf.convert_to_tensor(train_features, dtype=tf.float32),
            "support": [to_tf_sparse_tensor(s) for s in test_support],
            "row_indices": tf.convert_to_tensor(test_r_indices, dtype=tf.int32),
            "col_indices": tf.convert_to_tensor(test_c_indices, dtype=tf.int32),
        },
        training=False,
    )

    model.load_weights(checkpoint_path).expect_partial()
    print("Model weights restored from:", checkpoint_path)

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    val_acc_metric = tf.keras.metrics.BinaryAccuracy()
    test_acc_metric = tf.keras.metrics.BinaryAccuracy()

    # --- Validation Evaluation ---
    val_inputs = {
        "node_features": tf.convert_to_tensor(train_features, dtype=tf.float32),
        "support": [to_tf_sparse_tensor(s) for s in val_support],
        "row_indices": tf.convert_to_tensor(val_r_indices, dtype=tf.int32),
        "col_indices": tf.convert_to_tensor(val_c_indices, dtype=tf.int32),
    }
    val_labels_tensor = tf.convert_to_tensor(val_labels, dtype=tf.float32)

    val_preds = model(val_inputs, training=BN_AS_TRAIN)
    val_loss = loss_fn(val_labels_tensor, val_preds)
    val_acc_metric.update_state(val_labels_tensor, val_preds)
    val_acc = val_acc_metric.result().numpy()
    val_acc_metric.reset_states()

    print(
        "val_loss=",
        "{:.5f}".format(val_loss.numpy()),
        "val_acc=",
        "{:.5f}".format(val_acc),
    )

    # --- Test Evaluation ---
    test_inputs = {
        "node_features": tf.convert_to_tensor(train_features, dtype=tf.float32),
        "support": [to_tf_sparse_tensor(s) for s in test_support],
        "row_indices": tf.convert_to_tensor(test_r_indices, dtype=tf.int32),
        "col_indices": tf.convert_to_tensor(test_c_indices, dtype=tf.int32),
    }
    test_labels_tensor = tf.convert_to_tensor(test_labels, dtype=tf.float32)

    test_preds = model(test_inputs, training=BN_AS_TRAIN)
    test_loss = loss_fn(test_labels_tensor, test_preds)
    test_acc_metric.update_state(test_labels_tensor, test_preds)
    test_acc = test_acc_metric.result().numpy()
    test_acc_metric.reset_states()

    print(
        "test_loss=",
        "{:.5f}".format(test_loss.numpy()),
        "test_acc=",
        "{:.5f}".format(test_acc),
    )

    # --- K=0 Evaluation ---
    k_0_adj = sp.csr_matrix(adj_val.shape)
    k_0_support = get_degree_supports(
        k_0_adj, config["degree"], adj_self_con=ADJ_SELF_CONNECTIONS, verbose=False
    )
    for i in range(1, len(k_0_support)):
        k_0_support[i] = norm_adj(k_0_support[i])

    k_0_val_inputs = {
        **val_inputs,
        "support": [to_tf_sparse_tensor(s) for s in k_0_support],
    }
    k_0_val_preds = model(k_0_val_inputs, training=BN_AS_TRAIN)
    k_0_val_loss = loss_fn(val_labels_tensor, k_0_val_preds)
    val_acc_metric.update_state(val_labels_tensor, k_0_val_preds)
    k_0_val_acc = val_acc_metric.result().numpy()
    val_acc_metric.reset_states()

    print(
        "for k=0 val_loss=",
        "{:.5f}".format(k_0_val_loss.numpy()),
        "for k=0 val_acc=",
        "{:.5f}".format(k_0_val_acc),
    )

    k_0_test_inputs = {
        **test_inputs,
        "support": [to_tf_sparse_tensor(s) for s in k_0_support],
    }
    k_0_test_preds = model(k_0_test_inputs, training=BN_AS_TRAIN)
    k_0_test_loss = loss_fn(test_labels_tensor, k_0_test_preds)
    test_acc_metric.update_state(test_labels_tensor, k_0_test_preds)
    k_0_test_acc = test_acc_metric.result().numpy()
    test_acc_metric.reset_states()

    print(
        "for k=0 test_loss=",
        "{:.5f}".format(k_0_test_loss.numpy()),
        "for k=0 test_acc=",
        "{:.5f}".format(k_0_test_acc),
    )

    # --- K-edges Evaluation ---
    K = args.k
    if K >= 0:
        available_adj = dp.full_valid_adj + dp.full_train_adj
        available_adj = available_adj.tolil()
        for r, c in zip(test_r_indices, test_c_indices):
            available_adj[r, c] = 0
            available_adj[c, r] = 0
        available_adj = available_adj.tocsr()
        available_adj.eliminate_zeros()

        G = Graph(available_adj)
        get_edges_func = G.run_K_BFS

        new_adj = sp.lil_matrix(full_adj.shape)
        for r, c in zip(test_r_indices, test_c_indices):
            if K > 0:  # expand the edges
                nodes_to_expand = [r, c]
                for node in nodes_to_expand:
                    edges = get_edges_func(node, K)
                    for edge in edges:
                        i, j = edge
                        new_adj[i, j] = 1
                        new_adj[j, i] = 1

        new_adj = new_adj.tocsr()

        new_support = get_degree_supports(
            new_adj, config["degree"], adj_self_con=ADJ_SELF_CONNECTIONS, verbose=False
        )
        for i in range(1, len(new_support)):
            new_support[i] = norm_adj(new_support[i])

        k_test_inputs = {
            **test_inputs,
            "support": [to_tf_sparse_tensor(s) for s in new_support],
        }
        k_test_preds = model(k_test_inputs, training=BN_AS_TRAIN)
        k_test_loss = loss_fn(test_labels_tensor, k_test_preds)
        test_acc_metric.update_state(test_labels_tensor, k_test_preds)
        k_test_acc = test_acc_metric.result().numpy()
        test_acc_metric.reset_states()

        print(
            "for k={} test_loss={:.5f} test_acc=".format(K, k_test_loss.numpy()),
            "{:.5f}".format(k_test_acc),
        )

    print("Best val score saved in log: {}".format(config["best_val_score"]))
    print("Last val score saved in log: {}".format(log["val"]["acc"][-1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-k", type=int, default=0, help="K used for the variable number of edges case"
    )
    parser.add_argument(
        "-lf",
        "--load_from",
        type=str,
        required=True,
        help="Model directory to load from.",
    )
    parser.add_argument(
        "-amzd",
        "--amz_data",
        type=str,
        default="Men_bought_together",
        choices=[
            "Men_also_bought",
            "Women_also_bought",
            "Women_bought_together",
            "Men_bought_together",
        ],
        help="Dataset string.",
    )
    args = parser.parse_args()
    test_amazon(vars(args))
