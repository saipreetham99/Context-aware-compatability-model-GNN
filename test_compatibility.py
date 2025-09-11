import json
import tensorflow as tf
import argparse
import numpy as np
import scipy.sparse as sp
import time
from sklearn.metrics import roc_auc_score
from collections import namedtuple
import os

from utils import get_degree_supports, normalize_nonsym_adj, Graph, to_tf_sparse_tensor
from model.CompatibilityGAE import CompatibilityGAE
from dataloaders import DataLoaderPolyvore, DataLoaderFashionGen


def compute_auc(preds, labels):
    return roc_auc_score(labels.astype(int), preds)


def test_compatibility(args):
    args_dict = args
    args = namedtuple("Args", args_dict.keys())(*args_dict.values())

    load_from = args.load_from
    config_file = os.path.join(load_from, "results.json")
    log_file = os.path.join(load_from, "log.json")
    checkpoint_path = os.path.join(load_from, "best_epoch.weights.h5")

    with open(config_file) as f:
        config = json.load(f)
    with open(log_file) as f:
        log = json.load(f)

    # Dataloader
    DATASET = config["dataset"]
    print("initializing dataloader...")
    if DATASET == "polyvore":
        dl = DataLoaderPolyvore()
    elif DATASET == "fashiongen":
        dl = DataLoaderFashionGen()
    else:
        raise NotImplementedError(f"A data loader for dataset {DATASET} does not exist")

    train_features, _, _, _, _ = dl.get_phase("train")
    _, adj_val, val_labels, val_r_indices, val_c_indices = dl.get_phase("valid")
    # FIX 1: Capture adj_test
    test_features, adj_test, _, _, _ = dl.get_phase("test")
    dl.setup_test_compatibility(resampled=args.resampled)

    BN_AS_TRAIN = False
    ADJ_SELF_CONNECTIONS = True

    def norm_adj(adj_to_norm):
        return normalize_nonsym_adj(adj_to_norm)

    _, mean, std = dl.normalize_features(train_features, get_moments=True)
    test_features = dl.normalize_features(test_features, mean=mean, std=std)

    val_support = get_degree_supports(
        adj_val, config["degree"], adj_self_con=ADJ_SELF_CONNECTIONS
    )
    for i in range(1, len(val_support)):
        val_support[i] = norm_adj(val_support[i])

    # FIX 2: Create a test_support with the correct dimensions from adj_test
    print(f"Computing adj matrices up to {config['degree']}th degree")
    test_support = get_degree_supports(
        adj_test, config["degree"], adj_self_con=ADJ_SELF_CONNECTIONS
    )
    for i in range(1, len(test_support)):
        test_support[i] = norm_adj(test_support[i])

    num_support = len(val_support)

    model = CompatibilityGAE(
        input_dim=train_features.shape[1],
        hidden=config["hidden"],
        num_support=num_support,
        batch_norm=config["batch_norm"],
        dropout_rate=0.0,
    )

    # FIX 3: Use test_support to build the model with correct shapes
    _ = model(
        {
            "node_features": tf.convert_to_tensor(test_features, dtype=tf.float32),
            "support": [to_tf_sparse_tensor(s) for s in test_support],
            "row_indices": tf.convert_to_tensor(val_r_indices, dtype=tf.int32),
            "col_indices": tf.convert_to_tensor(val_c_indices, dtype=tf.int32),
        },
        training=False,
    )

    model.load_weights(checkpoint_path)
    print("Model weights restored from:", checkpoint_path)

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    val_acc_metric = tf.keras.metrics.BinaryAccuracy()

    @tf.function
    def eval_step(inputs):
        return tf.nn.sigmoid(model(inputs, training=BN_AS_TRAIN))

    def eval_model():
        val_inputs = {
            "node_features": tf.convert_to_tensor(
                dl.normalize_features(dl.valid_features, mean=mean, std=std),
                dtype=tf.float32,
            ),
            "support": [to_tf_sparse_tensor(s) for s in val_support],
            "row_indices": tf.convert_to_tensor(val_r_indices, dtype=tf.int32),
            "col_indices": tf.convert_to_tensor(val_c_indices, dtype=tf.int32),
        }
        val_labels_tensor = tf.convert_to_tensor(val_labels, dtype=tf.float32)

        val_preds = model(val_inputs, training=BN_AS_TRAIN)
        val_loss = loss_fn(val_labels_tensor, val_preds)
        val_acc_metric.update_state(val_labels_tensor, val_preds)
        val_acc = val_acc_metric.result().numpy()
        val_acc_metric.reset_state()

        print(
            "val_loss=",
            "{:.5f}".format(val_loss.numpy()),
            "val_acc=",
            "{:.5f}".format(val_acc),
        )

    eval_model()

    count = 0
    preds = []
    labels = []

    K = args.k
    for outfit in dl.comp_outfits:
        before_item = time.time()
        items, score = outfit

        if args.subset:
            if len(items) > 3:
                items = np.random.choice(items, 3, replace=False).tolist()

        query_r, query_c = [], []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                query_r.append(items[i])
                query_c.append(items[j])

        if not query_r:  # Skip if outfit has less than 2 items
            continue

        new_adj = sp.csr_matrix((test_features.shape[0], test_features.shape[0]))
        if K > 0:
            available_adj = dl.test_adj.copy().tolil()
            for r, c in zip(query_r, query_c):
                available_adj[r, c] = 0
                available_adj[c, r] = 0

            available_adj = available_adj.tocsr()
            available_adj.eliminate_zeros()

            G = Graph(available_adj)
            new_adj = new_adj.tolil()
            nodes_to_expand = np.unique(items)
            for node in nodes_to_expand:
                edges = G.run_K_BFS(node, K)
                for u, v in edges:
                    new_adj[u, v] = 1
                    new_adj[v, u] = 1
            new_adj = new_adj.tocsr()

        new_support = get_degree_supports(
            new_adj, config["degree"], adj_self_con=ADJ_SELF_CONNECTIONS, verbose=False
        )
        for i in range(1, len(new_support)):
            new_support[i] = norm_adj(new_support[i])

        eval_inputs = {
            "node_features": test_features,
            "support": [to_tf_sparse_tensor(s) for s in new_support],
            "row_indices": np.array(query_r, dtype=np.int32),
            "col_indices": np.array(query_c, dtype=np.int32),
        }

        pred_scores = eval_step(eval_inputs)
        predicted_score = tf.reduce_mean(pred_scores).numpy()

        print(
            f"[{count}] Mean score: {predicted_score:.4f}, Label: {score}, Elapsed: {time.time() - before_item:.4f}s"
        )
        count += 1

        preds.append(predicted_score)
        labels.append(score)

    preds = np.array(preds)
    labels = np.array(labels)

    AUC = compute_auc(preds, labels)

    eval_model()
    print(f"The AUC compat score is: {AUC}")
    print(f"Best val score saved in log: {config['best_val_score']}")
    print(f"Last val score saved in log: {log['val']['acc'][-1]}")

    if np.any(labels):
        print(f"mean positive prediction: {preds[labels.astype(bool)].mean()}")
    if np.any(~labels.astype(bool)):
        print(f"mean negative prediction: {preds[~labels.astype(bool)].mean()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-lf", "--load_from", type=str, required=True, help="Model directory used."
    )
    parser.add_argument(
        "-subset",
        "--subset",
        dest="subset",
        action="store_true",
        help="Use only a subset of the nodes that form the outfit (3 of them) and use the others as connections",
    )
    parser.add_argument(
        "-resampled",
        "--resampled",
        dest="resampled",
        action="store_true",
        help="Use the resampled test, where the invalid outfits are harder.",
    )
    parser.add_argument(
        "-k", type=int, default=1, help="K used for the variable number of edges case"
    )
    args = parser.parse_args()
    test_compatibility(vars(args))
