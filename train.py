import argparse
import time
import tensorflow as tf
import numpy as np
import os
import json
import shutil  # Added import

from utils import (
    get_degree_supports,
    normalize_nonsym_adj,
    write_log,
    support_dropout,
    to_tf_sparse_tensor,
)
from model.CompatibilityGAE import CompatibilityGAE
from dataloaders import DataLoaderPolyvore, DataLoaderFashionGen, DataLoaderAmazon

# Set random seed
seed = int(time.time())
np.random.seed(seed)
tf.random.set_seed(seed)

# Settings
ap = argparse.ArgumentParser()
ap.add_argument(
    "-d",
    "--dataset",
    type=str,
    default="polyvore",
    choices=["fashiongen", "polyvore", "amazon"],
    help="Dataset string.",
)
ap.add_argument(
    "-lr", "--learning_rate", type=float, default=0.001, help="Learning rate"
)
ap.add_argument("-wd", "--weight_decay", type=float, default=0.0, help="Weight decay")
ap.add_argument(
    "-e", "--epochs", type=int, default=4000, help="Number of training epochs"
)
ap.add_argument(
    "-hi",
    "--hidden",
    type=int,
    nargs="+",
    default=[350, 350, 350],
    help="Number of hidden units in the GCN layers.",
)
ap.add_argument("-do", "--dropout", type=float, default=0.5, help="Dropout fraction")
ap.add_argument(
    "-deg",
    "--degree",
    type=int,
    default=1,
    help="Degree of the convolution (Number of supports)",
)
ap.add_argument(
    "-sdir",
    "--summaries_dir",
    type=str,
    default="logs/",
    help="Directory for saving summaries",
)
ap.add_argument(
    "-cdir",
    "--checkpoint_dir",
    type=str,
    default="checkpoints/",
    help="Directory for saving model checkpoints",
)
# --- ADDED ARGUMENT FOR RESUMING ---
ap.add_argument(
    "-r",
    "--resume",
    type=str,
    default=None,
    help="Path to the weights file to resume training from.",
)
# ------------------------------------
ap.add_argument(
    "-sup_do",
    "--support_dropout",
    type=float,
    default=0.15,
    help="Dropout on support matrices",
)
fp = ap.add_mutually_exclusive_group(required=False)
fp.add_argument(
    "-bn",
    "--batch_norm",
    dest="batch_norm",
    action="store_true",
    help="Turn on batchnorm",
)
fp.add_argument(
    "-no_bn",
    "--no_batch_norm",
    dest="batch_norm",
    action="store_false",
    help="Turn off batchnorm",
)
ap.set_defaults(batch_norm=True)
ap.add_argument(
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
    help="Amazon dataset subset.",
)
args = vars(ap.parse_args())

print("Settings:", args, "\n")

# Define parameters
DATASET = args["dataset"]
NB_EPOCH = args["epochs"]
DO = args["dropout"]
HIDDEN = args["hidden"]
LR = args["learning_rate"]
WD = args["weight_decay"]
SUMMARIESDIR = args["summaries_dir"]
CHECKPOINTDIR = args["checkpoint_dir"]
DEGREE = args["degree"]
BATCH_NORM = args["batch_norm"]
SUP_DO = args["support_dropout"]
RESUME_PATH = args["resume"]  # Get the resume path
ADJ_SELF_CONNECTIONS = True
VERBOSE = True

# Prepare data loader
if DATASET == "fashiongen":
    dl = DataLoaderFashionGen()
elif DATASET == "polyvore":
    dl = DataLoaderPolyvore()
elif DATASET == "amazon":
    dl = DataLoaderAmazon(cat_rel=args["amz_data"])
else:
    raise NotImplementedError(f"Dataloader for {DATASET} not implemented.")

train_features, adj_train, train_labels, train_r_indices, train_c_indices = (
    dl.get_phase("train")
)
val_features, adj_val, val_labels, val_r_indices, val_c_indices = dl.get_phase("valid")

if DATASET == "amazon":
    # For Amazon, val/test features are the same as train features
    val_features = train_features

train_features, mean, std = dl.normalize_features(train_features, get_moments=True)
val_features = dl.normalize_features(val_features, mean=mean, std=std)

# Create directories
if not os.path.exists(SUMMARIESDIR):
    os.makedirs(SUMMARIESDIR)
if not os.path.exists(CHECKPOINTDIR):
    os.makedirs(CHECKPOINTDIR)

run_id = str(len(os.listdir(CHECKPOINTDIR)))
SUMMARIESDIR = os.path.join(SUMMARIESDIR, run_id)
CHECKPOINTDIR = os.path.join(CHECKPOINTDIR, run_id)
os.makedirs(SUMMARIESDIR, exist_ok=True)
os.makedirs(CHECKPOINTDIR, exist_ok=True)

log_file = os.path.join(SUMMARIESDIR, "log.json")
log_data = {"train": {"loss": [], "acc": []}, "val": {"loss": [], "acc": []}}

# Prepare model inputs
train_support = get_degree_supports(
    adj_train, DEGREE, adj_self_con=ADJ_SELF_CONNECTIONS
)
val_support = get_degree_supports(adj_val, DEGREE, adj_self_con=ADJ_SELF_CONNECTIONS)

for i in range(1, len(train_support)):
    train_support[i] = normalize_nonsym_adj(train_support[i])
    val_support[i] = normalize_nonsym_adj(val_support[i])

num_support = len(train_support)

# Convert to tensors
train_features_tensor = tf.convert_to_tensor(train_features, dtype=tf.float32)
train_labels_tensor = tf.convert_to_tensor(train_labels, dtype=tf.float32)
train_r_indices_tensor = tf.convert_to_tensor(train_r_indices, dtype=tf.int32)
train_c_indices_tensor = tf.convert_to_tensor(train_c_indices, dtype=tf.int32)
base_train_support_tensors = [to_tf_sparse_tensor(s) for s in train_support]

val_features_tensor = tf.convert_to_tensor(val_features, dtype=tf.float32)
val_labels_tensor = tf.convert_to_tensor(val_labels, dtype=tf.float32)
val_r_indices_tensor = tf.convert_to_tensor(val_r_indices, dtype=tf.int32)
val_c_indices_tensor = tf.convert_to_tensor(val_c_indices, dtype=tf.int32)
val_support_tensors = [to_tf_sparse_tensor(s) for s in val_support]

# Initialize model, optimizer, loss, and metrics
model = CompatibilityGAE(
    input_dim=train_features.shape[1],
    hidden=HIDDEN,
    num_support=num_support,
    batch_norm=BATCH_NORM,
    dropout_rate=DO,
)

# --- ADDED LOGIC TO LOAD WEIGHTS ---
if RESUME_PATH:
    if os.path.exists(RESUME_PATH):
        print(f"\n--- Resuming training from checkpoint: {RESUME_PATH} ---\n")
        model.load_weights(RESUME_PATH)
    else:
        print(
            f"\n--- WARNING: Checkpoint file not found at '{RESUME_PATH}'. Starting from scratch. ---\n"
        )
# ------------------------------------

optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
train_acc_metric = tf.keras.metrics.BinaryAccuracy()
val_acc_metric = tf.keras.metrics.BinaryAccuracy()


@tf.function
def train_step(features, support, labels, r_indices, c_indices):
    with tf.GradientTape() as tape:
        predictions = model(
            {
                "node_features": features,
                "support": support,
                "row_indices": r_indices,
                "col_indices": c_indices,
            },
            training=True,
        )
        loss = loss_fn(labels, predictions)
        if WD > 0:
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_weights])
            loss += WD * l2_loss

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_acc_metric.update_state(labels, predictions)
    return loss


@tf.function
def val_step(features, support, labels, r_indices, c_indices):
    predictions = model(
        {
            "node_features": features,
            "support": support,
            "row_indices": r_indices,
            "col_indices": c_indices,
        },
        training=False,
    )
    loss = loss_fn(labels, predictions)
    val_acc_metric.update_state(labels, predictions)
    return loss


best_val_score = 0
best_epoch = 0

print("Training...")
for epoch in range(NB_EPOCH):
    t = time.time()

    current_train_support = base_train_support_tensors
    if SUP_DO > 0:
        # Apply support dropout dynamically
        temp_support = []
        temp_support.append(train_support[0])  # Keep identity matrix
        for i in range(1, len(train_support)):
            modified = support_dropout(train_support[i].copy(), SUP_DO, edge_drop=True)
            modified.data[...] = 1
            modified = normalize_nonsym_adj(modified)
            temp_support.append(modified)
        current_train_support = [to_tf_sparse_tensor(s) for s in temp_support]

    train_loss = train_step(
        train_features_tensor,
        current_train_support,
        train_labels_tensor,
        train_r_indices_tensor,
        train_c_indices_tensor,
    )
    train_acc = train_acc_metric.result()

    val_loss = val_step(
        val_features_tensor,
        val_support_tensors,
        val_labels_tensor,
        val_r_indices_tensor,
        val_c_indices_tensor,
    )
    val_acc = val_acc_metric.result()

    if VERBOSE:
        print(
            f"[*] Epoch: {epoch + 1:04d} train_loss={train_loss:.5f} train_acc={train_acc:.5f} "
            f"val_loss={val_loss:.5f} val_acc={val_acc:.5f} time={time.time() - t:.5f}s"
        )

    log_data["train"]["loss"].append(float(train_loss))
    log_data["train"]["acc"].append(float(train_acc))
    log_data["val"]["loss"].append(float(val_loss))
    log_data["val"]["acc"].append(float(val_acc))
    write_log(log_data, log_file)

    if val_acc > best_val_score:
        best_val_score = val_acc
        best_epoch = epoch
        model.save_weights(os.path.join(CHECKPOINTDIR, "best_epoch.weights.h5"))

    # Reset metrics at the end of each epoch
    train_acc_metric.reset_state()
    val_acc_metric.reset_state()

print("\nOptimization Finished!")
print(f"Best validation score = {best_val_score:.5f} at epoch {best_epoch}")

# Save final model
model.save_weights(os.path.join(CHECKPOINTDIR, "final_model.weights.h5"))

# Store results
results = args.copy()
results.update(
    {"best_val_score": float(best_val_score), "best_epoch": best_epoch, "seed": seed}
)
json_outfile = os.path.join(CHECKPOINTDIR, "results.json")
with open(json_outfile, "w") as outfile:
    json.dump(results, outfile)

# Also copy to summaries dir for compatibility with test scripts
shutil.copy(json_outfile, SUMMARIESDIR)
if os.path.exists(os.path.join(CHECKPOINTDIR, "best_epoch.weights.h5")):
    shutil.copy(os.path.join(CHECKPOINTDIR, "best_epoch.weights.h5"), SUMMARIESDIR)


print("\nFinal results saved to:", json_outfile)
