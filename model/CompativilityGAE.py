import tensorflow as tf
from .layers import GCN, MLPDecoder


class CompatibilityGAE(tf.keras.Model):
    def __init__(
        self,
        input_dim,
        hidden,
        num_support,
        batch_norm=False,
        dropout_rate=0.5,
        **kwargs,
    ):
        super(CompatibilityGAE, self).__init__(**kwargs)

        self.hidden_layers_config = hidden
        self.num_support = num_support
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate

        self.gcn_layers = []
        current_dim = input_dim
        for i, hidden_dim in enumerate(self.hidden_layers_config):
            self.gcn_layers.append(
                GCN(
                    input_dim=current_dim,
                    output_dim=hidden_dim,
                    num_support=self.num_support,
                    batch_norm=self.batch_norm,
                    dropout=self.dropout_rate,
                    name=f"gcn_{i}",
                )
            )
            current_dim = hidden_dim

        self.decoder = MLPDecoder(
            input_dim=current_dim, n_out=1, dropout=0.0, name="mlp_decoder"
        )

    def call(self, inputs, training=False):
        """
        Forward pass for the CompatibilityGAE model.

        Args:
            inputs (dict): A dictionary containing the model inputs:
                - 'node_features': The input features for the nodes.
                - 'support': A list of support matrices (adjacency matrices).
                - 'row_indices': Row indices for the edges to be predicted.
                - 'col_indices': Column indices for the edges to be predicted.
            training (bool): A flag indicating whether the model is in training mode.

        Returns:
            tf.Tensor: The output predictions for the specified edges.
        """
        node_features = inputs["node_features"]
        support = inputs["support"]
        r_indices = inputs["row_indices"]
        c_indices = inputs["col_indices"]

        x = node_features
        for layer in self.gcn_layers:
            x = layer((x, support), training=training)

        # The decoder takes the final node embeddings and the edge indices
        output = self.decoder((x, r_indices, c_indices), training=training)

        return output
