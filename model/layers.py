# file: model/layers.py
from .initializations import uniform_init, he_init
import tensorflow as tf


class GCN(tf.keras.layers.Layer):
    """Graph convolution layer for multiple degree adjacencies"""

    def __init__(
        self,
        input_dim,
        output_dim,
        num_support,
        dropout=0.0,
        act=tf.nn.relu,
        bias=True,
        batch_norm=False,
        init="def",
        **kwargs,
    ):
        super(GCN, self).__init__(**kwargs)

        self.act = act
        self.bias = bias
        self.batch_norm = batch_norm
        self.dropout_rate = dropout

        self.dropout_layer = tf.keras.layers.Dropout(self.dropout_rate)
        if self.batch_norm:
            self.batch_norm_layer = tf.keras.layers.BatchNormalization()

        self.weights_list = []
        for i in range(num_support):
            if init == "he":
                initializer = he_init
            else:
                initializer = uniform_init

            self.weights_list.append(
                self.add_weight(
                    shape=(input_dim, output_dim),
                    initializer=initializer,
                    name=f"weights_{i}",
                )
            )

        if self.bias:
            self.bias_weight = self.add_weight(
                shape=(output_dim,), initializer="zeros", name="bias"
            )

    def call(self, inputs, training=False):
        x, support = inputs
        x = self.dropout_layer(x, training=training)

        supports_out = []
        for i in range(len(support)):
            w = self.weights_list[i]
            pre_sup = tf.matmul(x, w)
            sup_out = tf.sparse.sparse_dense_matmul(support[i], pre_sup)
            supports_out.append(sup_out)

        output = tf.add_n(supports_out)

        if self.bias:
            output += self.bias_weight

        if self.batch_norm:
            output = self.batch_norm_layer(output, training=training)

        return self.act(output)


class MLPDecoder(tf.keras.layers.Layer):
    """
    MLP-based decoder model layer for edge-prediction.
    """

    def __init__(
        self, input_dim, dropout=0.0, act=lambda x: x, n_out=1, use_bias=True, **kwargs
    ):
        super(MLPDecoder, self).__init__(**kwargs)

        self.act = act
        self.use_bias = use_bias
        self.n_out = n_out
        self.dropout_rate = dropout

        self.dropout_layer = tf.keras.layers.Dropout(self.dropout_rate)

        self.w = self.add_weight(
            shape=(input_dim, n_out), initializer=uniform_init, name="weights"
        )
        if self.use_bias:
            self.b = self.add_weight(shape=(n_out,), initializer="zeros", name="bias")

    def call(self, inputs, training=False):
        node_inputs, r_indices, c_indices = inputs
        node_inputs = self.dropout_layer(node_inputs, training=training)

        row_inputs = tf.gather(node_inputs, r_indices)
        col_inputs = tf.gather(node_inputs, c_indices)

        diff = tf.abs(row_inputs - col_inputs)
        outputs = tf.matmul(diff, self.w)

        if self.use_bias:
            outputs += self.b

        if self.n_out == 1:
            outputs = tf.squeeze(outputs, axis=-1)

        return self.act(outputs)
