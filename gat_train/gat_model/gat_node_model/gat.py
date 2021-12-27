import numpy as np
import tensorflow as tf

from utils import layers
from gat_node_models.base_gattn import BaseGAttN


# hid_units = [128, 64, 64] # numbers of hidden units per each attention head in each layer
# n_heads = [1, 1, 1, 1]
class GAT(BaseGAttN):
    @staticmethod
    def inference(inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
            adj_mat, hid_units, n_heads, activation=tf.nn.elu, residual=False):

        attns = []
        # multi-head attention 1阶
        # input: [batch_size, node_size, emb_size]
        # hid_units = [128, 64, 64]
        for _ in range(n_heads[0]):
            attns.append(layers.sp_attn_head(inputs, adj_mat=adj_mat,
                out_sz=hid_units[0], activation=activation, nb_nodes = nb_nodes,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        h_1 = tf.concat(attns, axis=-1)
        middle_state = [h_1]

        # spare multi-head attention 2阶
        for i in range(1, len(hid_units)):
            h_old = h_1
            attns = []
            for _ in range(n_heads[i]):
                attns.append(layers.sp_attn_head(h_1, adj_mat=adj_mat,
                    out_sz=hid_units[i], activation=activation, nb_nodes = nb_nodes,
                    in_drop=ffd_drop, coef_drop=attn_drop, residual=residual))
            h_1 = tf.concat(attns, axis=-1)
            middle_state.append(h_1)
        h_1 = tf.concat(middle_state, axis=-1)
        out = []
        # attention self  3阶
        for i in range(n_heads[-1]):
            out.append(layers.sp_attn_head(h_1, adj_mat=adj_mat,
                out_sz=nb_classes, activation=lambda x: x, nb_nodes = nb_nodes,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=False))

        logits = tf.add_n(out) / n_heads[-1]

        return logits, h_1
