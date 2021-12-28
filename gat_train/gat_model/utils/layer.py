import numpy as np
import tensorflow as tf

conv1d = tf.layers.conv1d

def attn_head(seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False):
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)
        # [batch_size, node_size, emb_size]
        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)
        # [batch_size, node_size, out_sz]

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)  # [batch_size, node_size, 1]
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)  # [batch_size, node_size, 1]
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])
        # [batch_size, node_size, 1] + [batch_size, 1, node_size]  => [batch_size, node_size, node_size]
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)  # [batch_size, node_size, node_size]

        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, in_drop)

        vals = tf.matmul(coefs, seq_fts)
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
            else:
                seq_fts = ret + seq

        return activation(ret)  # activation

# Experimental sparse attention head (for running on datasets such as Pubmed)
# N.B. Because of limitations of current TF implementation, will work _only_ if batch_size = 1!
def sp_attn_head(seq, out_sz, adj_mat, activation, nb_nodes, in_drop=0.0, coef_drop=0.0, residual=False):
    with tf.name_scope('sp_attn'):
        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)
        # simplest self-attention possible
        # seq: [batch_size, nb_nodes, emb_size]
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)  # (batch_size, nb_nodes, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)   # (batch_size, nb_nodes, 1)
        adj_mat = tf.sparse_reshape(adj_mat, [1, nb_nodes, nb_nodes])
        logits = tf.sparse_add(adj_mat * f_1, adj_mat * tf.transpose(f_2, [0, 2, 1]))
        # adj_mat * f_1 =>
        # (batch_size, nb_nodes, nb_nodes)*(batch_size, nb_nodes, 1) => (batch_size, nb_nodes, nb_nodes)
        # adj_mat * tf.transpose(f_2, [0, 2, 1]) =>
        # (batch_size, nb_nodes, nb_nodes) * (batch_size, 1, nb_nodes) => (batch_size, nb_nodes, nb_nodes)

        # print("logits.shape", logits.shape)  # logits.shape (bat, nb_nodes, nb_nodes)

        lrelu = tf.SparseTensor(indices=logits.indices,
                values=tf.nn.leaky_relu(logits.values),
                dense_shape=logits.dense_shape)

        coefs = tf.sparse_softmax(lrelu)
        # print('coefs.shape', coefs.shape)  # [batch_size, node_size, node_size]
        coefs = tf.SparseTensor(indices=coefs.indices,
                values=tf.nn.dropout(coefs.values, coef_drop),
                dense_shape=coefs.dense_shape)
        seq_fts = tf.nn.dropout(seq_fts, in_drop)

        # As tf.sparse_tensor_dense_matmul expects its arguments to have rank-2,
        # here we make an assumption that our input is of batch size 1, and reshape appropriately.
        # The method will fail in all other cases!
        coefs = tf.sparse_reshape(coefs, [nb_nodes, nb_nodes])
        seq_fts = tf.squeeze(seq_fts)

        vals = tf.sparse_tensor_dense_matmul(coefs, seq_fts)
        vals = tf.expand_dims(vals, axis=0)
        node_num = seq.shape[1]
        vals.set_shape([1, node_num, out_sz])
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
            else:
                seq_fts = ret + seq
        return activation(ret)  # activation

