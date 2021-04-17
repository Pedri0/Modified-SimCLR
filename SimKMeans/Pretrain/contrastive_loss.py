import tensorflow.compat.v2 as tf
LARGE_NUM = 1e9


def contrastive_loss(hidden, hidden_norm=True, temperature=1.0, strategy=None):
    if hidden_norm:
        hidden = tf.math.l2_normalize(hidden, -1)
    hidden1, hidden2 = tf.split(hidden, 2, 0)
    batch_size = tf.shape(hidden1)[0]

    if strategy is not None:
        hidden1_large = tpu_cross_replica_concat(hidden1, strategy)
        hidden2_large = tpu_cross_replica_concat(hidden2, strategy)
        enlarged_batch_size = tf.shape(hidden1_large)[0]
        replica_context = tf.distribute.get_replica_context()
        replica_id = tf.cast(tf.cast(replica_context.replica_id_in_sync_group, tf.uint32), tf.int32)
        labels_idx = tf.range(batch_size) + replica_id * batch_size
        labels = tf.one_hot(labels_idx, enlarged_batch_size * 2)
        masks = tf.one_hot(labels_idx, enlarged_batch_size)

    else:
        hidden1_large = hidden1
        hidden2_large = hidden2
        labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
        masks = tf.one_hot(tf.range(batch_size), batch_size)
    
    similarity1 = tf.matmul(hidden1, hidden1_large, transpose_b=True) / temperature
    similarity1 = similarity1 - masks * LARGE_NUM
    similarity2 = tf.matmul(hidden2, hidden2_large, transpose_b=True) / temperature
    similarity2 = similarity2 - masks * LARGE_NUM
    similarity12 = tf.matmul(hidden1, hidden2_large, transpose_b=True) / temperature
    similarity21 = tf.matmul(hidden2, hidden1_large, transpose_b=True) / temperature
    
    loss_a = tf.nn.softmax_cross_entropy_with_logits(labels, tf.concat([similarity12, similarity1], 1))
    loss_b = tf.nn.softmax_cross_entropy_with_logits(labels, tf.concat([similarity21, similarity2], 1))
    loss = tf.reduce_mean(loss_a + loss_b)
    
    return loss, similarity12, labels

def tpu_cross_replica_concat(tensor, strategy=None):
    if strategy is None or strategy.num_replicas_in_sync <= 1:
        return tensor

    num_replicas = strategy.num_replicas_in_sync
    replica_context = tf.distribute.get_replica_context()
    with tf.name_scope('tpu_cross_replica_concat'):
        # This creates a tensor that is like the input tensor but has an added replica dimension as
        # the outermost dimension. On each replica it will contain the local values and zeros for all
        # other values that need to be fetched from other replicas.
        ext_tensor = tf.scatter_nd(indices=[[replica_context.replica_id_in_sync_group]],
                updates=[tensor], shape=tf.concat([[num_replicas], tf.shape(tensor)], axis=0))
        
        # As every value is only present on one replica and 0 in all others, adding
        # them all together will result in the full tensor on all replicas.
        ext_tensor = replica_context.all_reduce(tf.distribute.ReduceOp.SUM, ext_tensor)

        # Flatten the replica dimension.
        # The first dimension size will be: tensor.shape[0] * num_replicas
        # Using [-1] trick to support also scalar input.
        return tf.reshape(ext_tensor, [-1] + ext_tensor.shape.as_list()[2:])