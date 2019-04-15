import tensorflow as tf
class SparseFactorizationMachine(object):
    def __init__(self, model_name="sparse_fm"):
        self.model_name = model_name

    def build(self, features, labels, mode, params):
        print("export features {0}".format(features))
        print(mode)
        if mode == tf.estimator.ModeKeys.PREDICT:
            sp_indexes = tf.SparseTensor(indices=features['DeserializeSparse:0'],
                         values=features['DeserializeSparse:1'],
                         dense_shape=features['DeserializeSparse:2'])
            sp_vals = tf.SparseTensor(indices=features['DeserializeSparse_1:0'],
                                      values=features['DeserializeSparse_1:1'],
                                      dense_shape=features['DeserializeSparse_1:2'])
        if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
            sp_indexes = features['feature_ids']
            sp_vals = features['feature_vals']
            print("sp: {0}, {1}".format(sp_indexes, sp_vals))
        batch_size = params["batch_size"]
        feature_max_num = params["feature_max_num"]
        optimizer_type = params["optimizer_type"]
        factor_vec_size = params["factor_size"]

        bias = tf.get_variable(name="b", shape=[1], initializer=tf.glorot_normal_initializer())
        w_first_order = tf.get_variable(name='w_first_order', shape=[feature_max_num, 1], initializer=tf.glorot_normal_initializer())
        linear_part = tf.nn.embedding_lookup_sparse(w_first_order, sp_indexes, sp_vals, combiner="sum") + bias
        
        w_second_order = tf.get_variable(name='w_second_order', shape=[feature_max_num, factor_vec_size], initializer=tf.glorot_normal_initializer())
        embedding = tf.nn.embedding_lookup_sparse(w_second_order, sp_indexes, sp_vals, combiner="sum")
        embedding_square = tf.nn.embedding_lookup_sparse(tf.square(w_second_order), sp_indexes, tf.square(sp_vals), combiner="sum")
        sum_square = tf.square(embedding)
        second_part = 0.5*tf.reduce_sum(tf.subtract(sum_square, embedding_square), 1)
        y_hat = linear_part + tf.expand_dims(second_part, -1)
        predictions = tf.sigmoid(y_hat)
        print "y_hat: {0}, second_part: {1}, linear_part: {2}".format(y_hat, second_part, linear_part)
        pred = {"prob": predictions}
        
        export_outputs = {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(predictions)
        }
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode, 
                predictions=predictions,
                export_outputs=export_outputs)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=tf.squeeze(y_hat)))
        if optimizer_type == "sgd":
            opt = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])
        elif optimizer_type == "ftrl":
            opt = tf.train.FtrlOptimizer(learning_rate=params['learning_rate'],)
        elif optimizer_type == "adam":
            opt = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
        elif optimizer_type == "momentum":
            opt = tf.train.MomentumOptimizer(learning_rate=params['learning_rate'], momentum=params['momentum'])
        train_step = opt.minimize(loss,global_step=tf.train.get_global_step())
        eval_metric_ops = {
            "auc" : tf.metrics.auc(labels, predictions)
        }

        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss, train_op=train_step)
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss, eval_metric_ops=eval_metric_ops)