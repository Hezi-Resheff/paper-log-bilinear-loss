
import numpy as np
import tensorflow as tf
from keras import backend as K


def loss_function_generator(conf_mat, log=False, alpha=.5):
    """
    Generate Bilinear/Log-Bilinear loss functions combined with the rgular cross-entorpy loss
    (1 - alpha)*cross_entropy_loss + alpha*bilinar/log-bilinar

    :param conf_mat: np.Array
        all positive confusion matrix. A higher value in [i, j] indicates a higher penalty for making the mistake of
        classifying an example really of class i, as class j (i.e. placing weight there, since the output is a
        probability vector).

    :param log: bool
        generate the log-blinear loss?

    :param alpha: float
        the trade-off paramter between the cross-entropy and bilinear/log-bilinear parts of the loss

    :return: lambda
        f: y_true, y_pred -> loss
    """

    # Just to be sure -- get rid of the diagonal part of the conf-mat
    conf_mat -= np.eye(conf_mat.shape[0]) * np.diag(conf_mat)

    # Need a tf.constant version of the conf mat
    cm = tf.constant(conf_mat)
    I = tf.constant(np.eye(conf_mat.shape[0]), dtype=np.float32)

    # The regular cross-entropy loss
    diagonal_loss = lambda y_true, y_pred: -K.mean(K.batch_dot(K.expand_dims(K.dot(y_true, I), 1), K.expand_dims(tf.log(y_pred + 1e-10), 2)))

    # The off-disgonal part of the loss -- how we weigh the error i->j
    if log:
        off_diagonal_loss = lambda y_true, y_pred: -K.mean(K.batch_dot(K.expand_dims(K.dot(y_true, cm), 1), K.expand_dims(tf.log(1 - y_pred + 1e-10), 2)))
    else:
        off_diagonal_loss = lambda y_true, y_pred: K.mean(K.batch_dot(K.expand_dims(K.dot(y_true, cm), 1), K.expand_dims(y_pred, 2)))

    return lambda y_true, y_pred: diagonal_loss(y_true, y_pred)*(1-alpha) + off_diagonal_loss(y_true, y_pred)*alpha


def bilinear_loss(cm, alpha=.5):
    return loss_function_generator(cm, log=False, alpha=alpha)


def log_bilinear_loss(cm, alpha=.5):
    return loss_function_generator(cm, log=True, alpha=alpha)