"""This module provides the a softmax cross entropy loss for training FCN.

In order to train VGG first build the model and then feed apply vgg_fcn.up
to the loss. The loss function can be used in combination with any optimizer
(e.g. Adam) to finetune the whole model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def loss(logits, labels, num_classes, head=None):
    """Calculate the loss from the logits and the labels.

    Args:
      logits: tensor, float - [batch_size, width, height, num_classes].
          Use vgg_fcn.upscore as logits.
      labels: Labels tensor, int32 - [batch_size, width, height, num_classes].
          The ground truth of your data.
      head: numpy array - [num_classes]
          Weighting the loss of each class
          Optional: Prioritize some classes

    Returns:
      loss: Loss tensor of type float.
    """

    labels = tf.squeeze(labels, axis=[3])
    ambiguous_pixels = tf.not_equal(labels, 21)
    labels = tf.where(ambiguous_pixels, labels, tf.fill(dims=tf.shape(labels), value=1))
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=labels,
        name="entropy")
    cross_entropy = tf.multiply(cross_entropy, tf.cast(ambiguous_pixels, tf.float32))
    return tf.reduce_mean(cross_entropy)
    # with tf.name_scope('cross_entropy_loss'):
    #     softmax = tf.nn.softmax_cross_entropy_with_logits(
    #         labels=labels,
    #         logits=logits,
    #         dim=-1,
    #         name='cross_entropy'
    #     )
    #     return tf.reduce_mean(softmax)
    # with tf.name_scope('loss'):
    #     logits = tf.reshape(logits, (-1, num_classes))
    #     epsilon = tf.constant(value=1e-4)
    #     labels = tf.to_float(tf.reshape(labels, (-1, num_classes)))

    #     softmax = tf.nn.softmax(logits) + epsilon

    #     if head is not None:
    #         cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax),
    #                                        head), reduction_indices=[1])
    #     else:
    #         cross_entropy = -tf.reduce_sum(
    #             labels * tf.log(softmax), reduction_indices=[1])

    #     cross_entropy_mean = tf.reduce_mean(cross_entropy,
    #                                         name='xentropy_mean')
    #    tf.add_to_collection('losses', cross_entropy_mean)

    #   loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    # return loss


def get_eval_metric_ops(labels, predictions, params):
    """Return a dict of the evaluation Ops.
    Args:
        labels (Tensor): Labels tensor for training and evaluation.
        predictions (Tensor): Predictions Tensor.
    Returns:
        Dict of metric results keyed by name.
    """
    return {
        'MeanIOU': tf.metrics.mean_iou(
            labels=labels,
            predictions=predictions,
            num_classes=params.n_classes,
            name='mean_iou')
    }


