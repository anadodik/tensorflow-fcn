"""This module provides the a softmax cross entropy loss for training FCN.

In order to train VGG first build the model and then feed apply vgg_fcn.up
to the loss. The loss function can be used in combination with any optimizer
(e.g. Adam) to finetune the whole model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def loss(logits, labels, params, head=None):
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
    weights = 1.0
    if params.has_ambiguous:
        weights = tf.not_equal(labels, params.n_classes)
    return tf.losses.sparse_softmax_cross_entropy(
        labels=labels,
        logits=logits,
        weights=weights
    )


def get_eval_metric_ops(labels, predictions, params):
    """Return a dict of the evaluation Ops.
    Args:
        labels (Tensor): Labels tensor for training and evaluation.
        predictions (Tensor): Predictions Tensor.
    Returns:
        Dict of metric results keyed by name.
    """
    labels = tf.squeeze(labels, axis=[3])
    labels = tf.reshape(labels, [tf.shape(labels)[0], -1])
    predictions = tf.reshape(predictions, [tf.shape(predictions)[0], -1])
    weights = None
    if params.has_ambiguous:
        weights = tf.not_equal(labels, params.n_classes)
        # labels = tf.where(ambiguous_pixels, labels, tf.cast(predictions, tf.int32))
    return {
        'MeanIOU': tf.metrics.mean_iou(
            labels=labels,
            predictions=predictions,
            num_classes=params.n_classes,
            weights=weights,
            name='mean_iou'),
        'Accuracy': tf.metrics.accuracy(
            labels=labels,
            predictions=predictions,
            weights=weights,
            name='accuracy'),
        'Precision': tf.metrics.precision(
            labels=labels,
            predictions=predictions,
            weights=weights,
            name='precision'),
        'Recall': tf.metrics.recall(
            labels=labels,
            predictions=predictions,
            weights=weights,
            name='precision'),
        'MeanPerClassAccuracy': tf.metrics.mean_per_class_accuracy(
            labels=labels,
            predictions=predictions,
            num_classes=params.n_classes,
            weights=weights,
            name='mean_per_class_accuracy'),
    }


