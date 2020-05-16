import numpy as np
import tensorflow as tf


def quantize(W, num_levels, bucket_size, seed):
    """
    the input tensor is reshaped into vector form and divided into buckets of length d. it uses maximum value of the vector as the scaling parameter for quantization. The output scale is such that by multiplying it with quantized values, the points will be reconstructed.
    :param W: input tensor to be quantizer
    :param bucket_size: bucket size
    :param num_levels: number of levels for quantizing W, output will be in the range [-num_levels, ..., +num_levels]
    :return: quantized values and the scale
    """

    w = tf.reshape(W, shape=[-1, bucket_size])
    w_shape = tf.shape(w)

    # 1- normalize w to become in [-num_levels, num_levels]
    max_w = tf.reduce_max(tf.abs(w), axis=1, keepdims=True) + 1e-12
    scale = max_w / num_levels
    y = w / scale

    # 2- generate dither, add it to y and then quantize
    u = tf.random.stateless_uniform(shape=w_shape, minval=-0.5, maxval=0.5, dtype=tf.float32, seed=[seed, 0])
    # an integer number in the range -num_levels, ..., num_levels
    q = tf.cast(tf.round(y + u), tf.int8)

    return q, scale


def dequantize(q, scale, num_levels, bucket_size, seed):
    """
    dequantize the received quantized values, usign the bucket size d and scales
    :param Q: quantized values
    :param scale: scale to multiply to the quantized values to reconstruct the original data
    :param bucket_size: bucket size
    :return: ndarray of the same shape as Q, dequantized values
    """

    w_shape = tf.shape(q)

    u = tf.random.stateless_uniform(shape=w_shape, minval=-0.5, maxval=0.5, dtype=tf.float32, seed=[seed, 0])
    w = tf.multiply((q - u), scale)

    return w