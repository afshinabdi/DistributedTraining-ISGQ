"""
    Implementation of the paper
    Dan Alistarh, Demjan Grubic, Ryota Tomioka, and Milan Vojnovic,
    'QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding', NIPS 2017

    QSGD quantizer:
    q, scale =  _qsgd_quantizer(x, s, seed=None, order=np.inf):

    Dequantizer:
    y = scale * q / s
"""

import numpy as np


def quantize(W, d=None, num_levels=2):
    """
    quantize input tensor W using QSGD method. the input tensor is reshaped into vecot form and divided into buckets of
    length d. it used maximum value of the vector as the scaling parameter for quantization. The output scale is such that
    by multiplying it with quantized values, the points will be reconstructed.
    :param W: input tensor to be quantizer
    :param d: bucket size
    :param num_levels: number of levels for quantizing |W|
    :return: quantized values and the scale
    """

    if d is None:
        d = W.size

    if W.size % d != 0:
        raise ValueError('the number of variables must be divisible by the bucket size (d).')

    w = np.reshape(W, newshape=(-1, d))
    norm_w = np.linalg.norm(w, ord=np.inf, axis=1) + np.finfo(float).eps
    
    # 1- normalize w
    sign_w = np.sign(w)
    y = np.abs(w) / norm_w[:, np.newaxis]

    # 2- initial quantization (q0(y) = l where y is in [l/s, (l+1)/s)
    q0 = np.floor(y * num_levels)  # an integer number in the range 0, 1, ..., s
    d = num_levels * y - q0  # d is the normalized distance of each point to the left boundary of the quantization interval

    # 3- create random binary numbers, b_i = 0 with probability (1-d) and b_i = 1 with probability d
    b = np.zeros(shape=w.shape)
    b[np.random.random(size=w.shape) < d] = 1

    q = sign_w * (q0 + b)
    scale = norm_w / num_levels

    Q = np.reshape(q, newshape=W.shape).astype(np.int)

    return Q, scale


def dequantize(Q, scale, d=None):
    """
    dequantize the received quantized values, usign the bucket size d and scales
    :param Q: quantized values
    :param scale: scale to multiply to the quantized values to reconstruct the original data
    :param d: bucket size
    :return: ndarray of the same shape as Q, dequantized values
    """

    if d is None:
        d = Q.size

    if Q.size % d != 0:
        raise ValueError('the number of variables must be divisible by the bucket size (d).')

    if d == Q.size:
        W = scale[0] * Q
    else:
        q = np.reshape(Q, (-1, d))
        w = q * scale[:, np.newaxis]

        W = np.reshape(w, newshape=Q.shape)

    return W
