"""
    Implementation of different indirect quantization algorithms
"""

import numpy as np
import scipy.linalg as sla
import quadprog
import distributed_training.optimum_quantizer as opt_quantizer


class DeterministicISGQuantizer:
    """
        implementation of the indirect quantization, G=X' Y.
        naive: X and Y are quantized independently according to their expected distribution
        mv: First, X and Y are quantized independently according to their expected distribution.
            Next, the reconstruction points are optimized to minimize the error |G-X'Y|
        mvu: First, X and Y are quantized independently according to their expected distribution.
            Then, the reconstruction points are optimized such that |G-X'Y| is minimized subject to sum(G-X'Y)=0
    """

    # initialize the quantizer engine, the supported quantization levels and data models
    def __init__(self, num_levels=(2, 4, 8), models=('sn', 'sfn', 'u', 'su'), sparsity_thr=1e-6):
        self._num_levels = num_levels
        self._models = models
        self._quantizers = {}
        for t in self._models:
            q = opt_quantizer.OptimumQuantizer()
            q.initialize_quantizer(model=t, num_levels=self._num_levels, sparsity_thr=sparsity_thr)
            self._quantizers[t] = q

    # quantize the input signals based on their distribution model, number of quantization levels and method
    def quantize(self, X, Y, model=('sfn', 'sn'), num_levels=(2, 2), method='naive', opt_iterations=1):

        qX, cX = self._quantize(X, model[0], num_levels[0])
        qY, cY = self._quantize(Y, model[1], num_levels[1])

        if method == 'mv':
            # optimize the centers of the bins for the minimum variance indirect quantizer
            G = np.matmul(X.transpose(), Y)
            # optimize for the centers of the quantizers
            for _ in range(opt_iterations):
                X_hat = cX[qX]
                optimize_centers_mviq(A=X_hat.transpose(), B=G, Q=qY, centers=cY, keep_sparsity=(model[0][0] == 's'))

                Y_hat = cY[qY]
                optimize_centers_mviq(
                    A=Y_hat.transpose(), B=G.transpose(), Q=qX, centers=cX, keep_sparsity=(model[1][0] == 's')
                )

        elif method == 'mvu':
            # optimize the centers of the bins for the minimum variance unbiased indirect quantizer
            G = np.matmul(X.transpose(), Y)
            # optimize for the centers of the quantizers
            for _ in range(opt_iterations):
                X_hat = cX[qX]
                optimize_centers_mvuiq(A=X_hat.transpose(), B=G, Q=qY, centers=cY, keep_sparsity=(model[0][0] == 's'))

                Y_hat = cY[qY]
                optimize_centers_mvuiq(
                    A=Y_hat.transpose(), B=G.transpose(), Q=qX, centers=cX, keep_sparsity=(model[1][0] == 's')
                )

        return qX, cX, qY, cY

    def _quantize(self, X, model, num_levels):
        # 1- if necessary, normalize x
        if model in ('uniform', 'u', 'sparse-uniform', 'su'):
            scale = 1.0
        else:
            scale = sla.norm(X) / np.sqrt(np.count_nonzero(np.abs(X) > 1e-10) + 1e-12)

        y = X / scale
        qX, cX = self._quantizers[model].quantize(y, num_levels)
        cX = scale * cX

        return qX, cX


# =============================================================================
def optimize_centers_mviq(A, B, Q, centers, keep_sparsity=True):
    """ minimize reconstruction error after weighting by matrix A
        min_{c_i} \|A.(\sum_i Q_i c_i) - B\|_F^2
    """
    num_levels = len(centers)
    thr = sla.norm(A) * 1e-6

    # 1- compute A*(Q==i) and store it. find the non-empty quantization bins in the process
    valid_idx = []
    AQ = [np.zeros(1) for _ in range(num_levels)]
    for i in range(num_levels):
        AQ[i] = np.matmul(A, Q == i)

        if (sla.norm(AQ[i]) >= thr) and ((centers[i] != 0) or not keep_sparsity):
            # check whether the i-th bin has any effect on the quantization performance and
            # do not consider sparse values (center=0)
            valid_idx += [i]

    if not valid_idx:
        return

    # 2- find the optimum reconstruction points for the non-empty quantization bins
    # 2.a- create matrix M, used in the optimization problem
    num_valid = len(valid_idx)
    M = np.zeros(shape=(num_valid, num_valid))
    e = np.zeros(shape=num_valid)
    for r in range(num_valid):
        for c in range(r, num_valid):
            # np.trace(np.matmul(AQ[valid_idx[c]].transpose(), AQ[valid_idx[r]]))
            M[r, c] = np.sum(AQ[valid_idx[c]] * AQ[valid_idx[r]])
            M[c, r] = M[r, c]

        # np.trace(np.matmul(B.transpose(), AQ[valid_idx[r]]))
        e[r] = np.sum(AQ[valid_idx[r]] * B)

    # 2.b- solve for Mx=e
    v = sla.lstsq(M, e)[0]

    # 3- copy the found center points
    centers[valid_idx] = v

    return centers


def optimize_centers_mvuiq(A, B, Q, centers, keep_sparsity=True):
    """ minimize reconstruction error after weighting by matrix A and make it unbiased
        min_{c_i} \|A.(\sum_i Q_i c_i) - B\|_F^2 such that sum(B-A(\sum_i Q_i c_i)) = 0
    """
    num_levels = len(centers)
    thr = sla.norm(A) * 1e-6

    # 1- compute A*(Q==i) and store it. find the non-empty quantization bins in the process
    valid_idx = []
    AQ = [np.zeros(1) for _ in range(num_levels)]
    for i in range(num_levels):
        AQ[i] = np.matmul(A, Q == i)

        if (sla.norm(AQ[i]) >= thr) and ((centers[i] != 0) or not keep_sparsity):
            # check whether the i-th bin has any effect on the quantization performance and
            # do not consider sparse values (center=0)
            valid_idx += [i]

    if not valid_idx:
        return

    # 2- find the optimum reconstruction points for the non-empty quantization bins
    # 2.a- create matrix M, used in the optimization problem
    num_valid = len(valid_idx)
    d = np.sum(B)
    f = np.zeros(num_valid)
    M = np.zeros(shape=(num_valid, num_valid))
    e = np.zeros(shape=num_valid)

    for r in range(num_valid):
        f[r] = np.sum(AQ[valid_idx[r]])
        for c in range(r, num_valid):
            # trace(AQ[valid_idx[c]].T @ AQ[valid_idx[r]])
            M[r, c] = np.sum(AQ[valid_idx[c]] * AQ[valid_idx[r]])
            M[c, r] = M[r, c]

        # trace(B.T @ AQ[valid_idx[r]])
        e[r] = np.sum(AQ[valid_idx[r]] * B)

    # 2.b- solve for min |Mx-e| such that fx=d
    if num_valid == 0:
        v = 0
    elif num_valid == 1:
        v = d / f[0]
    elif num_valid == 2:
        # for the special binary case, the solution can be found easily
        scale = sla.norm(f) + 1e-12
        f /= scale
        d /= scale
        u = np.array([-f[1], f[0]])
        a = (e - d * M.dot(f)).dot(u) / (M.dot(u).dot(u) + 1e-12)
        v = d * f + a * u
    else:
        # use quadratic programming (Goldfarb-Idnani algorithm) to solve the problem
        d = np.array([d]).astype(np.float)
        f = np.reshape(f, newshape=(-1, 1))
        v = quadprog.solve_qp(M, e, f, d, 1)[0]

    # 3- copy the found center points
    centers[valid_idx] = v

    return centers


# =============================================================================
class DitheredISGQuantizer:
    """
       implementation of the indirect quantization, G=X' Y, using random dithered quantization
    """
    def set_seed(self, seed):
        np.random.seed(seed)

    # dithered quantization
    def quantize(self, W, num_levels=2, sparse=True, bucket_size=None):
        """
        the input tensor is reshaped into vector form and divided into buckets of length d.
        it uses maximum value of the vector as the scaling parameter for quantization.
        The output scale is such that by multiplying it with quantized values, the points will be reconstructed.
        :param W: input tensor to be quantizer
        :param bucket_size: bucket size
        :param num_levels: number of levels for quantizing W, output will be in the range
                            [-num_levels, ..., +num_levels]
        :return: quantized values and the scale
        """

        if bucket_size is None:
            bucket_size = W.size

        if W.size % bucket_size != 0:
            raise ValueError('the number of variables must be divisible by the bucket size.')

        w = np.reshape(W, newshape=(-1, bucket_size))

        # 1- normalize w to become in [-num_levels, num_levels]
        max_w = np.amax(np.abs(w), axis=1) + 1e-12
        scale = max_w / num_levels
        y = w / scale[:, np.newaxis]

        # 2- generate dither, add it to y and then quantize
        u = np.random.uniform(-0.5, 0.5, size=y.shape)
        # an integer number in the range -num_levels or 0, ..., num_levels
        q = np.around(y + u).astype(np.int8)

        Q = np.reshape(q, newshape=W.shape)
        if sparse:
            # quantize 0 values separately
            Q[np.abs(W) < 1e-12] = num_levels + 1

        return Q, scale

    def dequantize(self, Q, scale, num_levels=2, sparse=True, bucket_size=None):
        """
        dequantize the received quantized values, usign the bucket size d and scales
        :param Q: quantized values
        :param scale: scale to multiply to the quantized values to reconstruct the original data
        :param bucket_size: bucket size
        :return: ndarray of the same shape as Q, dequantized values
        """

        if bucket_size is None:
            bucket_size = Q.size

        if Q.size % bucket_size != 0:
            raise ValueError('the number of variables must be divisible by the bucket size.')

        if bucket_size == Q.size:
            u = np.random.uniform(-0.5, 0.5, size=Q.shape)
            W = scale[0] * (Q - u)
        else:
            q = np.reshape(Q, (-1, bucket_size))
            u = np.random.uniform(-0.5, 0.5, size=q.shape)
            w = (q - u) * scale[:, np.newaxis]

            W = np.reshape(w, newshape=Q.shape)

        # check for the sparse dequantization
        if sparse:
            W[Q == (num_levels + 1)] = 0

        return W
