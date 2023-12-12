#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Roughly based on: http://stackoverflow.com/questions/11443302/compiling-numpy-with-openblas-integration

from __future__ import print_function

import timeit
import numpy as np
from time import time

# Let's take the randomness out of random numbers (for reproducibility)
np.random.seed(0)
# Matrix multiplication
def matrix_multi(A,B):
    np.dot(A, B)
    del A, B

# Vector multiplication
def vector_multi(C,D):
    for i in range(1000):
        np.dot(C, D)
    del C, D

# Singular Value Decomposition (SVD)
def svd(E):
    np.linalg.svd(E, full_matrices = False)
    del E

# Cholesky Decomposition
def cholesky_decomp(F):
    for i in range(10):
        np.linalg.cholesky(F)

# Eigendecomposition
def eigen_decomp(G):
    np.linalg.eig(G)

n = 10
size = 4096
A, B = np.random.random((size, size)), np.random.random((size, size))
C, D = np.random.random((size * 128,)), np.random.random((size * 128,))
E = np.random.random((int(size / 2), int(size / 4)))
F = np.random.random((int(size / 2), int(size / 2)))
F = np.dot(F, F.T)
G = np.random.random((int(size / 2), int(size / 2)))

# Prints average time per run
# Some runs use multiple repititons to have a more reasonable runtime
def bench():
    print('Numpy bench:')
    print("numpy matrix multiplication took on average: {}s".format(timeit.timeit('matrix_multi(A,B)', number=n, globals=globals())/n))
    print("numpy vector multiplication took on average: {}s".format(timeit.timeit('vector_multi(C,D)', number=n, globals=globals())/n))
    print("numpy svd took on average: {}s".format(timeit.timeit('svd(E)', number=n, globals=globals())/n))
    print("numpy cholesky decomposition took on average: {}s".format(timeit.timeit('cholesky_decomp(F)', number=n, globals=globals())/n))
    print("numpy eigen decomposition took on average: {}s".format(timeit.timeit('eigen_decomp(G)', number=2, globals=globals())/n))

