#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Given a similarity matrix on sparse or numpy format, it creates a
Laplacian Embedding, for various type of graph Laplacian as well as
normalization.
So far the similarity is assumed to represent a fully connected graph.
'''
# try:
#     np
# except:
import numpy as np
# graphical utilities
import matplotlib.pyplot as plt
import time
# import sys
from scipy import sparse as sp
# from sinkhorn_knopp import sinkhorn_knopp as skp
"""
if 'sinkhorn' not in sys.modules:
    '''
    TODO: may not be true for every-one
    '''
    sys.path.append("/usr/local/lib/python3.5/dist-packages")
"""


def check_matrix(matrix):
    '''
    check that the matrix is square and symmetric and return its
    type.
    '''
    # check for squareness
    (n, m) = matrix.shape
    if n != m:
        raise ValueError('the matrix is not square')
    # check for symmetry
    if sp.issparse(matrix):
        if not (sp.find(matrix)[2] >= 0).all():
            raise ValueError('the matrix is not nonnegative')
        if not (abs(matrix-matrix.T) > 1e-10).nnz == 0:
            raise ValueError('specified similarity matrix is not\
                symmetric')
        return('sparse')
    else:
        if not np.all(matrix >= 0):
            raise ValueError('matrix is not nonnegative')
        if not np.allclose(matrix, matrix.T, atol=1e-6):
            raise ValueError('specified similarity matrix is not\
                symmetric.')
        return('dense')


def normalize(matrix, type_normalization):
    '''
    normalize a similarity matrix with coifman
    or sinkhorn normalization
    '''
    type_matrix = check_matrix(matrix)
    n = matrix.shape[0]
    if type_normalization is None:
        if type_matrix == 'sparse':
            return(sp.csr_matrix(matrix, copy=True, dtype='float64'))
        else:
            return(matrix)

    if type_normalization not in ['coifman', 'sinkhorn']:
        raise ValueError('the normalisation is not admissible')

    if type_normalization == 'coifman' and type_matrix == 'dense':
        W = matrix
        D = np.diag(1./matrix.dot(np.ones(n)))
        W = D.dot(matrix.dot(D))
        return(W)
    if type_normalization == 'coifman' and type_matrix == 'sparse':
        W = sp.csr_matrix(matrix, copy=True, dtype='float64')
        d = W.sum(axis=1).getA1()
        # d = np.array(sim_mat.sum(axis=0, dtype=sim_mat.dtype))[0]
        Dinv = sp.diags(1./d)
        W = Dinv.dot(W.dot(Dinv))
        return(W)
    if type_normalization == 'sinkhorn' and type_matrix == 'dense':
        sk = skp.SinkhornKnopp()
        normalized_matrix = sk.fit(matrix)
        return(normalized_matrix)
    else:
        raise ValueError('There is a problem in the choice of \
        type_normalization')


def make_laplacian_emb_dense(matrix,
                             dim,
                             type_laplacian='unnormalized',
                             type_normalization=None,
                             scaled=False, verbose=0):
        '''
        Create a vector of dimension dim for every node whose pair-wise
        distances are stored in similarity_matrix.
        compute first k eigenvectors of matrix of the similarity_matrix
        L = diag(W 1) - W

        INPUT:
            matrix should be a numpy matrix here.
        '''
        admissible_type_laplacian = ['symmetric',
                                     'random_walk',
                                     'unnormalized']
        if type_laplacian not in admissible_type_laplacian:
            raise ValueError('the parameter type_laplacian is\
             not well specified!')

        n = matrix.shape[0]
        n_vec = dim+1
        # normalize similarity_matrix
        W = normalize(matrix, type_normalization)

        # compute a laplacian
        if type_laplacian == 'random_walk':
            Laplacian = np.eye(n) - W.dot(np.diag(1/W.dot(np.ones(n))))
        elif type_laplacian == 'symmetric':
            Laplacian = np.eye(n) - np.diag(1/W.dot(np.ones(n))**(0.5)).\
                dot(W.dot(np.diag(1/W.dot(np.ones(n))**(0.5))))
        elif type_laplacian == 'unnormalized':
            Laplacian = np.diag(W.dot(np.ones(n))) - W

        # compute embedding
        [d, Vec] = np.linalg.eig(Laplacian)
        idx = d.argsort()
        d = d[idx]
        Vec = Vec[:, idx]
        if scaled:
            if scaled == 'CTD':
                dsqrt = np.sqrt(d[1:n_vec])
            else:
                dsqrt = np.arange(1, n_vec)
            V = Vec[:, 1:n_vec].dot(np.diag(1./dsqrt))
        else:
            V = Vec[:, 1:n_vec]
        '''
        TODO: check if it not supposed to be Vec[:,2:k].dot(np.diag(d[2:k]))
        '''
        return(V)


def make_laplacian_emb_sparse(matrix,
                              dim,
                              type_laplacian='unnormalized',
                              type_normalization=None,
                              scaled=False, verbose=0):
    '''
    Create a vector of dimension dim for every node whose pair-wise
    distances are stored in similarity_matrix.
    compute first k eigenvectors of matrix of the similarity_matrix
    L = diag(W 1) - W
    '''
    admissible_type_laplacian = ['symmetric', 'random_walk', 'unnormalized']
    if type_laplacian not in admissible_type_laplacian:
        raise ValueError('the parameter type_laplacian is not well specified!'
                         '(choose between symmetric, random_walk,'
                         ' or unnormalized)')

    n = matrix.shape[0]
    n_vec = dim + 1
    W = normalize(matrix, type_normalization)
    W.dtype = 'float64'
    # compute a laplacian
    lap = sp.csgraph.laplacian(W, normed=False, return_diag=False)

    if type_laplacian == 'random_walk':
        d = W.sum(axis=1).getA1()
        Dinv = sp.diags(1./d)
        lap = Dinv.dot(lap)
    elif type_laplacian == 'symmetric':
        lap = sp.csgraph.laplacian(W, normed=True, return_diag=False)

    # lap = sp.csr_matrix(lap, copy=True, dtype='float64')
    lap.dtype = 'float64'

    # Compute embedding
    t0 = time.time()
    # Largest eigenvalue first
    (evals_max, _) = sp.linalg.eigsh(lap, 1, which='LA', tol=1e-15, ncv=20)
    t1 = time.time()
    if verbose > 1:
        print('Computed largest eigenvalue of A in {}s.\n'.format(t1-t0))
    maxval = float(evals_max)
    # Then largest eigenvalue of minus laplacian
    # (it is faster to compute largest eigenvalues)
    m_lap = lap
    m_lap *= -1
    m_lap += sp.diags(np.tile(maxval, (n)))

    # m_lap = maxval * sp.identity(n) - lap
    # evec0 = np.ones(n)  # , v0=evec0)
    evals_small, evecs_small = sp.linalg.eigsh(m_lap, n_vec, which='LA', tol=1e-15)
    # eval_s, evec_s = eigsh(lap, n_vec, which='SM', v0=evec0)
    t2 = time.time()
    if verbose > 0:
        print('Computed Laplacian embedding of dim. {} in {}s.\n'.format(
            dim, t2-t1))
    evals_small = maxval - evals_small
    idx = np.array(evals_small).argsort()
    d = evals_small[idx]
    Vec = evecs_small[:, idx]
    V = Vec[:, 1:n_vec]
    if scaled:
        if scaled == 'CTD':
            dsqrt = np.sqrt(d[1:n_vec])
        else:
            dsqrt = np.arange(1, n_vec)
        V = V.dot(np.diag(1./dsqrt))
    return(V)


def make_laplacian_emb(matrix,
                       dim,
                       type_laplacian='unnormalized',
                       type_normalization=None,
                       scaled=False,
                       verb=0):
    type_matrix = check_matrix(matrix)
    if type_matrix == 'dense':
        V = make_laplacian_emb_dense(matrix,
                                     dim,
                                     type_laplacian,
                                     type_normalization,
                                     scaled, verbose=verb)
    else:
        V = make_laplacian_emb_sparse(matrix,
                                      dim,
                                      type_laplacian,
                                      type_normalization,
                                      scaled, verbose=verb)
    return(V)


def vizualize_embedding(embedding, title=None, perm=None):
        embedding = np.real(embedding)
        n = embedding.shape[0]
        plt.figure(int(time.clock()*100))
        if perm is not None:
            plt.scatter(embedding[:, 0], embedding[:, 1], c=perm)
        else:
            plt.scatter(embedding[:, 0], embedding[:, 1], c=np.arange(n))
        if title is not None:
            plt.title(title)


if __name__ == '__main__':
    # Embedding_graph()
    D = np.abs(np.random.normal(0, 1, (5, 5)))
    D = np.transpose(D) + D
    S = sp.csr_matrix(D)

    embeddi = make_laplacian_emb(D,
                                 3,
                                 type_laplacian='unnormalized',
                                 type_normalization='coifman',
                                 scaled=False)

    print(embeddi.shape)
    vizualize_embedding(embeddi)

    embeddi = make_laplacian_emb(S,
                                 3,
                                 type_laplacian='unnormalized',
                                 type_normalization='coifman',
                                 scaled=False)
    print(embeddi.shape)
    vizualize_embedding(embeddi)
