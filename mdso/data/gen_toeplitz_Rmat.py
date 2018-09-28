#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate various Similarity matrix
through the MatrixGenerator methods
gen_matrix for synthetic data, and
gen_E_coli_matrix for DNA data.
"""
import numpy as np
# from scipy import sparse as sp
from scipy.linalg import toeplitz


def gen_lambdas(type_matrix, n):
    '''
    Generates lambdas to define a toeplitz matrix with
    diagonal elements t_k = lambdas[k]
    '''
    array_lambdas = np.zeros(n)
    if type_matrix == 'LinearBanded':
        # Bandwidth = 10% ?
        cov = int(np.floor(n/10))
        array_lambdas[:cov] = cov - abs(np.arange(cov))

    elif type_matrix == 'LinearStrongDecrease':
        alpha = 0.1
        array_lambdas = np.exp(-alpha*np.arange(n))

    elif type_matrix == 'CircularBanded':
        # Bandwidth = 10% ?
        cov = int(np.floor(n/10))
        array_lambdas[:cov] = cov - abs(np.arange(cov))
        array_lambdas[-cov:] = array_lambdas[:cov][::-1]

    elif type_matrix == 'CircularStrongDecrease':
        alpha = 0.1
        array_lambdas = np.exp(-alpha*np.arange(n))
        p = int(np.floor(n/2))
        array_lambdas[-p:] = array_lambdas[:p][::-1]

    else:
        raise ValueError("Unrecognized type_matrix !")

    return(array_lambdas)


def gen_toeplitz_sim(lambdas):
    '''Build Toeplitz strong-R-matrix'''
    return(toeplitz(lambdas))
#
#
# def sym_max(X):
#     """
#     Returns symmetrization of sparse matrix X.
#     X_sym = max(X, X.T) rather than X + X.T to avoid adding up values when
#     there are duplicates in the overlap file.
#     If X is triangular, max(X, X.T) and X + X.T are equal.
#
#     TODO : check how many values are not symmetric
#     and separate cases where Aij = 0 ...
#     """
#
#     dif_mat = X - X.T
#     dif_mat.data = np.where(dif_mat.data < 0, 1, 0)
#     return X - X.multiply(dif_mat) + X.T.multiply(dif_mat)


class MatrixGenerator():

    # Apply permutation
    def apply_perm(self, perm):
        '''
        Apply a permutation to the similarity matrix.
        perm is given as a numpy array
        '''
        n_ = self.n
        # check size is ok
        if np.shape(perm)[0] != n_:
            raise ValueError('the size of the permutation matrix does not match that of the\
                             similarity matrix.')
        # check perm is a permutation
        if not (np.sort(perm) == np.arange(n_)).all():
            raise ValueError('perm is not considered as a'
                             'permutation matrix of [0; \cdots; n-1]')
        self.sim_matrix = self.sim_matrix[perm]
        self.sim_matrix = self.sim_matrix.T[perm]
        self.sim_matrix = self.sim_matrix.T
        return self

    # Add additive noise
    def add_sparse_noise(self, noise_prop, noise_eps,
                         law='uniform'):
        '''
        Create a function that add a symetric sparse noise!
        noiseprop controls the support of the sparse noise
        noiseeps controls the eps amplitude of the noise
        '''
        n_ = self.n
        # first find a random support
        N = np.tril(np.random.rand(n_, n_))
        idx = np.where(N > noise_prop)
        N[idx] = 0
        # allocate value on the support
        [ii, jj] = np.where(N != 0)
        if law == 'gaussian':
            N[np.where(N != 0)] = noise_eps * np.abs(
                np.random.normal(0, 1, len(ii)))
        elif law == 'uniform':
            N[np.where(N != 0)] = noise_eps*np.random.rand(1, len(ii))
        # symetrize the noise
        N = N + N.T
        # Add noise to similarity matrix
        self.sim_matrix += N

        return self

    def gen_matrix(self, n, type_matrix='LinearBanded',
                   apply_perm=True, perm=None,
                   noise_prop=1, noise_ampl=0, law='uniform'):
        self.n = n
        lambdas = gen_lambdas(type_matrix, n)
        self.sim_matrix = gen_toeplitz_sim(lambdas)
        if apply_perm:
            if not perm:  # generate permutation if not provided by user
                perm = np.random.permutation(n)
            self.apply_perm(perm)
            self.true_perm = perm
        else:
            self.true_perm = np.arange(n)
        if noise_ampl > 0:
            normed_fro = np.sqrt(np.mean(self.sim_matrix**2))
            self.add_sparse_noise(noise_prop, noise_ampl*normed_fro, law=law)

        return self
#
    # def gen_E_coli_matrix(self, apply_perm=False):
    #     """
    #     generate similarity matrix from E. coli ONT reads [ref Loman et al.]
    #     TODO :
    #     - change the path to data folder if this is a package ?
    #     - recompute reads_pos with minimap2 instead of BWA.
    #     """
    #     # Read data matrix
    #     data_dir = './data/'
    #     mat_fn = data_dir + 'ecoli_mat.csv'
    #     pos_fn = data_dir + 'ecoli_ref_pos.csv'
    #     mat_idxs = np.genfromtxt(mat_fn, delimiter=',')
    #     reads_pos = np.genfromtxt(pos_fn, delimiter=',')
    #     n_reads = reads_pos.shape[0]
    #     sim_mat = sp.coo_matrix((mat_idxs[:, 2],
    #                              (mat_idxs[:, 0]-1, mat_idxs[:, 1]-1)),
    #                             shape=(n_reads, n_reads),
    #                             dtype='float64').tocsr()
    #     sim_mat = sym_max(sim_mat)
    #     # Remove unaligned reads (unknown ground turh position)
    #     in_idx = np.argwhere(reads_pos < 7e6)[:, 0]
    #     sim_lil = sim_mat.tolil()
    #     self.n = len(in_idx)
    #     if apply_perm:
    #         perm = np.random.permutation(self.n)
    #         self.true_perm = perm
    #         in_idx = in_idx[perm]
    #     else:
    #         self.true_perm = np.arange(self.n)
    #     sim_lil = sim_lil[in_idx, :][:, in_idx]
    #     self.sim_matrix = sim_lil.tocsr()
    #
    #     return self
