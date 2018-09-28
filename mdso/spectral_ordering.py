#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main functions for getting latent ordering through
the spectral embedding of a similarity matrix, as in
arXiv ...
"""
from time import time
import numpy as np
from scipy.sparse.csgraph import connected_components
from .spectral_embedding_ import make_laplacian_emb, check_matrix
from .new_sim_from_embedding_ import get_sim_from_embedding
from .merge_connected_components_ import merge


def get_linear_ordering(new_embedding):
    '''
    Baseline Spectral Linear Ordering Algorithm (Atkins)
    input : 1d spectral embedding (Fiedler vector)
    output : permutation that sorts the entries of the Fiedler vector.
    '''
    shape_ebd = np.shape(new_embedding)
    if len(shape_ebd) > 1:
        # print("Only 1D embedding is needed for Atkins' algo."
        #       "Using first eigenvector only.")
        first_eigen = new_embedding[:, 0]
    else:
        first_eigen = new_embedding

    return(np.argsort(first_eigen))


def get_circular_ordering(new_embedding):
    '''
    Baseline Spectral Circular Ordering Algorithm (Coifman)
    input : 2d spectral embedding
    output : permutation that sorts the angles between the entries of the first
    and second eigenvectors.
    '''
    first_eigen = new_embedding[:, 0]
    second_eigen = new_embedding[:, 1]
    ratio_eigen = np.divide(second_eigen, first_eigen)
    eigen_angles = np.arctan(ratio_eigen)
    eigen_angles[np.where(first_eigen < 0)] += np.pi

    return(np.argsort(eigen_angles))


class SpectralBaseline():
    """
    Basic Spectral Ordering Algorithm.
    For Linear Seriation, uses Atkins' method [ref]
    For Circular Seriation, uses Coifman's method [ref]
    """
    def __init__(self, circular=False, type_laplacian=None,
                 type_normalization=None, scaled=False):
        self.circular = circular
        if circular:
            self.new_dim = 3
            if not type_laplacian:
                type_laplacian = 'random_walk'
        else:
            self.new_dim = 2
            if not type_laplacian:
                type_laplacian = 'unnormalized'
        self.type_lap = type_laplacian
        self.type_norm = type_normalization
        self.scaled = scaled

    def fit(self, X):
        """
        /!\ X must be connected.

        """
        # Get 1d or 2d Spectral embedding to retrieve the latent ordering
        self.new_embedding_ = make_laplacian_emb(
            X, self.new_dim,
            type_laplacian=self.type_lap,
            type_normalization=self.type_norm,
            scaled=self.scaled)

        if self.circular:
            self.ordering_ = get_circular_ordering(self.new_embedding_)
        else:
            self.ordering_ = get_linear_ordering(self.new_embedding_)

        return self


class SpectralOrdering():
    """
    Main functions for getting latent ordering through
    the spectral embedding of a similarity matrix, as in
    arXiv ...

    Parameters
    ----------
    dim : int, default 10
        The number of dimensions of the spectral embedding.

    k_nbrs : int, default 10
        The number of nearest neighbors in the local alignment algorithm.

    type_laplacian : string, default "random_walk"
        type of normalization of the Laplacianm Can be "unnormalized",
        "random_walk", or "symmetric".

    scaled : string or boolean, default True
        if scaled is False, the embedding is just the concatenation of the
        eigenvectors of the Laplacian, i.e., all dimensions have the same
        weight.
        if scaled is "CTD", the k-th dimension of the spectral embedding
        (k-th eigenvector) is re-scaled by 1/sqrt(lambda_k), in relation
        with the commute-time-distance.
        If scaled is True or set to another string than "CTD", then the
        heuristic scaling 1/sqrt(k) is used instead.

    min_cc_len : int, default 10
        if the new similarity matrix is disconnected, keep only connected
        components of size larger than min_cc_len

    merge_if_ccs : bool, default False
        if the new similarity matrix is disconnected

    Attributes
        ----------
        embedding : array-like, (n_pts, dim)
            spectral embedding of the input matrix.

        new_sim : array-like, (n_pts, n_pts)
            new similarity matrix

        dense : boolean
            whether the input matrix is dense or not.
            If it is, then new_sim is also returned dense (otherwise sparse).
    """
    def __init__(self, dim=10, k_nbrs=10, norm_sim=False,
                 type_new_sim=None,
                 norm_local_diss=True, circular=False,
                 type_laplacian='random_walk', type_norm=None,
                 scaled=True, preprocess_only=False, min_cc_len=10,
                 merge_if_ccs=False, verb=0, do_eps_graph=False,
                 eps_val=None):
        self.dim = dim
        self.k_nbrs = k_nbrs
        self.norm_sim = norm_sim
        self.type_new_sim = type_new_sim
        self.norm_local_diss = norm_local_diss
        self.circular = circular
        self.type_laplacian = type_laplacian
        self.type_norm = type_norm
        self.scaled = scaled
        self.preprocess_only = preprocess_only
        self.min_cc_len = min_cc_len
        self.merge_if_ccs = merge_if_ccs
        self.verb = verb
        self.do_eps_graph = do_eps_graph
        self.eps_val = eps_val

    def merge_connected_components(self, X, mode='similarity'):
        if not type(self.partial_orderings) == list:
            raise TypeError("self.ordering should be a list (of lists)")
        if not type(self.partial_orderings[0]) == list:
            return self
        else:
            self.ordering = merge(self.partial_orderings, X,
                                  self.embedding, h=self.k_nbrs, mode=mode)
            self.ordering = np.array(self.ordering)
        return self

    def fit(self, X):
        """
        Creates a Laplacian embedding and a new similarity matrix
        """

        # If dim == 0, just run the baseline spectral method
        if self.dim == 1:
            # self.new_sim = X
            # Create a baseline spectral seriation solver
            ordering_algo = SpectralBaseline(circular=self.circular)
            ordering_algo.fit(X)
            self.ordering = ordering_algo.ordering_
            return(self)
        else:

            # Compute the Laplacian embedding
            self.embedding = make_laplacian_emb(
                X, self.dim, type_laplacian=self.type_laplacian,
                type_normalization=self.type_norm, scaled=self.scaled,
                verb=self.verb)

            self.dense = True if check_matrix(X) == 'dense' else False

            # Get the cleaned similarity matrix from the embedding
            if self.verb > 0:
                print("Compute new similarity from embedding")
            t0 = time()
            self.new_sim = get_sim_from_embedding(
                self.embedding,
                X,
                k_nbrs=self.k_nbrs,
                type_simil=self.type_new_sim,
                norm_local_diss=self.norm_local_diss,
                norm_sim=self.norm_sim,
                return_dense=self.dense,
                eps_graph=self.do_eps_graph,
                eps_val=self.eps_val)
            if self.verb > 0:
                print("Done in {}".format(time()-t0))

        # Get the latent ordering from the cleaned similarity matrix
        if not self.preprocess_only:
            # Make sure we have only one connected componen
            # in the new similarity.
            (n_c, lbls) = connected_components(self.new_sim, directed=False,
                                               return_labels=True)
            if n_c == 1:
                # Create a baseline spectral seriation solver
                ordering_algo = SpectralBaseline(circular=self.circular)
                ordering_algo.fit(self.new_sim)
                self.ordering = ordering_algo.ordering_
            else:
                if self.verb > 0:
                    print("new similarity disconnected. "
                          "Reordering connected components.")
                # Create a baseline spectral seriation solver
                # Set circular=False because if we have broken the circle
                # in several pieces, we only have local linear orderings.
                ordering_algo = SpectralBaseline(circular=False)

                # Get one ordering per connected component
                size_ccs = [sum(lbls == i_cc) for i_cc in range(n_c)]
                size_ccs = np.array(size_ccs)
                ord_ccs = np.argsort(-size_ccs)
                n_large_cc = sum(size_ccs > self.min_cc_len)
                self.partial_orderings = []
                # Convert sparse matrix to lil format for slicing
                if not self.dense:
                    self.new_sim = self.new_sim.tolil()
                for cc_idx in ord_ccs[:n_large_cc]:
                    in_cc = np.argwhere(lbls == cc_idx)[:, 0]
                    ordering_algo.fit(self.new_sim[in_cc, :][:, in_cc])
                    self.partial_orderings.append(
                        in_cc[ordering_algo.ordering_])

                self.partial_orderings = [list(
                    partial_order) for partial_order in self.partial_orderings]

                if self.merge_if_ccs:
                    if self.verb > 0:
                        print("Merging connected components.")
                    self.merge_connected_components(X)
                else:
                    self.ordering = self.partial_orderings

        return self

    def fit_transform(self, X):
        """

        """
        self.fit(X)
        return self.ordering
