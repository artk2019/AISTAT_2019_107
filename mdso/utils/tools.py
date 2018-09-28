#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Various useful functions
"""
import numpy as np
from scipy.sparse.csgraph import connected_components


def get_conn_comps(mat, min_cc_len=1):
    """
    Returns a list of connected components of the matrix mat by decreasing size
    of the connected components, for all cc of size larger or equal than
    min_cc_len
    """
    n_c, lbls = connected_components(mat)
    srt_lbls = np.sort(lbls)
    dif_lbls = np.append(np.array([1]), srt_lbls[1:] - srt_lbls[:-1])
    dif_lbls = np.append(dif_lbls, np.array([1]))
    switch_lbls = np.where(dif_lbls)[0]
    diff_switch = switch_lbls[1:] - switch_lbls[:-1]
    ord_ccs = np.argsort(-diff_switch)
    len_ccs = diff_switch[ord_ccs]
    ccs_l = []
    for (i, cc_idx) in enumerate(ord_ccs):
        if len_ccs[i] < min_cc_len:
            break
        ccs_l.append(np.where(lbls == cc_idx)[0])
    return ccs_l
