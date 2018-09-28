#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some functions to asses the quality of the ordering found
when we have a ground truth.
"""
import numpy as np
from scipy.stats import kendalltau


def inverse_perm(perm):
    ''' reverse a permutation '''
    return(np.argsort(perm))


def kendall_circular(true_perm, order_perm, use_dicho=False):
    '''
    TODO : make it faster for large n with a coarser grained slicing first,
    i.e., taking np.roll with a larger value than 1 and then zooming in.
    '''
    n = true_perm.shape[0]
    if (order_perm.shape[0] != n):
        print("wrong length of permutations in kendall_circular!")
    order_perm = true_perm[order_perm]
    id_perm = np.arange(n)
    scores = np.zeros(n)
    if not use_dicho:
        for i in range(n):
            scores[i] = abs(kendalltau(id_perm, order_perm)[0])
            order_perm = np.roll(order_perm, 1, axis=0)
    else:
        test_dist = 10
        dir = +1
        move_dist = np.floor(n/2)
        score_prec = abs(kendalltau(id_perm, order_perm)[0])
        old_pos = 0
        scores[old_pos] = score_prec
        for i in range(n):
            if move_dist < 2:
                break
            if test_dist < move_dist:  # do fine-grained search now
                test_dist = 1
            new_pos = int((old_pos + dir * move_dist) % n)
            order_perm = np.roll(order_perm, int(dir*move_dist), axis=0)
            # TODO : chose dir locally by checking if score increases by moving
            # forward of backward with test_dist
            new_score = abs(kendalltau(id_perm, order_perm)[0])
            scores[new_pos] = new_score
            old_pos = new_pos
            if new_score < score_prec:
                dir *= -1
                move_dist /= 2
                move_dist = np.floor(move_dist)

    return(np.max(scores), np.argmax(scores))


def evaluate_ordering(perm, true_perm, criterion='kendall',
                      circular=False):
    '''
    evaluate the model.
    INPUT:
        - the ground truth permutation
        - the ordered_chain

        AR to TK : Why did you reverse the permutation in
        evaluate function in algo_read_local.py ?
    '''
    # (s1, s2) = perm.shape
    # (ts1, ts2) = true_perm.shape
    # if not(s1 == ts1 and s2 == ts2):
    #     print("Problem : perm of shape {}x{}, "
    #           "and true_perm of shape {}x{}".format(s1, s2, ts1, ts2))
    l1 = len(perm)
    l2 = len(true_perm)
    if not l1 == l2:
        print("Problem : perm of length {}, "
              "and true_perm of length {}".format(l1, l2))
        print("perm : {}".format(perm))
    if criterion == 'kendall':
        if circular:
            (score, _) = kendall_circular(true_perm, perm)
        else:
            score = abs(kendalltau(true_perm, inverse_perm(perm))[0])
        return(score)


if __name__ == '__main__':
    n = 100
    true_perm = np.arange(n)
    shift = np.random.randint(n)
    perm = np.roll(true_perm, shift, axis=0)
    (score, shift) = kendall_circular(true_perm, perm)
    (score_bis, shift_bis) = kendall_circular(true_perm, perm, use_dicho=True)
