#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to run a batch of n_avrg experiments with a fixed combination of
parameters (n, k, dim, ampl_noise, type_mat, scaled, type_lap), useful if you
wish to run experiments on a cluster, with a loop on the values of (n, k, dim,
ampl_noise, type_mat, scaled, type_lap) in a shell script (e.g.,
run_exps_cluster.sh) with my SGE cluster.
"""
import os
import argparse
import pathlib
import numpy as np
from mdso import MatrixGenerator, SpectralOrdering, evaluate_ordering


def fetch_res(n, k, dim, ampl_noise, type_mat, scaled, type_lap, n_avrg,
              save_res_dir):
    """
    Get the results from a given experiment when they were saved to a file,
    in order to use them for plotting in plot_from_res, or not to redo the same
    computation twice in run_synthetic_exps.
    """
    fn = "n_{}-k_{}-dim_{}-ampl_{}-type_mat_{}" \
         "-scaled_{}-type_lap_{}-n_avrg_{}.res".format(n, k, dim, ampl_noise,
                                                       type_mat, scaled,
                                                       type_lap, n_avrg)
    fn = save_res_dir + "/" + fn
    with open(fn, 'r') as f:
        first_line = f.readlines()[0]
    res_mean, res_var = [float(el) for el in first_line.split()]

    return(res_mean, res_var)


def run_one_exp(n, k, dim, ampl, type_matrix, scaled, n_avrg,
                type_lap=None, type_noise='gaussian'):
    """
    Run n_avrg experiments for a given set of parameters, and return the mean
    kendall-tau score and the associated standard deviation (among the
    n_avrg instances).
    """

    # Pre-defined settings for Laplacian_Fiedler : if circular, random_walk,
    # if linear, unnormalized; Laplacian_init : random_walk
    if not type_lap:
        type_lap = 'random_walk'

    if type_matrix[0] == 'L':
        circular = False
    elif type_matrix[0] == 'C':
        circular = True
    else:
        raise ValueError("type matrix must be in ['LinearBanded',"
                         "'CircularBanded', 'LinearStrongDecrease',"
                         "'CircularStrongDecrease']")
    # Create matrix generator
    data_gen = MatrixGenerator()
    # Create spectral solver
    reord_method = SpectralOrdering(dim=dim, k_nbrs=k, circular=circular,
                                    scaled=scaled,
                                    type_laplacian=type_lap,
                                    verb=1)
    # Initialize array of results
    scores = np.zeros(n_avrg)
    for i_exp in range(n_avrg):
        np.random.seed(i_exp)
        data_gen.gen_matrix(n, type_matrix=type_matrix, apply_perm=True,
                            noise_ampl=ampl, law=type_noise)
        this_perm = reord_method.fit_transform(data_gen.sim_matrix)
        scores[i_exp] = evaluate_ordering(this_perm, data_gen.true_perm,
                                          circular=circular)
        print('.', end='')
    print('')

    return(scores.mean(), scores.std(), scores)


# Define argument parser
parser = argparse.ArgumentParser(description="run some experiments"
                                 "with combination of parameters dim, "
                                 "amplitude, k, type_laplacian_initial, "
                                 "type_laplacian_fiedler, type_matrix")

parser.add_argument("-r", "--root_dir",
                    help="directory where to store result files.",
                    type=str,
                    default="./")
parser.add_argument("-i", "--type_laplacian_initial",
                    help="Laplacian for embedding. 'random_walk' or "
                    "'unnormalized'.",
                    type=str,
                    default='random_walk')
parser.add_argument("-m", "--type_matrix",
                    help="Type of similarity matrix. 'LinearBanded', "
                    "'LinearStrongDecrease', 'CircularBanded' or "
                    "'CircularStrongDecrease'.",
                    type=str,
                    default='LinearBanded')
parser.add_argument("-k", "--k_nbrs",
                    help="number of nearest-neighbors used in to approximate "
                    "a local line",
                    type=int,
                    default=15)
parser.add_argument("-d", "--dim", help="dimension of the Laplacian embedding",
                    type=int,
                    default=3)
parser.add_argument("-a", "--amplitude_noise",
                    help="amplitude of the noise on the matrix.",
                    type=float,
                    default=0.5)
parser.add_argument("-n", "--n",
                    help="number of elements (size of similarity matrix)",
                    type=int,
                    default=500)
parser.add_argument("-s", "--scale_embedding",
                    help="If scaled == 0, do not apply any scaling."
                    "If scaled == 1, apply CTD scaling to embedding, "
                    "(y_k /= sqrt(lambda_k))."
                    "If scaled == 2, apply default scaling (y_k /= sqrt(k)).",
                    type=int,
                    default=2)
parser.add_argument("-e", "--n_exps",
                    help="number of experiments performed to average results",
                    type=int,
                    default=100)
parser.add_argument("--type_noise",
                    help="type of noise ('uniform' or 'gaussian')",
                    type=str,
                    default='gaussian')

# Get arguments
args = parser.parse_args()

n = args.n
k = args.k_nbrs
dim = args.dim
type_matrix = args.type_matrix
type_lap = args.type_laplacian_initial
scale_code = args.scale_embedding
if scale_code == 0:
    scaled = False
elif scale_code == 1:
    scaled = 'CTD'
else:
    scaled = True
type_noise = args.type_noise
ampl = args.amplitude_noise
n_avrg = args.n_exps
save_res_dir = args.root_dir

# Create directory for results if it does not already exist
if save_res_dir:
    pathlib.Path(save_res_dir).mkdir(parents=True, exist_ok=True)

print("n:{}, k:{}, dim:{}, ampl:{}, "
      "type_matrix:{}, scaled:{}, "
      "type_lap:{}, "
      "n_avrg:{}".format(n, k, dim, ampl,
                         type_matrix,
                         scaled,
                         type_lap, n_avrg))

if save_res_dir:
    # Check if the file already exists and read results if so
    fn = "n_{}-k_{}-dim_{}-ampl_{}" \
         "-type_mat_{}" \
         "-scaled_{}-type_lap_{}-n_avrg_{}." \
         "res".format(n, k, dim, ampl,
                      type_matrix, scaled,
                      type_lap, n_avrg)
    fn = save_res_dir + "/" + fn
    if os.path.isfile(fn):
        (mn, stdv) = fetch_res(n, k, dim, ampl,
                               type_matrix,
                               scaled,
                               type_lap,
                               n_avrg,
                               save_res_dir)
    else:
        # Run the experiments if the result file does not already exist
        (mn, stdv, scores) = run_one_exp(n, k, dim, ampl, type_matrix, scaled,
                                         n_avrg, type_lap=type_lap)
    # Print results
    print("MEAN_SCORE:{}, STD_SCORE:{}"
          "".format(mn, stdv))
    fh = open(fn, 'a')
    print(mn, stdv, file=fh)
    print(scores, file=fh)
    fh.close()
