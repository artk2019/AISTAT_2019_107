#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use the multidimensional spectral ordering method to compute the layout
of a bacterial (E. coli) genome with Oxford Nanopore Technology long reads.
"""
from time import time
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import coo_matrix, find
from scipy.sparse.csgraph import connected_components
from sklearn.manifold import SpectralEmbedding
from Bio import SeqIO
from mdso import SpectralOrdering, evaluate_ordering


# ##############################################################################
# Tools to handle the overlaps between reads to get the similarity matrix
# ##############################################################################
class MiniOvl:
    """ Overlap between two reads, named 1 and 2, from line of minimap file.
    Such a line contains :
    query name, length, 0-based start, end, strand,
    target name, length, start, end, the number of matching bases.
    Parameters
    ----------
    mini_line : str (line from minimap file)
    Attributes
    ----------
    id1 : str (read id of read 1)
    id2 : str (read id of read 2)
    len1 : int (length of read 1)
    len2 : int (length of read 2)
    b1 : int (basepair number of the beginning of the overlap on read 1)
    e1 : int (basepair number of the end of the overlap on read 1)
    b2 : int (basepair number of the beginning of the overlap on read 2)
    e2 : int (basepair number of the end of the overlap on read 2)
    strand : char ('+' if the two reads are on same strand and '-' otherwise)
    n_match : int (number of matching bases (see minimap
    [https://github.com/lh3/minimap] documentation))
    """
    def __init__(self, mini_line):
        fields = mini_line.split()
        self.id1 = fields[0]
        self.len1 = int(fields[1])
        self.b1 = int(fields[2])
        self.e1 = int(fields[3])
        self.strand = fields[4]
        self.id2 = fields[5]
        self.len2 = int(fields[6])
        self.b2 = int(fields[7])
        self.e2 = int(fields[8])
        self.n_match = int(fields[9])
        # self.n_coll = int(fields[10])
        # self.n_frac_match = int(fields[11])

    def switch_ids(self, id1, id2):
        """ Switch reads in the overlap object (read 1 becomes 2 and 2 becomes
        1).
        """
        if (self.id1 == id2) and (self.id2 == id1):
            self.id1, self.id2 = self.id2, self.id1
            self.len1, self.len2 = self.len2, self.len1
            self.b1, self.b2 = self.b2, self.b1
            self.e1, self.e2 = self.e2, self.e1
        else:
            assert self.id1 == id1 and self.id2 == id2, \
                "id1 : {}, id2 : {} \n self.id1 : {}, self.id2 : {}".format(
                    id1, id2, self.id1, self.id2)


def compute_overlaps(mini_fn, record_list):
    """ Compute list of overlaps from minimap output file and list of reads.
    Parameters
    ----------
    mini_fn : str (path to minimap file)
    record_list : list (list of reads in Bio.SeqIO.records format)
    Returns
    ----------
    read_nb2id : dict (keys : read number, values : read id)
    ovl_list : list (of overlaps as MiniOvl objects)
    i_list : list (of read indices (int) i to build sparse coo_matrix such that
    A[i,j] ~ overlap between reads i and j)
    j_list : list (of read indices (int) j to build sparse coo_matrix such that
    A[i,j] ~ overlap between reads i and j)
    k_list : list (of indices (int) k such that ovl_list[k] is the overlap
    between i_list[k] and j_list[k])
    n_match_list : list (of number of matches (int) such that A[i,j] = number
    of matches between i and j)
    ovl_len_list : list (of length of overlap between i and j)
    n_reads : int (number of reads)
    """
    # Construct {read name : read number} dictionary
    read_nb_dic = {}
    cpt = 0
    for record in record_list:
        if record.id in read_nb_dic:
            msg = "Same id {} for reads {} and {} ! "
            "Run [https://github.com/antrec/spectrassembler/]"
            "check_reads.py "
            "on your data first.".format(record.id, read_nb_dic[record.id],
                                         cpt)
            raise ValueError(msg)
        read_nb_dic[record.id] = cpt
        cpt += 1
    n_reads = cpt

    idx = 0
    h_list = []
    k_list = []
    ovl_list = []
    n_match_list = []
    ovl_len_list = []
    fh = open(mini_fn, 'r')

    for line in fh:

        ovl = MiniOvl(line)
        i_idx = read_nb_dic[ovl.id1]
        j_idx = read_nb_dic[ovl.id2]

        # Discard self matches
        if i_idx == j_idx:
            continue

        # Keep 1D indexing : h = n*i + j
        h_idx = n_reads*i_idx + j_idx

        # Check if another overlap between i and j already exists
        duplicate_cond = (h_idx in h_list[-300:])
        if duplicate_cond:
            dupl_idx = h_list[-300:].index(h_idx) + \
                len(h_list) - min(300, len(h_list))
            dupl_ovl = ovl_list[dupl_idx]

            # Drop the overlap if the preexisting one is more significant
            if dupl_ovl.n_match > ovl.n_match:
                continue

            # Replace the preexisting overlap by the new one otherwise
            else:
                n_match_list[dupl_idx] = dupl_ovl.n_match
                ovl_len = (abs(dupl_ovl.e1 - dupl_ovl.b1) +
                           abs(dupl_ovl.e2 - dupl_ovl.b2))/2
                ovl_len_list[dupl_idx] = ovl_len
                continue

        # Add the overlap if there was no other overlap between i and j
        ovl_list.append(ovl)
        h_list.append(h_idx)
        k_list.append(idx)
        idx += 1
        n_match_list.append(ovl.n_match)
        ovl_len = (abs(ovl.e1 - ovl.b1) + abs(ovl.e2 - ovl.b2))/2
        ovl_len_list.append(ovl_len)

    fh.close()
    # Convert to numpy arrays
    h_list = np.array(h_list)
    n_match_list = np.array(n_match_list)
    ovl_len_list = np.array(ovl_len_list)
    k_list = np.array(k_list)

    # Recover i_list and j_list from h_list indexing (h = n_reads*i + j)
    i_list = h_list//n_reads
    j_list = h_list - n_reads*i_list

    # fh.close()
    read_nb2id = {v: k for (k, v) in read_nb_dic.items()}

    return (i_list, j_list, k_list, read_nb2id, ovl_list, n_match_list,
            ovl_len_list, n_reads)


def sym_max(X):
    """
    Returns symmetrization of sparse matrix X.
    X_sym = max(X, X.T) rather than X + X.T to avoid adding up values when
    there are duplicates in the overlap file.
    If X is triangular, max(X, X.T) and X + X.T are equal.

    TODO : check how many values are not symmetric
    and separate cases where Aij = 0 ...
    """

    dif_mat = X - X.T
    dif_mat.data = np.where(dif_mat.data < 0, 1, 0)
    return X - X.multiply(dif_mat) + X.T.multiply(dif_mat)


def build_sim_mat(ovlp_fn, reads_fn, percentile_thr=5):
    """
    Build similarity matrix between DNA reads from minimap2 overlap file
    (ovlp_fn), using the raw reads file (reads_fn) to get the reads id
    corresponding to a given position, keeping only the overlaps with value
    over the threshold ovlp_thr.
    """
    if reads_fn[-1].lower() == 'a':
        reads_fmt = 'fasta'
    elif reads_fn[-1].lower() == 'q':
        reads_fmt = 'fastq'
    else:
        raise NameError(" reads function must be in .fasta or .fastq format")
    reads_fh = open(reads_fn, "r")
    record_list = list(SeqIO.parse(reads_fh, reads_fmt))
    (i_list, j_list, k_list, read_nb2id, ovl_list, n_match_list,
     ovl_len_list, n_reads) = compute_overlaps(ovlp_fn, record_list)
    iis = np.array(i_list)
    jjs = np.array(j_list)
    vvs = np.array(n_match_list)
    ovlp_thr = np.percentile(vvs, percentile_thr)
    over_thr = np.where(vvs > ovlp_thr)[0]

    sim_mat = coo_matrix((vvs[over_thr],
                          (iis[over_thr], jjs[over_thr])),
                         shape=(n_reads, n_reads),
                         dtype='float64').tocsr()
    sim_mat = sym_max(sim_mat)

    return sim_mat


# ##############################################################################
# Handle the mapping of the reads against reference genome to get
# ground truth positions
# ##############################################################################
def get_aln(header):

    (name, _, chrom, pos, qual) = header

    return {'name': name,
            'chrm': chrom,
            'pos': int(pos),
            'qual': int(qual)}


def read_id2idx(reads_fn):

    if reads_fn[-1] == 'q':
        fmt = 'fastq'
    else:
        fmt = 'fasta'

    reads_idx = {}
    reads_fh = open(reads_fn, 'r')
    num_read = 0
    for record in SeqIO.parse(reads_fh, fmt):
        reads_idx[record.id] = num_read
        num_read += 1
    reads_fh.close()

    return reads_idx


def get_headers(sam_fn):

    # Load headers from sam file
    headers = []
    fh = open(sam_fn, 'r')
    for line in fh:
        header = line.split('\t')[:5]
        headers.append(header)
    fh.close()

    # Check if eukaryotic or prokaryotic and strip headers
    chr_names = []
    skipline = 0
    for header in headers:
        if header[0] == '@SQ':
            chr_names.append(header[1][3:])
            skipline += 1
        elif header[0][0] == '@':
            skipline += 1
            continue
        else:
            break

    headers = headers[skipline:]

    return (headers, chr_names)


def algn_dic(headers, reads_id_dic):

    algns = {}

    for header in headers:
        aln = get_aln(header)
        read_idx = reads_id_dic[aln['name']]

        if read_idx in algns:
            if algns[read_idx]['qual'] > aln['qual']:
                continue

        algns[read_idx] = aln

    return algns


def int_to_roman(input):
    """
    Convert an integer to Roman numerals.
    """

    if not isinstance(input, int):
        raise TypeError("expected integer, got {}".format(type(input)))
    if not 0 < input < 4000:
        raise ValueError("Argument must be between 1 and 3999")
    ints = (1000, 900,  500, 400, 100,  90, 50,  40, 10,  9,   5,  4,   1)
    nums = ('M',  'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV',
            'I')
    result = ""
    for i in range(len(ints)):
        count = int(input // ints[i])
        result += nums[i] * count
        input -= ints[i] * count

    return result


def get_pos_from_aln_file(aln_fn, reads_fn):
    (headers, chr_names) = get_headers(aln_fn)

    names_chr = {name: int_to_roman(idx+1) for (idx,
                                                name) in enumerate(chr_names)}

    reads_dic = read_id2idx(reads_fn)
    algnmts = algn_dic(headers, reads_dic)

    for (idx, read_aln) in algnmts.items():
        read_aln['number'] = idx
        if read_aln['chrm'] in names_chr:
            read_aln['chr_nb'] = names_chr[read_aln['chrm']]
        else:
            read_aln['chr_nb'] = '*'

    alns_df = pd.DataFrame(list(algnmts.values())).sort_values('number')

    return alns_df


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


ovlp_fn = '/home/arecanat/Desktop/test_genome_exp/results/ovlp.paf'
reads_fn = '/home/arecanat/Desktop/test_genome_exp/ecoli_data/reads.fasta'
aln_fn = '/home/arecanat/Desktop/test_genome_exp/results/aln.sam'
# Build the similarity matrix
sim_mat = build_sim_mat(ovlp_fn, reads_fn, percentile_thr=0)
(iis, jjs, vvs) = find(sim_mat)
ovlp_thr = np.percentile(vvs, 40)
over_thr = np.where(vvs > ovlp_thr)[0]
sim_mat = coo_matrix((vvs[over_thr],
                     (iis[over_thr], jjs[over_thr])),
                     shape=sim_mat.shape,
                     dtype='float64').tocsr()


# Get the true position of the reads
alns_df = get_pos_from_aln_file(aln_fn, reads_fn)
positions = alns_df['pos'].values

# Restrict similarity matrix to largest connected component if disconnected
ccs = get_conn_comps(sim_mat, min_cc_len=10)
sub_idxs = ccs[0]
new_mat = sim_mat.tolil()[sub_idxs, :]
new_mat = new_mat.T[sub_idxs, :].T
sub_pos = positions[sub_idxs]
true_perm = np.argsort(sub_pos)
true_inv_perm = np.argsort(true_perm)
new_mat.shape

# Parameters for Spectral Ordering
apply_perm = False  # whether to randomly permute the matrix, so that the
# ground truth is not the trivial permutation (1, ..., n).

# Set parameters for the ordering algorithm
k_nbrs = 50  # number of neighbors in the local linear fit in the embedding
dim = 5  # number of dimensions of the embedding
circular = True  # whether we are running Circular or Linear Seriation
scaled = 'CTD'  # whether or not to scale the coordinates of the embedding so
# that the larger dimensions have fewer importance
type_lap = 'unnormalized'  # Remark : we have observed stranged (and poor)
# results with the normalized Laplacians
min_cc_len = 10  # Drop the tiny connected components

# Call Spectral Ordering method
reord_method = SpectralOrdering(dim=dim, k_nbrs=k_nbrs, circular=circular,
                                scaled=scaled, type_laplacian=type_lap,
                                verb=1,
                                type_new_sim='exp',
                                norm_local_diss=False,
                                norm_sim=False,
                                merge_if_ccs=True,
                                min_cc_len=min_cc_len,
                                do_eps_graph=True,
                                preprocess_only=True)
# Run the spectral ordering method on the DNA reads similarity matrix
t0 = time()
reord_method.fit(new_mat)
my_ebd = reord_method.embedding
tme = time() - t0

print("my embedding in {}s".format(tme))

skl_method = SpectralEmbedding(n_components=dim, affinity='precomputed')
skl_method.fit(new_mat.toarray())
skl_ebd = skl_method.embedding_
tskl = time() - tme
print("sklearn embedding in {}s".format(tskl))

skl_amg = SpectralEmbedding(n_components=dim, affinity='precomputed',
                            eigen_solver='amg')
skl_amg.fit(new_mat)
amg_ebd = skl_amg.embedding_
tamg = time() - tskl
print("sklearn amg embedding in {}s".format(tamg))

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(my_ebd[:, 0], my_ebd[:, 1], my_ebd[:, 2])

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(skl_ebd[:, 0], skl_ebd[:, 1], skl_ebd[:, 2])

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(amg_ebd[:, 0], amg_ebd[:, 1], amg_ebd[:, 2])
