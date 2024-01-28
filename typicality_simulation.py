import numpy as np
from numpy import kron
from Polar_Code_Designer import designer, awgn_p1
from math import log2
from itertools import product
import sys
# import matplotlib as mpl
# mpl.use('pgf')
import matplotlib.pyplot as plt
import copy
import numba
from numba import int64, float64, jit, njit, vectorize
import typing
# import tikzplotlib



# define a litany of helper functions
G2 = np.array([
    [1, 0],
    [1, 1]
])

def pattern2mat(pattern):
    n = len(pattern)
    mat = np.zeros((n,n))
    for i in range(n):
        mat[pattern[i], i] = 1
    return mat

def R(N):
    evens = list(range(0,N,2))
    odds = list(range(1,N,2))
    return pattern2mat(evens + odds)

def G(N):
    if N > 2:
        return kron(np.identity(N // 2), G2) @ R(N) @ kron(np.identity(N // 2), G2)
    else:
        return G2

def bit2prob(bit, prob0, prob1):
    if bit == '0':
        return prob0
    else:
        return prob1
    
def binary_entropy(p):
    return -p * log2(p) - (1-p) * log2(1-p)

num_polar_trials = 10000 # number of trials in Monte Carlo method to design polar code
num_flip_probs = 100 # number of intrinsic bit-flip values to test
blocklength_power = 3 # power of blocklength of code
n = 2**blocklength_power
fig, ax = plt.subplots()
num_errs=4 # number of design error probabilities
design_err_prob_arr = np.linspace(0.1, 0.4, num=num_errs, endpoint=True)
for j in range(num_errs):
    design_err_prob = design_err_prob_arr[j]
    success_arr = []
    entropy_arr = []
    # design classical polar code of length n: choose information and frozen bits
    bit_assignments = designer(int(log2(n)), awgn_p1, design_err_prob, num_polar_trials, 0.1)
    info_bits = [k for k, val in enumerate(bit_assignments) if val == 0.5]
    frozen_bits = [k for k, val in enumerate(bit_assignments) if val == 0]
    # generate codewords
    num_info_bits = len(info_bits)
    num_frozen_bits = n - num_info_bits
    rate = num_info_bits / n
    # create table
    vecs = product(range(2), repeat=n)
    col0 = np.array([np.zeros(num_frozen_bits, dtype=int)], dtype=int)
    col1 = np.array([np.zeros(n, dtype=int)], dtype=int)
    # build syndrome table
    for error in vecs:
        syndrome = np.take((np.asarray(error) @ G(n)) % 2, frozen_bits) # type: ignore
        if not syndrome.tolist() in col0.tolist():
            col0 = np.append(col0, np.array([syndrome]), axis=0)
            col1 = np.append(col1, np.array([error]), axis=0)
        if col0.shape[0] >= 2**num_frozen_bits:
            break
    col0 = col0.astype('int')        
    col0 = col0.astype('str')
    col1 = col1.astype('int')
    col1 = col1.astype('str')
    # print(col0[0])
    if (col0[0].shape)[0] == 0:
        sys.exit("Typical subspace empty")
    
    for p0 in np.linspace(1 - design_err_prob, 1, num = num_flip_probs * (1 - j // 4), endpoint=False):
        p1 = 1 - p0
        # find eigenvalues of typical words
        typ_probs = []
        for k in col1:
            typ_probs.append([bit2prob(l, p0, p1) for l in k])
        typ_probs = [np.prod(k) for k in typ_probs]
        entropy_arr.append(binary_entropy(p0))
        success_arr.append(np.sum(typ_probs))
    fig = plt.figure()
    ax.scatter(entropy_arr, success_arr, label = 'Design Error Probability: {}'.format(round(design_err_prob, 1)), s = 2)
    ax.plot([1 - design_err_prob, 1 - design_err_prob + 0.1], [rate, rate], ls='-')
ax.set_xlabel("Binary Entropy of Bit-Flip Probability")
ax.set_ylabel("Probability of Success")
ax.set_title("Binary Entropy of Bit Source vs. Probability of Success for Blocklength {}".format(n))
ax.legend()
plt.rcParams['figure.dpi'] = 1200
plt.rcParams['savefig.dpi'] = 1200
