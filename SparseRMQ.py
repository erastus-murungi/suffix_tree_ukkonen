from math import log2
import numpy as np
from sys import maxsize


def construct_sparse_table(array):
    """preprocess RMQ for sub arrays of length 2k using dynamic programming.
    We will keep an array M[0, N-1][0, logN]
    where M[i][j] is the index of the minimum value in the sub array starting at i having length 2^j.
    So, the overall complexity of the algorithm is <O(N logN), O(1)>
    Uses O(N logN) space"""

    n = len(array)
    m = int(log2(n))
    sparse = np.full((n, m + 1), maxsize, dtype='int64')

    for i in range(n):  # intervals of length 1
        sparse[i][0] = i
    for j in range(1, m + 1):  # log(n)
        if (1 << j) > n:  # 1 << j == 2^j
            break
        for i in range(n):
            if (i + (1 << j) - 1) >= n:
                break  # i + 2^j - 1
            else:
                if array[sparse[i][j - 1]] < array[sparse[i + (1 << (j - 1))][j - 1]]:
                    sparse[i][j] = sparse[i][j - 1]
                else:
                    sparse[i][j] = sparse[i + (1 << (j - 1))][j - 1]
    return sparse


def query_sparse_table(low, high, array=None, sparse=None):
    """In this operation we can query on an interval or segment and
     return the answer to the problem on that particular interval."""

    length = (high - low) + 1
    k = int(log2(length))
    if array[sparse[low][k]] <= array[sparse[low + length - (1 << k)][k]]:
        return sparse[low][k]
    else:
        return sparse[high - (1 << k) + 1][k]


def euler_tour(source):
    E = []
    D = []
    R = {}
    if source is None:
        raise ValueError("Build cartesian tree first")

    def euler_visit(curr_node, nodes, depths, rep, curr_depth, index):
        depths.append(curr_depth)
        nodes.append(curr_node)
        if curr_node not in rep:
            rep[curr_node] = index

        if len(curr_node.out) > 0:
            for k in curr_node.out:
                index = euler_visit(curr_node.out[k], nodes, depths, rep, curr_depth + 1, index + 1)
                nodes.append(curr_node)
                depths.append(curr_depth)
                index += 1
        return index
    euler_visit(source, E, D, R, 0, 0)
    # print('->'.join(map(str, E)))
    return E, D, R
