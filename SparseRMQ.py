from math import log2


def construct_sparse_table(array, sparse):
    """preprocess RMQ for sub arrays of length 2k using dynamic programming.
    We will keep an array M[0, N-1][0, logN]
    where M[i][j] is the index of the minimum value in the sub array starting at i having length 2^j.
    So, the overall complexity of the algorithm is <O(N logN), O(1)>
    Uses O(N logN) space"""

    n = len(array)
    m = int(log2(n))

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
