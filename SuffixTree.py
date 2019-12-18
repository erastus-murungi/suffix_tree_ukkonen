"""An implementation of Ukkonen's suffix tree algorithm
Uses O(1) space per edge label. the character '#' indicates the last character of the string
"""
from dataclasses import dataclass
from random import choice
from string import ascii_lowercase
from datetime import datetime
from typing import List, Iterable, Sized
from array import array
from SparseRMQ import construct_sparse_table
from sys import maxsize


@dataclass
class End:
    end: float


class SuffixNode:
    """A class representing an edge in a suffix tree
    If self.out is empty, then the edge is just a leaf node
    """

    def __init__(self, start, end):
        self.out = {}  # a dictionary of the children of a node
        self.suffix_link = None
        self.start = start
        self.end = end

    def __repr__(self):
        """Simple repr method showing S[i: j]"""
        return '{}{}'.format(self.__class__.__qualname__, "[" + str(self.start) + ", " + str(self.end.end) + "]")

    def __len__(self):
        """Returns number of characters - 1"""
        return self.end.end - self.start


@dataclass
class ActivePoint:
    active_node: SuffixNode
    active_length: int
    active_edge: int


class SuffixTree:
    __sentinels__ = ('$', '@', '*', '&' '%', '#')  # lexicographically lower than chars of input string

    def __init__(self, s: [str, Iterable, Sized]):
        if type(s) == str:
            self.s = s + SuffixTree.__sentinels__[0]
        else:
            #  Generalized suffix tree
            assert (len(s) <= len(SuffixTree.__sentinels__)), "not enough sentinels for the strings"
            self.s = ''.join([p[0] + p[1] for p in zip(s, SuffixTree.__sentinels__)])  # append sentinels to strings

        self.root = SuffixNode(0, End(-1))
        self.a = ActivePoint(self.root, 0, 0)
        self.end = End(-1)
        self.rem = 0
        self.SA = None
        self.subtree_sizes = None

        #  incase LCA queries are made
        self.E = None
        self.D = None
        self.R = None
        self.sparse = None

    def build_suffix_tree(self):

        for i, c in enumerate(self.s):
            self._start_phase(i)

        self.create_labels()

    def _start_phase(self, i):
        last_node = None
        self.end.end += 1
        self.rem += 1

        while self.rem > 0:

            if self.a.active_length == 0:

                if self.s[i] in self.a.active_node.out:
                    self.a.active_length += 1
                    self.a.active_edge = self.a.active_node.out[self.s[i]].start
                    break
                else:
                    self.root.out[self.s[i]] = SuffixNode(i, self.end)
                    self.rem -= 1
            else:
                ch = self.get_next_character(i)
                if ch is not None:
                    if ch == self.s[i]:
                        edge = self.select_edge()
                        if last_node is not None:
                            last_node.suffix_link = edge
                            # print("Suffix link added from " + repr(last_node) + "  to  " + repr(edge))
                        self.walk(i)
                        break
                    else:
                        internal_node = self.split(i)
                        if last_node is not None:
                            last_node.suffix_link = internal_node
                            # print(i, "Suffix link added from " + repr(last_node) + "  to  " + repr(internal_node))
                        last_node = internal_node
                        internal_node.suffix_link = self.root
                        # print(i, "Suffix link added from " + repr(last_node) + "  to root")
                else:
                    edge = self.select_edge()
                    edge.out[self.s[i]] = SuffixNode(i, self.end)
                    if last_node is not None:
                        last_node.suffix_link = edge
                        # print(i, "Suffix link added from " + repr(last_node) + "  to  " + repr(edge))
                    last_node = edge

                if self.a.active_node != self.root:
                    self.a.active_node = self.a.active_node.suffix_link

                else:
                    self.a.active_edge += 1
                    self.a.active_length -= 1
                self.rem -= 1

    def walk(self, j):
        edge = self.select_edge()

        if len(edge) < self.a.active_length:
            self.a.active_node = edge
            self.a.active_length = (self.a.active_length - len(edge))
            self.a.active_edge = edge.out[self.s[j]].start

        else:
            self.a.active_length += 1

    def get_next_character(self, i):
        edge = self.select_edge()

        if len(edge) >= self.a.active_length:
            return self.s[edge.start + self.a.active_length]

        elif len(edge) + 1 == self.a.active_length:
            if self.s[i] in edge.out:
                return self.s[i]

        else:
            self.a.active_node = edge
            self.a.active_edge = self.a.active_edge + len(edge) + 1
            self.a.active_length = self.a.active_length - len(edge) - 1
            return self.get_next_character(i)
        return None

    def select_edge(self):
        return self.a.active_node.out[self.s[self.a.active_edge]]

    def split(self, i):
        edge = self.select_edge()

        start_pos = edge.start
        edge.start += self.a.active_length
        internal_node = SuffixNode(start_pos, End(start_pos + self.a.active_length - 1))
        leaf_node = SuffixNode(i, self.end)
        internal_node.out[self.s[edge.start]] = edge
        internal_node.out[self.s[leaf_node.start]] = leaf_node

        self.a.active_node.out[self.s[internal_node.start]] = internal_node

        return internal_node

    def count_leaves(self):
        """Counts the number of leaves per node of the tree"""

        self.subtree_sizes = {}

        def helper(node: SuffixNode, st_size) -> int:
            if len(node.out) == 0:
                st_size[node] = 1
                return 1
            else:
                s = 0
                for k in node.out:
                    s += helper(node.out[k], st_size)
                st_size[node] = s
                return s

        size = helper(self.root, self.subtree_sizes)
        assert size == len(self.s)

    def build_suffix_array(self):
        """naive suffix array algorithm using lexicographical depth first search"""
        self.SA = array('l')

        def helper(u):
            if len(u.out) == 0:
                self.SA.append(u.index)
            else:
                for v in sorted(u.out):
                    helper(u.out[v])

        for node in sorted(self.root.out):
            helper(self.root.out[node])
        return self.SA

    def find_wrapper(self, t, rt=''):

        def helper(edge, pattern, i):
            if pattern[i] in edge.out:
                new_root = edge.out[pattern[i]]

                if len(new_root) + 1 >= len(pattern) - i:  # the whole pattern might be contained in this edge
                    k = 0
                    while k < len(pattern) - i:
                        if self.s[new_root.start + k] != pattern[i + k]:
                            print("pattern not in text")
                            return -1
                        k += 1
                    else:
                        if rt == 'node':
                            return new_root
                        else:
                            return new_root.start

                else:  # try to visit next edge
                    if self.s[new_root.start: new_root.end.end + 1] == pattern[i: i + len(new_root) + 1]:
                        i += len(new_root) + 1  # new starting position
                        return helper(new_root, pattern, i)
                    else:
                        return -1
            else:
                return -1

        if t == '':
            print("empty suffix exists")
            return False
        if t == '$':
            print("$ is used as a sentinel in this suffix tree")
            return True
        else:
            return helper(self.root, t, 0)

    def find(self, t) -> int:
        """If pattern is found in the text, returns the starting position of the pattern
        else return -1"""
        return self.find_wrapper(t, '')

    def number_of_occurrences(self, t):
        node = self.find_wrapper(t, 'node')
        assert self.subtree_sizes is not None
        return node.start, self.subtree_sizes[node]

    def euler_tour(self, source=None):
        E = []
        D = []
        R = {}
        if self.root is None:
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

        if source is None:
            source = self.root
        euler_visit(source, E, D, R, 0, 0)
        # print('->'.join(map(str, E)))
        return E, D, R

    def preprocess_lca(self):
        """perform an euler tour and save the depths of the nodes
        LCA(i, j) = E[RMQ_D(i, j)]"""

        # perform Euler tour and get the depths, representative, and node arrays
        self.E, self.D, self.R = self.euler_tour()
        sparse = array('l', [maxsize] * len(self.E))
        construct_sparse_table(self.D, sparse)
        self.sparse = sparse
    
    def get_lca(self):
        pass

    def create_labels(self) -> None:
        n = len(self.s)  # length of the text in the suffix tree

        def helper(node, carried_length):
            if len(node.out) == 0:  # we are at a leaf node
                node.index = n - (carried_length + len(node) + 1)

            else:
                j = len(node) + 1  # the number of characters in this edge
                for k in node.out:
                    helper(node.out[k], j + carried_length)

        for key in self.root.out:
            helper(self.root.out[key], 0)

    def _print_leaf_labels(self):
        """Debugging method to check whether self.create_labels() has been called successfully"""

        def helper(node, text):
            if len(node.out) == 0:
                print(node.index, ':', text[node.index:])
            else:
                for k in node.out:
                    helper(node.out[k], text)
        helper(self.root, self.s)

    def bwt(self):
        """Burrows-Wheeler Transform"""
        assert (self.SA is not None)
        n = len(self.SA)  # the length of BWT is the same as that of a SUFFIX ARRAY
        bwt = bytearray(n)
        for i in range(n):
            bwt[i] = ord(self.s[self.SA[i] - 1])
        return bwt

    def lcp_kasai(self):
        """Linear time LCP construction
        Has poor locality of reference
        """
        assert (self.SA is not None), "Empty Suffix Array"
        n = len(self.SA)
        h = 0
        ISA = [0] * n  # create an inverse suffix array/ also known as rank
        LCP = [0] * n
        for i in range(n):  # fill the inverse suffix array
            ISA[self.SA[i]] = i
        for i in range(n):
            if ISA[i] > 1:
                k = self.SA[ISA[i] - 1]
                while self.s[i + h] == self.s[k + h]:
                    h += 1
                LCP[ISA[i]] = h
                if h > 0:
                    h -= 1
        return LCP

    def prod_lcp_array(self):
        """O(n^2) time
        naive"""
        assert (self.SA is not None), "Empty Suffix Array"
        lcp_array = [0]
        for i in range(len(self.SA) - 1):  # O(n - 1) + O(n) + c1
            k = 0  # c_2
            while self.s[self.SA[i] + k] == self.s[self.SA[i + 1] + k]:
                k += 1
            lcp_array.append(k)
        return lcp_array

    def produce_text(self, sorted_=True):
        """Mainly for debugging purposes"""
        result = []

        def helper(node, s):
            if len(node.out) == 0:
                return result.append(s + self.s[node.start: node.end.end + 1])
            else:
                for char in node.out:
                    helper(node.out[char], s + self.s[node.start: node.end.end + 1])

        def helper_sorted(node, s):
            if len(node.out) == 0:
                return result.append(s + self.s[node.start: node.end.end + 1])
            else:
                for char in sorted(node.out):
                    helper(node.out[char], s + self.s[node.start: node.end.end + 1])

        if sorted_:
            for k in sorted(self.root.out):
                helper_sorted(self.root.out[k], '')
        else:
            for k in self.root.out:
                helper(self.root.out[k], '')

        return result


if __name__ == '__main__':
    # alpha = ['A', 'G', 'T', 'C']
    # while True:
    #     input_size = 300_000
    #     test = ''.join([choice(alpha) for _ in range(input_size)])
    #     ukk = SuffixTree(test)
    #     tic = datetime.now()
    #     ukk.build_suffix_tree()
    #     toc = datetime.now()
    #     print("Ukkonen ran in", (toc - tic).total_seconds(), "seconds for input size", input_size)
    #     ukk.SA = ukk.build_suffix_array()
    #     # t1 = datetime.now()
    #     # print("LCP naive ran in ", (datetime.now() - t1).total_seconds(), "for input size", input_size)
    #     # t2 = datetime.now()
    #     # ukk.lcp_kasai()
    #     # print("Kasai ran in ", (datetime.now() - t2).total_seconds(), "for input size", input_size)
    #     break

    ST = SuffixTree('abaabacadabra')
    ST.build_suffix_tree()
    ST.create_labels()
    ST.preprocess_lca()
    ST.count_leaves()
    print(ST.subtree_sizes[ST.root])
    # S = ST.produce_text()
    SA = ST.build_suffix_array()
    print(SA)
    # LCP = ST.prod_lcp_array()
    # LCP_KASAI = ST.lcp_kasai()
    # BWT = ST.bwt()
    # print(BWT)
