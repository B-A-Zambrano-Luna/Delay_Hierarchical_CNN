# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 15:56:52 2022

@author: LENOVO
"""

import numpy as np
import test_functions as test


class imate2test(object):
    def __init__(self, Image, Z_k, reduction=False):
        self.p = Z_k.get_prime()
        self.k = Z_k.get_radio()
        if not reduction:
            self.w, self.h = Image.shape
        else:
            k = Z_k.get_radio()
            p = Z_k.get_prime()
            self.w = p**(int(k/2))
            self.h = p**(k - int(k/2))
        self.test = []
        self.emmending = dict()
        self.leaves = [Image]
        self.image = Image
        self.Z_k = Z_k
        self.tree_temple = [np.ones((self.w, self.h))]

    def get_emmending(self):
        return self.emmending

    def get_image(self):
        return self.image

    def get_leaves(self):
        return self.leaves

    def fit(self):
        k = self.k
        p = self.p
        leaves = self.leaves
        tree_temple = self.tree_temple

        for level in range(1, k+1):

            if level % 2 == 1:
                for leaf_index in range(p**(level-1)):

                    leaf = leaves.pop(0)

                    leaves += np.array_split(leaf, p, axis=1)

                    leaf_aux = tree_temple.pop(0)
                    tree_temple += \
                        np.array_split(leaf_aux, p, axis=1)
            elif level % 2 == 0:
                for leaf_index in range(p**(level-1)):

                    leaf = leaves.pop(0)

                    leaves += np.array_split(leaf, p, axis=0)

                    leaf_aux = tree_temple.pop(0)
                    tree_temple += \
                        np.array_split(leaf_aux, p, axis=0)
            Z_k = self.Z_k
            Zipleaves = zip(Z_k, leaves)
            self.emmending = dict(Zipleaves)

    def get_test(self):
        k = self.k
        p = self.p
        adds = []
        emmending = self.emmending
        for leaf in emmending:
            value_leaf = emmending[leaf].mean()
            adds.append(value_leaf *
                        test.char_function(leaf,
                                           -k, p))
        return test.test_function(adds)

    def get_values(self):
        adds = []
        emmending = self.emmending
        for leaf in emmending:

            if emmending[leaf].size == 0:
                value_leaf = 0
            else:
                value_leaf = emmending[leaf].mean()

            adds.append(value_leaf)
        return np.array(adds)

    def inverse_transform(self, f):
        p = self.p
        k = self.k
        values = []
        try:
            if len(f) == p**k:
                values = f
        except:
            Z_k = self.Z_k
            for i in Z_k:
                values.append(f(i))

        copy_temple = self.tree_temple.copy()

        level = k
        while level >= 1:

            if level % 2 == 1:
                for node in range(p**(level-1)):

                    branch = []
                    for i in range(p):

                        leaf = copy_temple.pop(0)

                        if level == k:
                            leaf = values[i+node*p] * leaf

                        branch.append(leaf)

                    copy_temple\
                        .append(np.concatenate
                                (branch, axis=1))
            elif level % 2 == 0:
                for node in range(p**(level-1)):

                    branch = []
                    for i in range(p):

                        leaf = copy_temple.pop(0)

                        if level == k:
                            leaf = values[i+node*p] * leaf

                        branch.append(leaf)

                    copy_temple\
                        .append(np.concatenate
                                (branch, axis=0))

            level = level - 1
        return copy_temple[0]
