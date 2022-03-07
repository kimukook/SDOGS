# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 17:40:41 2017

@author: KimuKook, fixed version on Nov. 5, 2018.
"""
import  numpy   as np
from    dogs    import Utils
import  matplotlib.pyplot as plt
np.set_printoptions(linewidth=200, precision=5, suppress=True)


def vertex_find(A, b, lb, ub):
    '''
    Find the vertices of a simplex;
    Ax<b: Linear constraints;
    lb<x<ub: actual constraints;
    Attension: Do not try to include lb&ub inside A&b matrix.
    Example:
    A = np.array([[-1, 1],[1, -1]])
    b = np.array([[0.5], [0.5]])
    lb = np.zeros((2, 1))
    ub = np.ones((2, 1))
    :param A:
    :param b:
    :param lb:
    :param ub:
    :return:
    '''
    if A.shape[0] == 0 and b.shape[0] == 0:
        if len(lb) != 0 and len(ub) != 0:
            Vertex = Utils.bounds(lb, ub, len(lb))
        else:
            Vertex = []
            raise ValueError('All inputs A, b, lb and ub have dimension 0.')
    else:
        if len(lb) != 0:
            Vertex = np.matrix([[], []])
            m = A.shape[0]
            n = A.shape[1]
            if m == 0:
                Vertex = Utils.bounds(lb, ub, len(lb))
            else:
                for r in range(min(n, m) + 1):
                    from itertools import combinations
                    nlist = np.arange(1, m + 1)
                    C = np.array([list(c) for c in combinations(nlist, r)])
                    nlist2 = np.arange(1, n + 1)
                    D = np.array([list(d) for d in combinations(nlist2, n - r)])
                    if r == 0:
                        F = Utils.bounds(lb, ub, n)
                        for kk in range(F.shape[1]):
                            x = np.copy(F[:, kk]).reshape(-1, 1)
                            if np.all(np.dot(A, x) - b < 0):
                                Vertex = np.column_stack((Vertex, x))
                    else:
                        for ii in range(len(C)):
                            index_A = np.copy(C[ii])
                            v1 = np.arange(1, m + 1)
                            index_A_C = np.setdiff1d(v1, index_A)
                            A1 = np.copy(A[index_A - 1, :])
                            b1 = np.copy(b[index_A - 1])
                            for jj in range(len(D)):
                                index_B = np.copy(D[jj])
                                v2 = np.arange(1, n + 1)
                                index_B_C = np.setdiff1d(v2, index_B)
                                if len(index_B) != 0 and len(index_B_C) != 0:
                                    F = Utils.bounds(lb[index_B - 1], ub[index_B - 1], n - r)
                                    A11 = np.copy(A1[:, index_B - 1])
                                    A12 = np.copy(A1[:, index_B_C - 1])
                                    for kk in range(F.shape[1]):
                                        A11 = np.copy(A1[:, index_B - 1])
                                        A12 = np.copy(A1[:, index_B_C - 1])
                                        xd = np.linalg.lstsq(A12, b1 - np.dot(A11, F[:, kk].reshape(-1, 1)), rcond=None)[0]
                                        x = np.zeros((n, 1))
                                        x[index_B - 1] = np.copy(F[:, kk])
                                        x[index_B_C - 1] = np.copy(xd)
                                        if r == m or (np.dot(A[index_A_C - 1, :], x) - b[index_A_C - 1]).min() < 0:
                                            if (x - ub).max() < 1e-6 and (x - lb).min() > -1e-6 and Utils.mindis(x, Vertex)[0] > 1e-6:
                                                Vertex = np.column_stack((Vertex, x))
        else:
            m = A.shape[0]
            n = A.shape[1]
            from itertools import combinations
            nlist = np.arange(1, m + 1)
            C = np.array([list(c) for c in combinations(nlist, n)])
            Vertex = np.empty(shape=[n, 0])
            for ii in range(len(C)):
                index_A = np.copy(C[ii])
                v1 = np.arange(1, m + 1)
                index_A_C = np.setdiff1d(v1, index_A)
                A1 = np.copy(A[index_A - 1, :])
                b1 = np.copy(b[index_A - 1])
                A2 = np.copy(A[index_A_C - 1])
                b2 = np.copy(b[index_A_C - 1])
                x = np.linalg.lstsq(A1, b1, rcond=None)[0]
                if (np.dot(A2, x) - b2).max() < 1e-6:
                    Vertex = np.column_stack((Vertex, x))
        # cant plot Vertex directly. must transform it into np.array.
    return np.array(Vertex)


def plot_vertex(V):
    for i in range(V.shape[1] - 1):
        plt.plot(V[0, [i, i+1]], V[1, [i, i+1]], c='b')
    plt.plot(V[0, [-1, 0]], V[1, [-1, 0]], c='b')
    plt.show()

# test case
# plot the simplex

#
#
# n = 2
# Acons = np.vstack(( np.array([[0.4, 1]]) ))
# bcons = np.vstack(( np.array([[0.6]]) ))
# lb = np.zeros((n, 1))
# ub = np.ones((n, 1))
# V = vertex_find(-Acons, -bcons, lb, ub)
# print(V)
# plot_vertex(V)

# # test case, for box domain
# n = 2
# # For box domain, just use one dimensional empty vector.
# A = np.array([])
# b = np.array([])
# lb = np.zeros((n, 1))
# ub = np.ones((n, 1))
# V = vertex_find(A, b, lb, ub)
# print(V)
# plot_vertex(V)
