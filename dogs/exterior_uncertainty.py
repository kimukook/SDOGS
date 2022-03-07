"""
This is the class ExteriorUncertainty that stores the main functions to compute exterior uncertainty function used in SDOGS algorithm

=====================================
Author  :  Muhan Zhao
Date    :  Mar. 1, 2022
Location:  UC San Diego, La Jolla, CA
=====================================


MIT License

Copyright (c) 2022 Muhan Zhao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy as np
from dogs import Utils


def update_exterior_uncertainty_params(sdogs):
    sdogs.Rmax, sdogs.max_dis = max_circumradius_delaunay_simplex(sdogs)
    # Determine the parameters b and c for exterior uncertainty function.
    sdogs.b, sdogs.c, status = exterior_uncertainty_parameter_solver(sdogs.Rmax, sdogs.max_dis)


def unevaluated_vertices_identification(simplex, xE):
    """
    Outline

    Determine whether or not vertices of the query Delaunay simplex is unevaluated.

    ---------
    Parameters

    :param simplex:   Query Delaunay simplex.
    :param xE     :   Evaluated data set.

    ---------
    Output

    :return: exist:   Unevaluated vertex exists or not.
    :return: index:   The indices of evaluated data point in simplex. For optimization solver usage.

    """
    unevaluated_exist = False
    N = simplex.shape[1]
    index = np.zeros(N, dtype=int)
    for i in range(N):
        vertex = simplex[:, i].reshape(-1, 1)
        val, idx, x_nn = Utils.mindis(vertex, xE)
        if val == 0:  # query vertex i exists in evaluated point set
            index[i] = 1  # indicate that this vertex is evaluated
        else:
            unevaluated_exist = True  # indicate that unevaluated point exists in query simplex.
    return unevaluated_exist, index


def max_circumradius_delaunay_simplex(sdogs):
    """
    Outline
    Determine the following parameters for the exterior uncertainty function

    Rmax   :    Determine the maximum circumradius of the safe Delaunay simplex;
    max_dis:    Determine the maximum min distance of unsafe point to the safe point.

    ----------
    Parameters

    :param sdogs:   SafeDogs class object;

    ----------
    Output

    :return Rmax    :   float
    :return max_dis :   float
    """
    n = sdogs.xi.shape[0]
    N = sdogs.tri.shape[0]
    Rmax = -1e+20
    max_dis = -1e+20
    for ii in range(N):
        # For each simplex, if interior Delaunay simplex, search for Rmax (maximum circumradius)
        #                   if exterior Delaunay simplex, search for M=maxmin distance of unexplored point to evaluated.
        simplex = np.copy(sdogs.xi[:, sdogs.tri[ii, :]])
        unevaluated_exist = unevaluated_vertices_identification(simplex, sdogs.xE)[0]
        if not unevaluated_exist:
            R2, xc = Utils.circhyp(sdogs.xi[:, sdogs.tri[ii, :]], n)
            Rmax = max(Rmax, np.sqrt(R2))
        else:
            # TODO for 2D case, this should be acceptable, but for higher dimension, ....
            for i in range(n + 1):
                vertex = simplex[:, i].reshape(-1, 1)
                val, idx, x_nn = Utils.mindis(vertex, sdogs.xE)
                if val == 0:
                    # vertex evaluated, pass
                    pass
                else:
                    max_dis = max(val, max_dis)
    if Rmax == -1e+20:  # No interior Delaunay simplex exists, usually this is for the very first iteration.
        # The goal is to set exterior and interior
        Rmax = 0.1
    return Rmax, max_dis


def exterior_uncertainty_parameter_solver(Rmax, max_dis):
    """
    Outline
    Determine the parameter b and c for uncertainty function in unsafe region using bisection method.

    ----------
    Parameters

    :param Rmax   :     The maximum circumradius of the safe Delauany simplex;
    :param max_dis:     The maximum min distance of unsafe point to the safe point.

    ----------
    Output

    :return b:  float
    :return c:  float
    """
    f = lambda x: (x - 1) / (x * (max_dis + (2 * Rmax / x) ** (1 / (x - 1))) ** x) + (1 / (2 * Rmax ** 2))
    low_bnd = 0.001
    upp_bnd = 0.9
    u = f(low_bnd)
    v = f(upp_bnd)
    iter = 1000
    e = upp_bnd - low_bnd
    delta = 0.00001
    eps = delta
    if np.sign(u) == np.sign(v):
        status = 'Boundaries have the same sign.'
        b = .01
    else:
        # binary search the solution of b
        status = 'Maximum iteration achieved'
        for i in range(iter):
            e /= 2
            mid = low_bnd + e
            f_mid = f(mid)
            if e < delta:
                status = 'Required error of subinterval bound achieved'
                break
            elif abs(f_mid) < eps:
                status = 'Function value tolerance achieved';
                break
            else:
                if np.sign(f_mid) != np.sign(u):
                    upp_bnd = np.copy(mid)
                    v = f_mid
                else:
                    low_bnd = np.copy(mid)
                    u = f_mid
        b = mid
    c = (2 * Rmax / b) ** (b - 1)
    return b, c, status


def exterior_uncertainty(x, sdogs):
    """
    Outline

    Determine the discrete min uncertainty function value at position x.
    e(x) = (||x - x'||_2 + c)^(b) - c^(b)

    ---------
    Parameters

    :param x    :   n-by-1 2D array, np.ndarray; The query point
    :param sdogs:   SafeDogs class object;

    ---------
    Output

    :return f:  1-by-, 1D array, np.ndarray; The value of exterior uncertainty function at query
    :return g:  n-by-1 2D array, np.ndarray; The gradient of exterior uncertainty function at query
    :return h:  n-by-n 2D array, np.ndarray; The hessian of exterior uncertainty function at query
    """
    x = x.reshape(-1, 1)
    assert sdogs.xE.shape[1] >= 1, 'Evaluated data set xE should have more than 1 data point'
    assert x.shape[0] == sdogs.xE.shape[0], 'x has different dimensions with xE.'

    dis, idx, x_nn = Utils.mindis(x, sdogs.xE)
    dis = (1e-10 if dis < 1e-10 else dis)

    f = (dis + sdogs.c) ** sdogs.b - sdogs.c ** sdogs.b
    f = np.atleast_1d(f)  # Must be an array, in case of future concatenation and other matrix operations
    g = sdogs.b * (dis + sdogs.c) ** (sdogs.b - 1) * (x - x_nn) / dis
    h = sdogs.b * (sdogs.b - 1) * (dis + sdogs.c) ** (sdogs.b - 2) * np.dot((x - x_nn), (x - x_nn).T) / dis ** 2 \
        + sdogs.b * (dis + sdogs.c) ** (sdogs.b - 1) * np.identity(sdogs.n) / dis \
        - sdogs.b * (dis + sdogs.c) ** (sdogs.b - 1) * np.dot((x - x_nn), (x - x_nn).T) / dis ** 3
    return f, g, h

