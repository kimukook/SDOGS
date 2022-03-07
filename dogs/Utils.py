#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 15:19:28 2017

@author: mousse
"""
import  numpy   as np
from    dogs    import SafeLearn
'''
Utils_SL.py is implemented to generate results for AlphaDOGS and DeltaDOGS applied on safe learning problems.
    - The objective function is denoted as F(x) and safe constraints are denoted as G(x), both are unknown to the user.
    - The initial safe experiment is given by the user. The performance of the initial experiment might be poor, but the 
        safe constraints will be guaranteed. 
    - Our goal is to find the parameters that satisfied the safe constraints and obtain the best performance.
     
    - The safe constraints will always be guaranteed during the optimization process.
     
This script contains following functions:
    
    bounds          :       Generate vertices under lower bound 'bnd1' and upper bound 'bnd2' for 'n' dimensions;
    mindis          :       Generate the minimum distances and corresponding point from point x to set xi;
    circhyp         :       Generate hypercircumcircle for Delaunay simplex, return the circumradius and center;
    normalize_bounds:       Normalize bounds for Delta DOGS optm solver;
    physical_bounds :       Retransform back to the physical bounds for function evaluation;
    
    search_bounds   :       Generate bounds for the Delaunay simplex during the continuous search minimization process;
    hyperplane_equations:   Determine the math expression for the hyperplane;
     
    test_fun        :       Return the test function info;
    fun_eval        :       Transform point from normalized bounds to the physical boudns;
    random_initial  :       Randomly generate the initial points.
    
'''
################################# Utils ####################################################


def bounds(bnd1, bnd2, n):
    #   find vertex of domain for a box domain.
    #   INPUT: n: dimension, bnd1: lower bound, bnd2: upper bound.
    #   OUTPUT: vertex of domain. 2^n number vector of n-D.
    #   Example:
    #           n = 3
    #           bnd1 = np.zeros((n, 1))
    #           bnd2 = np.ones((n, 1))
    #           bnds = bounds(bnd1,bnd2,n)
    #   Author: Shahoruz Alimohammadi
    #   Modified: Dec., 2016
    #   DELTADOGS package
    assert bnd1.shape == (n, 1) and bnd2.shape == (n, 1), 'lb(bnd1) and ub(bnd2) should be 2 dimensional vector.'
    bnds = np.kron(np.ones((1, 2 ** n)), bnd2)
    for ii in range(n):
        tt = np.mod(np.arange(2 ** n) + 1, 2 ** (n - ii)) <= 2 ** (n - ii - 1) - 1
        bnds[ii, tt] = bnd1[ii]
    return bnds


def mindis(x, xi):
    '''
    calculates the minimum distance from all the existing points
    :param x: x the new point
    :param xi: xi all the previous points
    :return: [ymin ,xmin ,index]
    '''
    x = x.reshape(-1, 1)
    assert xi.shape[1] != 0, 'The set of Evaluated points has size 0!'
    dis = np.linalg.norm(xi-x, axis=0)
    val = np.min(dis)
    idx = np.argmin(dis)
    xmin = xi[:, idx].reshape(-1, 1)
    return val, idx, xmin


def modichol(A, alpha, beta):
    #   Modified Cholesky decomposition code for making the Hessian matrix PSD.
    #   Author: Shahoruz Alimohammadi
    #   Modified: Jan., 2017
    n = A.shape[1]  # size of A
    L = np.identity(n)
    ####################
    D = np.zeros((n, 1))
    c = np.zeros((n, n))
    ######################
    D[0] = np.max(np.abs(A[0, 0]), alpha)
    c[:, 0] = A[:, 0]
    L[1:n, 0] = c[1:n, 0] / D[0]

    for j in range(1, n - 1):
        c[j, j] = A[j, j] - (np.dot((L[j, 0:j] ** 2).reshape(1, j), D[0:j]))[0, 0]
        for i in range(j + 1, n):
            c[i, j] = A[i, j] - (np.dot((L[i, 0:j] * L[j, 0:j]).reshape(1, j), D[0:j]))[0, 0]
        theta = np.max(c[j + 1:n, j])
        D[j] = np.array([(theta / beta) ** 2, np.abs(c[j, j]), alpha]).max()
        L[j + 1:n, j] = c[j + 1:n, j] / D[j]
    j = n - 1
    c[j, j] = A[j, j] - (np.dot((L[j, 0:j] ** 2).reshape(1, j), D[0:j]))[0, 0]
    D[j] = np.max(np.abs(c[j, j]), alpha)
    return np.dot(np.dot(L, (np.diag(np.transpose(D)[0]))), L.T)


def circhyp(x, N):
    # circhyp     Circumhypersphere of simplex
    #   [xC, R2] = circhyp(x, N) calculates the coordinates of the circumcenter
    #   and the square of the radius of the N-dimensional hypersphere
    #   encircling the simplex defined by its N+1 vertices.
    #   Author: Shahoruz Alimohammadi
    #   Modified: Jan., 2017
    #   DOGS package

    test = np.sum(np.transpose(x) ** 2, axis=1)
    test = test[:, np.newaxis]
    m1 = np.concatenate((np.matrix((x.T ** 2).sum(axis=1)), x))
    M = np.concatenate((np.transpose(m1), np.matrix(np.ones((N + 1, 1)))), axis=1)
    a = np.linalg.det(M[:, 1:N + 2])
    c = (-1.0) ** (N + 1) * np.linalg.det(M[:, 0:N + 1])
    D = np.zeros((N, 1))
    for ii in range(N):
        M_tmp = np.copy(M)
        M_tmp = np.delete(M_tmp, ii + 1, 1)
        D[ii] = ((-1.0) ** (ii + 1)) * np.linalg.det(M_tmp)
        # print(np.linalg.det(M_tmp))
    # print(D)
    xC = -D / (2.0 * a)
    #	print(xC)
    R2 = (np.sum(D ** 2, axis=0) - 4 * a * c) / (4.0 * a ** 2)
    #	print(R2)
    return R2, xC


def normalize_bounds(x0, lb, ub):
    n = len(lb)  # n represents dimensions
    m = x0.shape[1]  # m represents the number of sample data
    x = np.copy(x0)
    for i in range(n):
        for j in range(m):
            x[i][j] = (x[i][j] - lb[i]) / (ub[i] - lb[i])
    return x


def physical_bounds(x, lb, ub):
    '''
    :param x : normalized point
    :param lb: real lower bound
    :param ub: real upper bound
    :return: physical scale of the point
    '''
    x_phy = np.copy(x)
    n = len(lb)  # n represents dimensions
    try:
        m = x_phy.shape[1]  # m represents the number of sample data
    except:
        m = x_phy.shape[0]
    for i in range(n):
        for j in range(m):
            x_phy[i][j] = (x[i][j])*(ub[i] - lb[i]) + lb[i]

    return x_phy


def fun_eval(fun, lb, ub, x):
    x = x.reshape(-1, 1)
    x_phy = physical_bounds(x, lb, ub)
    y = fun(x_phy)
    return y


def search_bounds(xi):
    n = xi.shape[0]
    srch_bnd = []
    for i in range(n):
        rimin = np.min(xi[i, :])
        rimax = np.max(xi[i, :])
        temp = (rimin, rimax)
        srch_bnd.append(temp)
    simplex_bnds = tuple(srch_bnd)
    return simplex_bnds


def search_simplex_bounds(xi):
    '''
    Return the n+1 constraints defined by n by n+1 Delaunay simplex xi.
    The optimization for finding minimizer of Sc should be within the Delaunay simplex.
    :param xi: xi should be (n) by (n+1). Each column denotes a data point.
    :return: Ax >= b constraints.
    A: n+1 by n
    b: n+1 by 1
    '''
    n = xi.shape[0]  # dimension of input
    m = xi.shape[1]  # number of input, should be exactly the same as n+1.
    # The linear constraint, which is the boundary of the Delaunay triangulation simplex.
    A = np.zeros((m, n))
    b = np.zeros((m, 1))
    for i in range(m):
        direction_point = xi[:, i].reshape(-1, 1)  # used to determine the type of inequality, <= or >=
        plane_points = np.delete(xi, i, 1)  # should be an n by n square matrix.
        A[i, :], b[i, 0] = hyperplane_equations(plane_points)
        if np.dot(A[i, :].reshape(-1, 1).T, direction_point) < b[i, :]:
            # At this point, the simplex stays at the negative side of the equation, assign minus sign to A and b.
            A[i, :] = np.copy(-A[i, :].reshape(-1, 1).T)
            b[i, 0] = np.copy(-b[i, :])
    return A, b


def hyperplane_equations(points):
    '''
    Return the equation of n points hyperplane in n dimensional space.

    Reference website:
    https://math.stackexchange.com/questions/2723294/how-to-determine-the-equation-of-the-hyperplane-that-contains-several-points

    :param points: Points is an n by n square matrix. Each column represents a data point.
    :return: A and b (both 2 dimensional array) that satisfy Ax = b.
    '''
    n, m = points.shape  # n dimension of points. m should be the same as n
    base_point = points[:, -1].reshape(-1, 1)
    matrix = (points - base_point)[:, :-1].T  # matrix should be n-1 by n, each row represents points - base_point.
    A = np.zeros((1, n))
    b = np.zeros((1, 1))
    for j in range(n):
        block = np.delete(matrix, j, 1)  # The last number 1, denotes the axis. 1 is columnwise while 0 is rowwise.
        A[0, j] = (-1) ** (j+1) * np.linalg.det(block)
    b[0, 0] = np.dot(A, base_point)
    return A, b


def test_fun(fun_arg, n):
    fun = lb = ub = xmin = y0 = fname = []
    if fun_arg == 1:  # 2D test function: Goldstein-price
        # Notice, here I take ln(y)
        lb = -2 * np.ones((2, 1))
        ub = 2 * np.ones((2, 1))
        fun = lambda x: np.log((1+(x[0]+x[1]+1)**2*(19-14*x[0]+3*x[0]**2-14*x[1]+6*x[0]*x[1]+3*x[1]**2))*\
                        (30+(2*x[0]-3*x[1])**2*(18-32*x[0]+12*x[0]**2+48*x[1]-36*x[0]*x[1]+27*x[1]**2)))
        y0 = np.log(3)
        xmin = np.array([0.5, 0.25])
        fname = 'Goldstein-price'

    elif fun_arg == 2:  # schwefel
        lb = np.zeros((n, 1))
        ub = np.ones((n, 1))
        fun = lambda x: - sum(np.multiply(500 * x, np.sin(np.sqrt(abs(500 * x))))) / 250
        y0 = -1.6759316 * n  # targert value for objective function
        xmin = 0.8419 * np.ones((n, 1))
        fname = 'Schewfel'

    elif fun_arg == 3:  # rastinginn
        A = 3
        lb = -2 * np.ones((n, 1))
        ub = 2 * np.ones((n, 1))
        fun = lambda x: (sum((x - 0.7) ** 2 - A * np.cos(2 * np.pi * (x - 0.7)))) / 1.5
        y0 = 0.0
        xmin = 0.7 * np.ones((n, 1))
        fname = 'Rastinginn'
    # fun_arg == 4: Lorenz Chaotic system.

    elif fun_arg == 5:  # schwefel + quadratic
        fun = lambda x: - x[0][0] / 2 * np.sin(np.abs(500 * x[0][0])) + 10 * (x[1][0] - 0.92) ** 3
        lb = np.zeros((n, 1))
        ub = np.ones((n, 1))
        y0 = -0.44528425
        xmin = np.array([0.89536, 0.94188])
        fname = 'Schwefel + Quadratic'

    elif fun_arg == 6:  # Griewank function
        fun = lambda x: 1 + 1 / 4 * ((x[0][0] - 0.67) ** 2 + (x[1][0] - 0.21) ** 2) - np.cos(x[0][0]) * np.cos(
            x[1][0] / np.sqrt(2))
        lb = np.zeros((n, 1))
        ub = np.ones((n, 1))
        y0 = 0.08026
        xmin = np.array([0.21875, 0.09375])
        fname = 'Griewank'

    elif fun_arg == 7:  # Shubert function
        tt = np.arange(1, 6)
        fun = lambda x: np.dot(tt, np.cos((tt + 1) * (x[0][0] - 0.45) + tt)) * np.dot(tt, np.cos(
            (tt + 1) * (x[1][0] - 0.45) + tt))
        lb = np.zeros((n, 1))
        ub = np.ones((n, 1))
        y0 = -32.7533
        xmin = np.array([0.78125, 0.25])
        fname = 'Shubert'

    elif fun_arg == 8:  # 2D DimRed test function:  the Branin function
        a = 1
        b = 5.1 / (4*np.pi**2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8*np.pi)
        fun = lambda x: a*(x[1] - b*x[0]**2 + c*x[0] - r)**2 + s*(1-t)*np.cos(x[0]) + s
        lb = np.array([-5, 0]).reshape(-1, 1)
        ub = np.array([10, 15]).reshape(-1, 1)
        y0 = 0.397887
        # 3 minimizer
        xmin = np.array([0.5427728, 0.1516667])  # True minimizer np.array([np.pi, 2.275])
        # xmin2 = np.array([-np.pi, 12.275])
        # xmin3 = np.array([9.42478, 2.475])
        fname = 'Branin'

    elif fun_arg == 9:  # Six hump camel back function
        fun = lambda x: (4 - 2.1 * (x[0][0] * 3) ** 2 + (x[0][0] * 3) ** 4 / 3) * (x[0][0] * 3) ** 2 - (
                x[0][0] * x[1][0] * 6) + (-4 + 16 * x[1][0] ** 2) * x[1][0] ** 2 * 4
        lb = np.zeros((n, 1))
        ub = np.ones((n, 1))
        y0 = -1.0316
        xmin = np.array([0.029933, 0.3563])
        fname = 'Six hump camel'

    elif fun_arg == 10:  # Rosenbrock function
        fun = lambda x: np.sum(100*(x[:-1] - x[1:]**2)**2) + np.sum((x-1)**2)
        lb = np.zeros((n, 1))
        ub = 2 * np.ones((n, 1))
        y0 = 0
        xmin = normalize_bounds(np.ones((n, 1)), lb, ub).T[0]
        fname = 'Rosenbrock'

    elif fun_arg == 11:    # 3D Hartman 3
        lb = np.zeros((3, 1))
        ub = np.ones((3, 1))
        alpha = np.array([1,1.2,3.0,3.2])
        A = np.array([[3, 10, 30],
                      [0.1, 10, 35],
                      [3, 10, 30],
                      [0.1, 10, 35]])
        P = 1e-4*np.array([[3689, 1170, 2673],
                           [4699, 4387, 7470],
                           [1091, 8732, 5547],
                           [381 , 5743, 8828]])
        fun = lambda x: -np.dot(alpha, np.exp(-np.diag(np.dot(A, (np.tile(x, 4) - P.T)**2))))
        y0 = -3.86278
        xmin = np.array([0.114614, 0.555649, 0.852547])
        fname = 'Hartman 3'

    elif fun_arg == 12:  # 6D Hartman 6
        lb = np.zeros((6, 1))
        ub = np.ones((6, 1))
        alpha = np.array([1, 1.2, 3, 3.2])
        A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                      [0.05, 10, 17, 0.1, 8, 14],
                      [3, 3.5, 1.7, 10, 17, 8],
                      [17, 8, 0.05, 10, 0.1, 14]])
        P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                            [2329, 4135, 8307, 3736, 1004, 9991],
                            [2348, 1451, 3522, 2883, 3047, 6650],
                            [4047, 8828, 8732, 5743, 1091, 381]])
        # Notice, take -ln(-y)
        fun = lambda x: -np.log(-(-np.dot(alpha, np.exp(-np.diag(np.dot(A, (np.tile(x, 4) - P.T)**2))))))
        y0 = -np.log(-(-3.32237))
        xmin = np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])
        fname = 'Hartman 6'

    elif fun_arg == 13:  # 12D Toy test: Quadratic, 1 main direction
        a = np.arange(1, 13) / 12
        c = 0.01
        b = np.hstack((50 * np.ones(2), np.tile(c, 10)))
        lb = np.ones((n, 1))
        ub = np.zeros((n, 1))
        fun = lambda x: np.sum(b*(x.T[0] - a) ** 2)

    elif fun_arg == 15:  # 10D, schwefel 1D + flat quadratic 9D:
        fun = lambda x: - sum(np.multiply(500 * x[0], np.sin(np.sqrt(abs(500 * x[0]))))) / 250 + np.dot(0.001 * np.arange(2, 11).reshape(-1, 1).T, x[1:]**2)[0, 0]
        y0 = -1.675936
        lb = np.zeros((10, 1))
        ub = np.ones((10, 1))
        xmin = np.hstack((np.array([0.8419]), np.zeros(9)))
        fname = 'DR First Test'

    elif fun_arg == 16:
        # 12D: VERY BAD TEST PROBLEM. because alg starts with the relative error very small, and failed to improve it.
        fun = lambda x: np.exp(0.2 * x[0])[0] + np.exp(0.2 * x[1])[0] + (10*(x[1] - x[0] ** 2)**2 + (x[0] - 1)**2)[0] + np.sum(0.001 * (x[2:] - 0.1 * np.arange(3, 11).reshape(-1, 1)) ** 2 )
        lb = np.zeros((10, 1))
        ub = np.ones((10, 1))
        y0 = 2.341281845987039
        # y0 = -3withhold
        xmin = np.hstack((np.array([0.512, 0.723]), .1 * np.arange(3, 11)))
        fname = 'DR Second Test'

    elif fun_arg == 17:
        # DeltaDOGS + ASM high dimension of active subspace test.
        # 10D test problem, first 2D - Schwefel, the rest 8D are quadratic model.
        fun = lambda x: - sum(np.multiply(500 * x[:2], np.sin(np.sqrt(abs(500 * x[:2]))))) / 250 + .01 * np.dot( np.ones((1, 8)), (x[2:] - 0.1 * np.arange(3, 11).reshape(-1, 1))**2 )[0]
        lb = np.zeros((10, 1))
        ub = np.ones((10, 1))
        y0 = -3.3518
        xmin = np.hstack(( .8419 * np.ones(2), np.arange(3, 11) ))
        fname = 'DR Third test'

    elif fun_arg == 18:
        fun = lambda x: np.exp(0.7 * x[0] + 0.3 * x[1])
        lb = np.zeros((10, 1))
        ub = np.ones((10, 1))
        y0 = -3.3518
        xmin = np.hstack(( .8419 * np.ones(2), np.arange(3, 11) ))
        fname = 'DR exp test'

    elif fun_arg == 19:  # quadrotor design 2D
        lb = np.ones((n, 1)) * -1
        ub = np.zeros((n, 1))
        fun = lambda x: quadrotor_cost_function(x)[0]
        fname = 'quadrotor'
        # Note that the global min is defined within the safe region, the maximum position of the trajectory < 3.3
        y0 = 1.2210
        xmin = np.array([0, 0])

    return fun, lb, ub, y0, xmin, fname


def quadrotor_cost_function(k):
    k = k.reshape(-1, 1)
    x_des, y_des = quadrotor_desired_trajectory()
    x = quadrotor_trajectory_calculator(k)
    cost = np.linalg.norm(x - x_des)
    safe_cost = 3.2 - np.max(x)
    return cost, safe_cost


def quadrotor_desired_trajectory():
    x_des = np.arctan(5 * np.linspace(0, 4, 350) - 2.5) + 1.6

    # t = np.linspace(0, 5, 350)
    # x1 = t[0:70] ** 2
    # x2 = -(t[70:140] - 2) ** 2 + 2
    # length = t.shape[0] - x1.shape[0] - x2.shape[0]
    # x3 = x2[-1] * np.ones(length)
    # x_des = np.hstack((x1, x2, x3))

    y_des = np.linspace(0, 2, 350)
    return x_des, y_des


def quadrotor_trajectory_calculator(k):
    x_des, y_des = quadrotor_desired_trajectory()
    n = x_des.shape[0]
    xdot_des, ydot_des = reference_velocity(x_des, y_des)

    x    = np.zeros(n)
    y    = np.zeros(n)
    xdot = np.zeros(xdot_des.shape[0])
    ydot = np.zeros(ydot_des.shape[0])

    x[0] = x_des[0]
    y[0] = y_des[0]

    g  = 9.8
    dt = 5 / n
    psi = np.pi / 3
    for i in range(1, n):
        phi     = k[0] * (x[i-1] - x_des[i-1]) + k[1] * (xdot[i-1] - xdot_des[i-1])
        theta   = k[0] * (y[i-1] - y_des[i-1]) + k[1] * (ydot[i-1] - ydot_des[i-1])

        c = g / (np.cos(theta) * np.cos(phi))

        xddot   = c * (np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi))
        yddot   = c * (np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi))

        xdot[i] = xdot[i-1] + xddot * dt
        ydot[i] = ydot[i-1] + yddot * dt
        x[i]    = x[i-1] + xdot[i] * dt
        y[i]    = y[i-1] + ydot[i] * dt

    return x


def reference_velocity(x, y):
    n = x.shape[0]
    dt = 5 / n
    xdot = np.zeros(n)
    ydot = np.zeros(n)
    for i in range(1, n):
        xdot[i] = (x[i] - x[i - 1]) / dt
        ydot[i] = (y[i] - y[i - 1]) / dt
    return xdot, ydot


def test_safe_fun(safe_fun_arg, n):
    # ============================================================
    # NOTICE: The safe initial x0 must be on the initial mesh grid!
    # ============================================================
    # Lower bound and upper bound of the parameter space should be controlled by
    # utility function.
    safe_fun = safe_fun_name = x0 = L_safe = M = []
    if safe_fun_arg == 1:  # nD sinusoid safety function
        # The global minimum is included in the safe region for schwefel.
        # This safety function is the sin wave for each direction.
        # The safe region is the union of [.1, .9] for each direction.
        safe_fun = lambda x: np.sin( (x - 0.1) * 5 / 4 * np.pi)
        # L_safe: Lipschitz constant of safe constraint
        L_safe   = 4
        x0 = 0.25 * np.ones((n, 1))  # just for temporary use
        safe_fun_name = 'sin wave1'
        M = n

    elif safe_fun_arg == 2:
        # The global minimum is not included in the safe region for 1D schwefel
        safe_fun = lambda x: np.sin( x * 5 / 4 * np.pi)
        # L_safe: Lipschitz constant of safe constraint
        L_safe   = 4
        x0       = np.array([[0.25]])
        safe_fun_name = 'sin wave2'
        M = 1

    elif safe_fun_arg == 3:  # virtual wall standing at x = 3.3, try to avoid hitting the wall at final position.
        safe_fun = lambda x: np.array([[ quadrotor_cost_function(x)[1] ]])
        L_safe   = 3.5
        x0       = np.array([[0.1676], [0.1242]])
        safe_fun_name = 'quadrotor trajectory position'
        M = 1

    elif safe_fun_arg == 4:  # 2 safety functions, 2D params - nonconvex safe region.
        # First constraint: parabola; Second constraint: linear equation.
        safe_fun = lambda x: np.array([ x[0]**2 + x[1] ** 2 - 0.25,  1.9 - x[0] - x[1] ])
        L_safe   = 4
        x0   = np.array([[0.25], [0.75]])
        M = 2

    return safe_fun, x0, L_safe, M, safe_fun_name


def safe_eval_estimate(xE, y_safe, L_safe, x):
    x = x.reshape(-1, 1)
    val, idx, x_nn = mindis(x, xE)
    safe_estimate = y_safe[:, idx] - L_safe * val
    if (safe_estimate > 0).all():
        return True
    else:
        return False


def safe_mesh_quantizer(sdogs, xc):
    # TODO move this function to the safedogs class
    '''
    Quantize the minimizer of surrogate model onto the current mesh grid.
    The quantizer should be the closest and safe point to the minimizer on the mesh.
    :param sdogs:   The optimization class
    :param xc   :   The minimizer of surrogate model
    :return:
    '''
    # Generate the mesh grid in 1D
    MeshGrid = np.linspace(0, 1, sdogs.ms + 1)

    xc_grid = np.empty(shape=[sdogs.n, 0])
    turn = 0
    dis = 1e+10
    # Determine the unit rectangular on the current mesh grid that contains xc
    while xc_grid.shape[1] == 0:
        turn += 1
        GridBound = {}
        for i in range(sdogs.n):
            # Quantify each coordinate of xc onto 1D grid
            GridBound['{}'.format(i)] = MeshGrid[np.argsort(np.abs(MeshGrid - xc[i, :]))[:int(2*turn)]]
        mesh_grid = SafeLearn.meshgrid_generator(GridBound, sdogs.n)

        # query each point in mesh_grid, find the safe quantizer with the smallest surrogate surface value
        for i in range(mesh_grid.shape[1]):
            query = np.copy(mesh_grid[:, i].reshape(-1, 1))
            safe_criteria = np.min(sdogs.yS, axis=0) - sdogs.L_safe * np.linalg.norm(sdogs.xE - query, axis=0)
            if (safe_criteria > 0).any():
                query_surrogate_value = sdogs.surrogate_eval(sdogs, query)
                if query_surrogate_value < dis:
                    dis = query_surrogate_value
                    xc_grid = np.copy(query)
                else:
                    pass
            else:
                pass
    return xc_grid


# TODO I think we dont need add_sup anymore, we only need it when we combine xE and xU to keep track of the whole data set
#   delete this func. when you feel comfortable
def add_sup(sdogs):
    '''
    To avoid duplicate values in support points for Delaunay Triangulation.
    :param xE: Evaluated points.
    :param xU: Support points.
    :param ind_min: The minimum point's index in xE.
    return: Combination of unique elements of xE and xU and the index of the minimum yp.
    '''
    xmin = sdogs.xE[:, sdogs.ind_min]
    # Construct the concatenate of xE and xU and return the array that every column (point) is unique
    x_unique = np.copy(sdogs.xE)
    for x in sdogs.xU.T:
        dis, _, _ = mindis(x.reshape(-1, 1), x_unique)
        if dis > 1e-5:
            x_unique = np.hstack(( x_unique, x.reshape(-1, 1) ))
    # Find the minimum point's index: ind_min
    _, ind_min_new, _ = mindis(xmin.reshape(-1, 1), x_unique)
    return x_unique, ind_min_new


def unique_data(x):
    """
    Outline

    Find the column-wise unique elements in the given data set x
    ----------
    Parameters

    :param x:   n-by-N 2d np.ndarray;
    ----------
    Output

    :return :   n-by-(*) 2d np.ndarray;
    """
    _x = np.ascontiguousarray(x.T)
    unique_x = np.unique(_x.view([('', _x.dtype)] * _x.shape[1]))
    output = unique_x.view(_x.dtype).reshape((unique_x.shape[0], _x.shape[1])).T
    return output
