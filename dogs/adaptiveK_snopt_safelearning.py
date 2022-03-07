from        optimize        import snopta, SNOPT_options
import      numpy           as np
from        scipy.spatial   import Delaunay
import      scipy.io        as io
import      os
import      inspect
from        dogs            import Utils
from        dogs            import interpolation
from        dogs            import constantK_snopt_safelearning

'''
 adaptiveK_snopt.py file contains functions used for DeltaDOGS(Lambda) algorithm.
 Using the package optimize (SNOPT) provided by Prof. Philip Gill and Dr. Elizabeth Wong, UCSD.

 This is a script of DeltaDOGS(Lambda) dealing with linear constraints problem which is solved using SNOPT. 
 Notice that this scripy inplements the snopta function. (Beginner friendly)
 
 The adaptive-K continuous search function has the form:
 Sc(x) = (P(x) - y0) / K*e(x):

     Sc(x):     constant-K continuous search function;
     P(x):      Interpolation function: 
                    For AlphaDOGS: regressionparameterization because the function evaluation contains noise;
                    For DeltaDOGS: interpolationparameterization;
     e(x):      The uncertainty function constructed based on Delaunay triangulation.

 Function contained:
     tringulation_search_bound:   Search for the minimizer of continuous search function over all the Delaunay simplices 
                                  over the entire domain.
     Adoptive_K_search:           Search over a specific simplex.
     AdaptiveK_search_cost:       Calculate the value of continuous search function.


 LOG Dec. 4, 2018:   Function snopta still violates the constraints!
 LOG Dec. 4, 2018:   Put the actual constant into function bounds Flow and Fupp, do not include constant inside 
                        function evaluation F(x).
                        
 LOG Dec. 15, 2018:  The linear derivative A can not be all zero elements. Will cause error.
 
 LOG Dec. 16, 2018:  The 1D bounds of x should be defined by xlow and xupp, 
                        do not include them in F and linear derivative A.
                                               
 LOG Dec. 18, 2018:  The 2D active subspace - DeltaDOGS with SNOPT shows error message:
                        SNOPTA EXIT  10 -- the problem appears to be infeasible
                        SNOPTA INFO  14 -- linear infeasibilities minimized
                     Fixed by introducing new bounds on x variable based on Delaunay simplex.
                     
'''
##################################  adaptive K search SNOPT ###################################


def triangulation_search_bound_snopt(inter_par, xi, y0, ind_min, y_safe, L_safe, finest_mesh):
    # reddir is a vector
    inf = 1e+20
    n   = xi.shape[0]
    xE  = inter_par.xi
    # 0: Build up the Delaunay triangulation based on reduced subspace.
    if n == 1:
        sx = sorted(range(xi.shape[1]), key=lambda x: xi[:, x])
        tri = np.zeros((xi.shape[1] - 1, 2))
        tri[:, 0] = sx[:xi.shape[1] - 1]
        tri[:, 1] = sx[1:]
        tri = tri.astype(np.int32)
    else:
        options = 'Qt Qbb Qc' if n <= 3 else 'Qt Qbb Qc Qx'
        tri = Delaunay(xi.T, qhull_options=options).simplices
        keep = np.ones(len(tri), dtype=bool)
        for i, t in enumerate(tri):
            if abs(np.linalg.det(np.hstack((xi.T[t], np.ones([1, n + 1]).T)))) < 1E-15:
                keep[i] = False  # Point is coplanar, we don't want to keep it
        tri = tri[keep]
    # Sc contains the continuous search function value of the center of each Delaunay simplex

    # 1: Identify the minimizer of adaptive K continuous search function
    Sc = np.zeros([np.shape(tri)[0]])
    Scl = np.zeros([np.shape(tri)[0]])
    Sc_safe = np.zeros(tri.shape[0])

    Rmax, max_dis = constantK_snopt_safelearning.max_circumradius_delauany_simplex(xi, xE, tri)
    # Determine the parameters b and c for exterior uncertainty function.
    b, c, status  = constantK_snopt_safelearning.uncertainty_parameter_solver(Rmax, max_dis)

    for ii in range(np.shape(tri)[0]):
        R2, xc = Utils.circhyp(xi[:, tri[ii, :]], n)
        if R2 < inf:
            # initialize with body center of each simplex
            x     = np.dot(xi[:, tri[ii, :]], np.ones([n + 1, 1]) / (n + 1))
            exist = constantK_snopt_safelearning.unevaluated_vertices_identification(xi[:, tri[ii, :]], xE)[0]
            if exist == 0:  # The Delauany simplex considered is safe
                e = (R2 - np.linalg.norm(x - xc) ** 2)
            else:
                e = constantK_snopt_safelearning.discrete_min_uncertainty(x, xE, b, c)[0]

            Sc[ii]      = (interpolation.interpolate_val(x, inter_par) - y0) / e
            Sc_safe[ii] = (Sc[ii] if exist == 0 else inf)

            if np.sum(ind_min == tri[ii, :]):
                Scl[ii] = np.copy(Sc[ii])
            else:
                Scl[ii] = inf
        else:
            Scl[ii] = inf
            Sc[ii]  = inf

    if np.min(Sc) < 0:  # minimize p(x) subject to safe estimate
        Scp = np.zeros(tri.shape[0])
        for ii in range(tri.shape[0]):
            x = np.dot(xi[:, tri[ii, :]], np.ones([n + 1, 1]) / (n + 1))
            Scp[ii] = interpolation.interpolate_val(x, inter_par)
        ind = np.argmin(Scp)
        delta = 1.0e-5
        tol = 1.0e-4
        xmin, ymin, result = interpolation_min_safe_solver(xi[:, tri[ind, :]], inter_par, y0, y_safe, L_safe, delta, tol)

    else:
        # 2: Determine the minimizer of continuous search function at those 3 Delaunay simplices.
        # First index : Global one, the simplex that has minimum value of Sc at circumcenters.
        # Second index: Global one within the safe region.
        # Third index : Local one with the lowest interpolation value.
        index = np.array([np.argmin(Sc), np.argmin(Sc_safe), np.argmin(Scl)])
        xm    = np.zeros((n, 3))
        ym    = np.zeros(3)

        for i in range(3):
            temp_x, ym[i] = adaptivek_search_snopt_min(xi[:, tri[index[i], :]], inter_par, y0, y_safe, L_safe, b, c, finest_mesh)
            xm[:, i] = temp_x.T[0]
        ymin = np.min(ym)
        xmin = xm[:, np.argmin(ym)].reshape(-1, 1)
        if np.argmin(ym) == 0:
            result = 'sc global'
        elif np.argmin(ym) == 1:
            result = 'sc safe'
        else:
            result = 'sc local'
    val, idx, x_nn = Utils.mindis(xmin, xE)
    safe_estimate = y_safe[:, idx] - L_safe * val

    return xmin, ymin, result, safe_estimate
# =====================================  Continuous search function Minimization   ==================================


def adaptivek_search_snopt_min(simplex, inter_par, y0, y_safe, L_safe, b, c, finest_mesh):
    '''
    Find the minimizer of the search fucntion in a simplex using SNOPT package.
    The function F is composed as:  1st        - objective
                                    2nd to nth - simplex bounds
                                    n+1 th ..  - safe constraints
    :param simplex  :     Delauany simplex of interest, n by n+1 matrix.
    :param inter_par:     Interpolation info.
    :param y0       :     Target value of truth function
    :param y_safe   :     Safe function evaluation.
    :param L_safe   :     Lipschitz constant of safety functions.
    :param b        :     The parameters for exterior uncertainty function. It is determined once Delaunay-tri is fixed.
    :param c        :     The parameters for exterior uncertainty function. It is determined once Delaunay-tri is fixed.
    :return:              The minimizer of adaptive K continuous search function within the given Delauany simplex.
    '''
    inf = 1.0e+20
    xE  = inter_par.xi
    n   = xE.shape[0]
    M   = y_safe.shape[0]

    # -------  ADD THE FOLLOWING LINE WHEN DEBUGGING --------
    # simplex = xi[:, tri[ind, :]]
    # -------  ADD THE FOLLOWING LINE WHEN DEBUGGING --------

    # Determine if the boundary corner exists in simplex, if boundary corner detected:
    # e(x) = (|| x - x' || + c )^b - c^b,  x' in S^k
    # else, e(x) is the regular uncertainty function.
    exist, eval_indicators = constantK_snopt_safelearning.unevaluated_vertices_identification(simplex, xE)

    R2, xc = Utils.circhyp(simplex, n)
    x = np.dot(simplex, np.ones([n + 1, 1]) / (n + 1))

    # First find minimizer xr on reduced model, then find the 2D point corresponding to xr. Constrained optm.
    A_simplex, b_simplex = Utils.search_simplex_bounds(simplex)
    lb_simplex = np.min(simplex, axis=1)
    ub_simplex = np.max(simplex, axis=1)

    m = n + 1  # The number of constraints which is determined by the number of simplex boundaries.
    assert m == A_simplex.shape[0], 'The No. of simplex constraints is wrong'

    # nF: The number of problem functions in F(x),
    # including the objective function, linear and nonlinear constraints.

    # ObjRow indicates the numer of objective row in F(x).
    ObjRow  = 1

    if n > 1:
        # The first function in F(x) is the objective function, the rest are m simplex constraints.
        # The last part of functions in F(x) is the safe constraints.
        # In high dimension, A_simplex make sure that linear_derivative_A won't be all zero.

        nF = 1 + m + M  # the last 1 is the safe constraint.
        # Since adaptiveK using ( p(x) - f0 ) / e(x), the objective function is nonlinear.
        # The constraints are generated by simplex bounds, all linear.
        Flow = np.hstack((-inf, b_simplex.T[0], np.zeros(M) ))
        Fupp = inf * np.ones(nF)

        # The lower and upper bounds of variables x.
        xlow    = np.copy(lb_simplex) + finest_mesh/2
        xupp    = np.copy(ub_simplex) - finest_mesh/2

        # For the nonlinear components, enter any nonzero value in G to indicate the location
        # of the nonlinear derivatives (in this case, 2).

        # A must be properly defined with the correct derivative values.
        linear_derivative_A    = np.vstack((np.zeros((1, n)), A_simplex, np.zeros((M, n)) ))
        nonlinear_derivative_G = np.vstack((2 * np.ones((1, n)), np.zeros((m, n)), 2 * np.ones((M, n)) ))

    else:
        # For 1D problem, only have 1 objective function, the simplex constraint is defined by x bounds.
        # nF = 1 + M, 1 obj + M safe cons.
        nF = 1 + M + 1

        Flow    = np.hstack((-inf, np.zeros(M), -inf ))
        Fupp    = inf * np.ones(nF)
        xlow    = np.copy(lb_simplex) + finest_mesh/2
        xupp    = np.copy(ub_simplex) - finest_mesh/2

        linear_derivative_A    = np.vstack(( np.zeros((1, n)), np.zeros((M, n)), np.ones((1, n)) ))
        nonlinear_derivative_G = np.vstack(( 2 * np.ones((1 + M, n)), np.zeros((1, n)) ))

    x0      = x.T[0]
    save_opt_for_snopt_ak(n, nF, inter_par, xc, R2, y0, A_simplex, y_safe, L_safe, exist, b, c)

    options = SNOPT_options()
    options.setOption('Infinite bound', inf)
    options.setOption('Verify level', 3)
    options.setOption('Verbose', False)
    options.setOption('Print level', -1)
    options.setOption('Scale option', 2)
    options.setOption('Print frequency', -1)
    options.setOption('Scale option', 2)
    options.setOption('Feasibility tolerance', 1e-5)
    options.setOption('Summary', 'No')

    sol = snopta(dogsobj, n, nF, x0=x0, name='DeltaDOGS_snopt', xlow=xlow, xupp=xupp, Flow=Flow, Fupp=Fupp,
                    ObjRow=ObjRow, A=linear_derivative_A, G=nonlinear_derivative_G, options=options)

    x = sol.x
    y = sol.objective

    return x.reshape(-1, 1), y


def save_opt_folder_path():
    current_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    folder = current_path[:-5]      # -5 comes from the length of '/dogs'
    return folder


def save_opt_for_snopt_ak(n, nF, inter_par, xc, R2, y0, A_simplex, y_safe, L_safe, exist, b, c):
    var_opt = {}
    folder = save_opt_folder_path()
    if inter_par.method == "NPS":
        var_opt['inter_par_method'] = inter_par.method
        var_opt['inter_par_w']      = inter_par.w
        var_opt['inter_par_v']      = inter_par.v
        var_opt['inter_par_xi']     = inter_par.xi
    var_opt['n']  = n
    var_opt['nF'] = nF
    var_opt['xc'] = xc
    var_opt['R2'] = R2
    var_opt['y0'] = y0
    var_opt['A']  = A_simplex
    var_opt['y_safe'] = y_safe
    var_opt['L_safe'] = L_safe
    var_opt['exist']  = exist
    var_opt['b']   = b
    var_opt['c'] = c
    io.savemat(folder + "/opt_info_ak.mat", var_opt)
    return


def adaptivek_search_cost_snopt(x):
    x       = x.reshape(-1, 1)
    folder  = save_opt_folder_path()
    var_opt = io.loadmat(folder + "/opt_info_ak.mat")

    n  = var_opt['n'][0, 0]
    xc = var_opt['xc']
    R2 = var_opt['R2'][0, 0]
    y0 = var_opt['y0'][0, 0]
    nF = var_opt['nF'][0, 0]
    A  = var_opt['A']
    b  = var_opt['b'][0, 0]
    c  = var_opt['c'][0, 0]
    y_safe = var_opt['y_safe']
    L_safe = var_opt['L_safe'][0, 0]
    exist  = var_opt['exist'][0, 0]

    # Initialize the output F.
    F = np.zeros(nF)
    M = y_safe.shape[0]

    method       = var_opt['inter_par_method'][0]
    inter_par    = interpolation.Inter_par(method=method)
    inter_par.w  = var_opt['inter_par_w']
    inter_par.v  = var_opt['inter_par_v']
    inter_par.xi = var_opt['inter_par_xi']

    p  = interpolation.interpolate_val(x, inter_par)
    gp = interpolation.interpolate_grad(x, inter_par)

    if exist == 0:
        e = R2 - np.linalg.norm(x - xc) ** 2
        ge = - 2 * (x - xc)
    else:
        e, ge, gge = constantK_snopt_safelearning.discrete_min_uncertainty(x, inter_par.xi, b, c)

    # denominator = (1e-10 if abs(p-y0) < 1e-10 else p - y0)
    # F[0] = - e / denominator
    # DM   = - ge / denominator + e * gp / denominator ** 2
    F[0] = (p - y0) / e
    DM   = gp / e - ge * (p - y0) / e ** 2
    # G1: The gradient of the objective function, the continuous search function.
    G1   = DM.flatten()

    val, idx, x_nn = Utils.mindis(x, inter_par.xi)
    norm2_difference = np.sqrt(np.dot((x - x_nn).T, x - x_nn))
    norm2_difference = ( 1e-15 if norm2_difference < 1e-15 else norm2_difference )

    # G2: Safety function constraint gradient, flattened version, trick is size = M.
    G2 = np.tile((- L_safe * (x - x_nn) / norm2_difference).T[0], M)

    if n > 1:
        # nD data has n+1 simplex bounds.
        F[1 : 1 + (n + 1)] = (np.dot(A, x)).T[0]
        F[1 + (n + 1) : 1 + (n + 1) + M] = y_safe[:, idx] - L_safe * norm2_difference * np.ones(M)

    else:
        F[1 : 1 + M] = y_safe[:, idx] - L_safe * norm2_difference * np.ones(M)
        F[-1] = np.sum(x)

    G = np.hstack((G1, G2))

    return F, G


def dogsobj(status, x, needF, F, needG, G):
    # G is the nonlinear part of the Jacobian
    F, G = adaptivek_search_cost_snopt(x)
    return status, F, G
# =====================================   Interpolant Minimization   ==================================


def interpolation_min_safe_solver(simplex, inter_par, y0, y_safe, L_safe, delta, tol):

    xE = inter_par.xi
    n = xE.shape[0]
    x = np.dot(simplex, np.ones([n + 1, 1]) / (n + 1))
    # First: Determine the estimated safety function values at x ( p(x) < y0 ).
    val, idx, x_nn = Utils.mindis(x, xE)
    safe_estimate_x = y_safe[:, idx] - L_safe * val

    if (safe_estimate_x > 0 ).all():
        # Safety is guaranteed, root finding p(x) along x and x_nn
        f = lambda x: interpolation.interpolate_val(x, inter_par) - y0
        result = 'minimize p(x)'
    else:
        # Safety is not guaranteed, root finding safety function along x and x_nn
        f = lambda x:  y_safe[:, idx] - L_safe * np.linalg.norm(x - x_nn)
        result = 'minimize hat_psi(x)'

    xmin, status = bisection_root_finding(f, x, x_nn, delta, tol)
    ymin = f(xmin)
    return xmin, ymin, result


def bisection_root_finding(f, a, b, delta, tol):
    '''
    Finding the root for the objective function f, could be slightly positive solution due to safety functions.
    Thought about using false position method at the first time, but it seems that falsi method only works for
    1D function.
    :param f    :   Objective function, could be multi-dimensional.
    :param a    :   Center of the circumcircle.
    :param b    :   The closest evaluated data point to a.
    :param delta:   Tolerance of distance of your interval on variable x.
    :param tol  :   Tolerance of objective function values.
    :return     :   The root of f(must be >= 0).
    '''
    num_iter = 1000
    # The hard thing is, you dont know the evaluated data is upper bound or the lower bound.
    if np.linalg.norm(a) < np.linalg.norm(b):
        low_bnd_b = 0
        low_bnd = a
        upp_bnd = b
    else:
        low_bnd_b = 1
        low_bnd = b
        upp_bnd = a
    flow = f(low_bnd)
    fupp = f(upp_bnd)
    eps = delta

    e = np.copy(upp_bnd - low_bnd)

    if ((np.sign(flow) > 0).all() and (np.sign(fupp) > 0).all()) or ((np.sign(flow) < 0).all() and (np.sign(fupp) < 0).all()):
        x = np.copy(b)
        status = 'Error message: Boundaries have the same sign, should not happen in this case.'
    else:
        status = 'Maximum iteration reached.'
        for i in range(num_iter):
            e /= 2
            m = low_bnd + e
            fm = f(m)
            if np.max(e) < delta:
                status = 'Required error of subinterval bound achieved'
                break
            elif np.max(np.abs(fm)) < tol:
                status = 'The tolerance on the objective function value is reached.'
                break
            else:
                if (np.sign(fm) > 0 + tol).all():
                    # mid point is all positive, set the bound close to the evaluated data to be the mid point
                    if low_bnd_b == 1:
                        low_bnd = np.copy(m)
                        flow = np.copy(fm)
                    else:
                        upp_bnd = np.copy(m)
                        fupp = np.copy(fm)
                else:
                    # mid point has negative part, set bound close to the circumcenter to be the mid point
                    if low_bnd_b == 1:
                        upp_bnd = np.copy(m)
                        fupp = np.copy(fm)
                    else:
                        low_bnd = np.copy(m)
                        fupp = np.copy(fm)
        if (np.sign(fm) > 0 + tol).all():  # mid point satisfy the safety functions
            x = np.copy(m)
        else:  # mid point does not satisfy the safety functions, using the bound close to the evaluated data points.
            if low_bnd_b == 1:
                x = np.copy(low_bnd)
            else:
                x = np.copy(upp_bnd)
    return x, status
