import  os
import  inspect
import  numpy         as np
import  scipy.io        as io
from    dogs          import Utils
from    dogs          import SafeLearn
from    dogs          import plotting
from    dogs          import interpolation
from    dogs          import exterior_uncertainty
from    scipy.spatial import Delaunay
from    optimize      import snopta, SNOPT_options

'''
 constantK_snopt.py file contains functions used for DeltaDOGS(Lambda) algorithm.
 Using the package optimize (SNOPT) provided by Prof. Philip Gill and Dr. Elizabeth Wong, UCSD.

 This is a script of DeltaDOGS(Lambda) dealing with linear constraints problem which is solved using SNOPT. 
 Notice that this scripy inplements the snopta function. (Beginner friendly)

 The adaptive-K continuous search function has the form:
 Sc(x) = P(x) - K * e(x):

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
'''
##################################  Constant K search SNOPT ###################################


# TODO to be deleted! surrogate_eval inside safedogs now
def surrogate_eval(sdogs, x):
    x = x.reshape(-1, 1)
    e = sdogs.surrogate_eval(sdogs, x)
    f = sdogs.inter_par.inter_val(x) - sdogs.K * e
    return f

# TODO constant_surrogate_solver to be deleted. too cumbersome
def constant_surrogate_solver(sdogs):
    sdogs.iter += 1
    # Update the interpolation for f(x) and psi(x)
    sdogs.inter_par = interpolation.InterParams(sdogs.xE)
    sdogs.yp = sdogs.inter_par.interpolateparameterization(sdogs.yE)

    sdogs.safe_inter = interpolation.SafeInterParams(sdogs.xE, sdogs.yS)
    sdogs.yp_safe = sdogs.safe_inter.update()

    # local minimizer of interpolation yp
    sdogs.ind_min = np.argmin(sdogs.yp)

    # Update the Delaunay triangulation
    sdogs.xi, ind_min = Utils.add_sup(sdogs)
    SafeLearn.delaunay_triangulation(sdogs)

    # Calculate the parameters b & c for fractional uncertainty function
    SafeLearn.fractional_uncertainty_parameter(sdogs)

    # Update the safe region and expansion set
    # note that expansion set depends both on mesh size and iterations
    sdogs.safe_radius = np.min(sdogs.yS, axis=0) / sdogs.L_safe
    SafeLearn.safe_expansion_set(sdogs)
    # Calculate the Rmax and max_dis, b & c for new uncertainty function

    # At the beginning, focus on exploration. As the algorithm proceeds, turn into exploitation mode.
    # If the exploitation fails, expand the safe region.
    if sdogs.SinglePoint:
        sdogs.iter_type = 'refine'
        # xhat should be the very first safe initial at this case
        sdogs.xc = np.copy(sdogs.xhat)
        pass
    else:
        if sdogs.safe_expand_sign:  # If there is point inside expansion set, then trigger safe exploration.
            # - Previously, the point to evaluate is the max uncertainty in expansion set
            #   Now, the point to evaluate is the point that could possibly expand the most number
            #   of unsafe points in the expansion set
            # - Two circumstances that safe_expand_sign fails:
            #   1: SinglePoint occurred, then refine the mesh grid;
            #   2: expansion set has been exhausted.
            sdogs.xc = np.copy(sdogs.xhat)
            sdogs.iter_type = 'explore'
            # SafeDOGSplot.safe_contour_uncertainty_2Dplot(xE, safe_eval, y_safe, L_safe, Nm)

            # If xc_eval discovered by expansion set is close to evaluated data points,
            # This should not happen since uncertainty of xc should be bigger than epsilon. Thus, decrease epsilon
        else:
            xc, yc, result, safe_estimate = triangulation_search_bound_snopt(sdogs, ind_min)
            print(xc)
            sdogs.xc = Utils.safe_mesh_quantizer(sdogs, xc)
            print(sdogs.xc)
            sdogs.iter_type = 'exploit'
    sdogs.plot_func(sdogs)

    if Utils.mindis(sdogs.xc, sdogs.xE)[0] < 1e-6 or sdogs.SinglePoint:
        # 1. xc already evaluated, mesh refine.
        # 2. Only one single point exists in the safe set, mesh too coarse, refine the mesh.
        sdogs.iter_type = 'refine'
        sdogs.ms *= 2
        sdogs.delta = 1 / sdogs.ms
        sdogs.mesh_size = 1 / sdogs.ms
        # if not sdogs.safe_expand_sign:
        if not sdogs.SinglePoint:
            # If this mesh refinement is incurred by safe exploration, do not increase K
            sdogs.K *= 3
        print('after mesh refine')
        print(sdogs.xc)
        print('=====')
    else:
        sdogs.xE = np.hstack((sdogs.xE, sdogs.xc))
        sdogs.yE = np.hstack((sdogs.yE, sdogs.func_eval(sdogs.xc)))
        sdogs.yS = np.hstack((sdogs.yS, sdogs.safe_eval(sdogs.xc)))
    plotting.summary_display(sdogs)


def triangulation_search_bound_snopt(sdogs):
    """

    Minimize the constant K continuous search function within the safe region.
    :param sdogs:      safedogs class
    :param inter_par:  Interpolation info.
    :param xi:         Vertices of Delauany simplex.
    :param K:          Tradeoff parameter between exploitation and exploration
    :param ind_min:    Promising Delauany simplex index number.
    :param y_safe:     Safety function values of evaluated data point.
    :param L_safe:     Lipschitz constant of safety function.
    :return:           Minimizer of surrogate within safe region.
    """
    inf = 1e+20

    # 0: The Delaunay triangulation is constructed at the beginning of each iteration.

    # Sc contains the continuous search function value of the center of each Delaunay simplex
    # 1: Identify the value of constant K continuous search function at the circumcenter of each Delaunay simplex
    Sc                  = np.zeros(sdogs.tri.shape[0])
    Scl                 = np.zeros(sdogs.tri.shape[0])
    Sc_safe             = np.zeros(sdogs.tri.shape[0])

    for ii in range(np.shape(sdogs.tri)[0]):
        R2, xc          = Utils.circhyp(sdogs.xi[:, sdogs.tri[ii, :]], sdogs.n)
        if R2 < inf:
            # initialize with centroid of each simplex
            # x           = np.dot(sdogs.xi[:, sdogs.tri[ii, :]], np.ones([sdogs.n + 1, 1]) / (sdogs.n + 1))
            x           = np.mean(sdogs.xi[:, sdogs.tri[ii, :]], axis=1).reshape(-1, 1)
            Sc[ii]      = sdogs.surrogate_eval(x)

            # compute the safe estimate of centroids: if positive, Sc_safe[ii] to be Sc[ii]; else unsafe, Sc_safe to be inf
            sc_safe_est = np.max(np.min(sdogs.yS, axis=0) - sdogs.L_safe * np.linalg.norm(x - sdogs.xE))
            Sc_safe[ii] = (Sc[ii] if sc_safe_est >= 0 else inf)

            # If this simplex contains the DT vertex minimizer of yp
            if np.min(np.linalg.norm(sdogs.x_yp_min - sdogs.tri[ii, :])) <= 1e-10:
                Scl[ii] = np.copy(Sc[ii])
            else:
                Scl[ii] = inf
        else:
            Scl[ii]     = inf
            Sc[ii]      = inf

    # 2: Determine the minimizer of continuous search function at those 3 Delaunay simplices.
    # optm_result == 1: Global one, the simplex that has minimum value of Sc at circumcenters, might be unsafe
    # optm_result == 2: Global one within the safe region.
    # optm_result == 3: Local and might be unsafe
    index               = np.array([np.argmin(Sc), np.argmin(Sc_safe), np.argmin(Scl)])
    xm                  = np.zeros((sdogs.n, 3))
    ym                  = np.zeros(3)
    for i in range(3):
        temp_x, ym[i]   = constant_search_snopt(sdogs.xi[:, sdogs.tri[index[i], :]], sdogs)
        xm[:, i]        = temp_x.T[0]

    sdogs.yc            = np.min(ym)
    sdogs.xc            = xm[:, np.argmin(ym)].reshape(-1, 1)
    sdogs.optm_result   = np.argmin(ym)
    sdogs.xc_safe_est   = np.max(np.min(sdogs.yS, axis=0) - sdogs.L_safe * np.linalg.norm(sdogs.xc - sdogs.xE))
# ============================   Continuous search function Minimization   =================================


def constant_search_snopt(simplex, sdogs):
    '''
    Find the minimizer of the search fucntion in a simplex using SNOPT package.
    The function F is composed as:  1st        - objective
                                    2nd to nth - simplex bounds
                                    n+1 th ..  - safe constraints
    :param simplex  :   Delaunay simplex of interest, n by n+1 matrix.
    :param inter_par:   Interpolation info.
    :param K        :   Tradeoff parameter.
    :param y_safe   :   Safe function evaluation.
    :param L_safe   :   Lipschitz constant of safety functions.
    :param b        :   The parameters for exterior uncertainty function. It is determined once Delaunay-tri is fixed.
    :param c        :   The parameters for exterior uncertainty function. It is determined once Delaunay-tri is fixed.
    :return:            The minimizer of constant K continuous search function within the given Delaunay simplex.
    '''
    inf = 1.0e+20

    # -------  ADD THE FOLLOWING LINE WHEN DEBUGGING --------
    # simplex = xi[:, tri[index[i], :]]
    # -------  ADD THE FOLLOWING LINE WHEN DEBUGGING --------

    # - Determine if the boundary corner exists in simplex:
    #   If boundary corner detected:
    #     e(x) = (|| x - x' || + c )^b - c^b,  x' in S^k
    #   else
    #     e(x) is the regular uncertainty function.
    # - eval_indicators: Indicate which vertices of simplex is evaluated
    exist, eval_indicators = SafeLearn.unevaluated_vertices_identification(simplex, sdogs.xE)
    # The query simplex only has one evaluated vertex if unique_eval_vertex = 1.
    unique_eval_vertex = (1 if len(np.where(eval_indicators != 0)[0]) == 1 else 0)

    # Find the minimizer of the search fucntion in a simplex using SNOPT package.
    R2, xc = Utils.circhyp(simplex, sdogs.n)
    # x is the center of this simplex
    x = np.dot(simplex, np.ones([sdogs.n + 1, 1]) / (sdogs.n + 1))

    # First find minimizer xr on reduced model, then find the 2D point corresponding to xr. Constrained optm.
    A_simplex, b_simplex = Utils.search_simplex_bounds(simplex)
    lb_simplex = np.min(simplex, axis=1)
    ub_simplex = np.max(simplex, axis=1)

    m = sdogs.n + 1  # The number of constraints which is determined by the number of simplex boundaries.
    assert m == A_simplex.shape[0], 'The No. of simplex constraints is wrong'

    # nF: The number of problem functions in F(x), including the objective function, linear and nonlinear constraints.
    # ObjRow indicates the number of objective row in F(x).
    ObjRow = 1

    # solve for constrained minimization of safe learning within each open ball of the vertices of simplex.
    # Then choose the one with the minimum continuous function value.
    x_solver = np.empty(shape=[sdogs.n, 0])
    y_solver = []

    for i in range(sdogs.n + 1):
        # In n-dimensional simplex, there are n+1 vertices, for each of these vertices, there is a ball such that
        # all the points within that ball have the closest evaluated points as the query vertex. This is good for
        # building the exterior uncertainty function which needs this closest evaluated data point.

        vertex = simplex[:, i].reshape(-1, 1)
        # First find the y_safe[vertex]:
        val, idx, x_nn = Utils.mindis(vertex, sdogs.xE)

        if val > 1e-10:
            # This vertex is a unsafe boundary corner point. No safe-guarantee, do not optimize around support points.
            continue
        else:
            # safe_bounds = sdogs.yS[:, idx]
            safe_bounds = sdogs.yS[:, idx] ** 2

            if sdogs.n > 1 and unique_eval_vertex == 0:
                # The First part of F(x) is the objective function.
                # The second part of F(x) is the simplex bounds.
                # The third part of functions in F(x) is the safe constraints.
                # The fourth part of functions in F(x) is the nearest evaluated data points bound.
                # In high dimension, A_simplex make sure that linear_derivative_A won't be all zero.

                eval_indices = np.where(eval_indicators != 0)[0]
                other_eval_vertices_index = np.where(eval_indices != i)[0]
                n_other_eval_v = len(other_eval_vertices_index)
                nF = 1 + m + sdogs.M + n_other_eval_v

                Flow = np.hstack((-inf, b_simplex.T[0], -safe_bounds, np.zeros(n_other_eval_v)))
                Fupp = inf * np.ones(nF)

                # The lower and upper bounds of variables x.
                xlow = np.copy(lb_simplex)
                xupp = np.copy(ub_simplex)

                # Since constant using p(x) - K * e(x), the objective function is nonlinear.
                # The constraints are generated by simplex bounds, all linear.
                # The safety function is L2 norm, nonlinear

                # For the nonlinear components, enter any nonzero value in G to indicate the location
                # of the nonlinear derivatives (in this case, 2).

                # A must be properly defined with the correct derivative values.
                linear_derivative_A    = np.vstack((np.zeros((1, sdogs.n)), A_simplex, np.zeros((sdogs.M + n_other_eval_v, sdogs.n))))
                nonlinear_derivative_G = np.vstack((2 * np.ones((1, sdogs.n)), np.zeros((m, sdogs.n)),
                                                    2 * np.ones((sdogs.M + n_other_eval_v, sdogs.n))))
            elif sdogs.n > 1 and unique_eval_vertex == 1:
                # For higher-dimensional problem, if there is only one evaluated vertex of such vertex,
                # then we don't need to bound the nearest point in such simplex.

                # The First part of F(x) is the objective function.
                # The second part of F(x) is the simplex bounds.
                # The third part of functions in F(x) is the safe constraints.
                # In high dimension, A_simplex make sure that linear_derivative_A won't be all zero.

                nF = 1 + m + sdogs.M

                Flow = np.hstack((-inf, b_simplex.T[0], -safe_bounds))
                Fupp = inf * np.ones(nF)
                xlow = np.copy(lb_simplex)
                xupp = np.copy(ub_simplex)

                linear_derivative_A    = np.vstack((np.zeros((1, sdogs.n)), A_simplex, np.zeros((sdogs.M, sdogs.n))))
                nonlinear_derivative_G = np.vstack((2 * np.ones((1, sdogs.n)), np.zeros((m, sdogs.n)), 2 * np.ones((sdogs.M, sdogs.n))))

            else:  # n = 1

                # For 1D problem, the simplex constraint is defined in x bounds.
                # For 1D problem, the closest point from x to evaluated data points must be this querying vertex.

                # 1 obj + M safe con. Plus another 1 redundant constraint to make matrix A contain nonzero terms.

                # Since constant using p(x) - K * e(x), the objective function is nonlinear.
                # The safety function is L2 norm, nonlinear
                # The auxiliary function is linear.

                nF = 1 + sdogs.M + 1
                Flow = np.hstack((-inf, -safe_bounds, -inf))
                Fupp = inf * np.ones(nF)
                xlow = np.copy(lb_simplex)
                xupp = np.copy(ub_simplex)

                linear_derivative_A    = np.vstack((np.zeros((1 + sdogs.M, sdogs.n)), np.ones((1, sdogs.n))))
                nonlinear_derivative_G = np.vstack((2 * np.ones((1 + sdogs.M, sdogs.n)), np.zeros((1, sdogs.n))))

            x0 = x.T[0]

            # -------  ADD THE FOLLOWING LINE WHEN DEBUGGING --------
            # cd dogs
            # -------  ADD THE FOLLOWING LINE WHEN DEBUGGING --------

            # save_opt_for_snopt_ck(n, nF, inter_par, xc, R2, K, A_simplex, L_safe, vertex, exist, unique_eval_vertex,
            #                       simplex, M, b, c)
            save_opt_for_snopt_ck(sdogs, nF, xc, R2, A_simplex, vertex, exist, unique_eval_vertex, simplex)

            options = SNOPT_options()
            options.setOption('Solution print', False)
            options.setOption('Infinite bound', inf)
            options.setOption('Verify level', 3)
            options.setOption('Verbose', False)
            options.setOption('Print level', -1)
            options.setOption('Scale option', 2)
            options.setOption('Print frequency', -1)
            options.setOption('Major feasibility', 1e-10)
            options.setOption('Minor feasibility', 1e-10)
            options.setOption('Feasibility tolerance', 1e-5)
            options.setOption('Summary', 'No')

            result = snopta(dogsobj, sdogs.n, nF, x0=x0, name='DeltaDOGS_snopt', xlow=xlow, xupp=xupp, Flow=Flow, Fupp=Fupp,
                            ObjRow=ObjRow, A=linear_derivative_A, G=nonlinear_derivative_G, options=options)

            x_solver = np.hstack((x_solver, result.x.reshape(-1, 1)))
            y_solver.append(result.objective)

    y_solver = np.array(y_solver)
    y = np.min(y_solver)
    x = x_solver[:, np.argmin(y_solver)].reshape(-1, 1)

    return x, y


def folder_path():
    current_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    folder = current_path[:-5]  # -5 comes from the length of '/dogs'
    return folder


def save_opt_for_snopt_ck(sdogs, nF, xc, R2, A_simplex, vertex, exist, unique_eval_vertex,
                          simplex):
    var_opt = {}
    folder = folder_path()
    if sdogs.inter_par.method == "NPS":
        var_opt['inter_par_method'] = sdogs.inter_par.method
        var_opt['inter_par_w']      = sdogs.inter_par.w
        var_opt['inter_par_v']      = sdogs.inter_par.v
        var_opt['inter_par_xi']     = sdogs.inter_par.xi
    var_opt['n']  = sdogs.n
    var_opt['M']  = sdogs.M
    var_opt['nF'] = nF
    var_opt['xc'] = xc
    var_opt['R2'] = R2
    var_opt['K']  = sdogs.K
    var_opt['A']  = A_simplex
    var_opt['b']  = sdogs.b
    var_opt['c']  = sdogs.c
    var_opt['exist']  = exist
    var_opt['L_safe'] = sdogs.L_safe
    var_opt['vertex'] = vertex
    var_opt['simplex'] = simplex

    var_opt['unique_eval_vertex'] = unique_eval_vertex
    io.savemat(folder + "/opt_info_ck.mat", var_opt)
    return


def constantk_search_cost_snopt(x):
    x = x.reshape(-1, 1)
    folder = folder_path()
    var_opt = io.loadmat(folder + "/opt_info_ck.mat")

    n  = var_opt['n'][0, 0]
    M  = var_opt['M'][0, 0]
    xc = var_opt['xc']
    R2 = var_opt['R2'][0, 0]
    K  = var_opt['K'][0, 0]
    nF = var_opt['nF'][0, 0]
    A  = var_opt['A']
    b  = var_opt['b'][0, 0]
    c  = var_opt['c'][0, 0]
    L_safe = var_opt['L_safe'][0, 0]
    vertex = var_opt['vertex']
    exist  = var_opt['exist'][0, 0]

    simplex = var_opt['simplex']
    unique_eval_vertex = var_opt['unique_eval_vertex'][0, 0]

    xE = var_opt['inter_par_xi']
    inter_par = interpolation.InterParams(xE)
    inter_par.w = np.copy(var_opt['inter_par_w'])
    inter_par.v = np.copy(var_opt['inter_par_v'])
    inter_par.xi = np.copy(xE)

    # Initialize the output F and G.
    F = np.zeros(nF)

    p = inter_par.inter_val(x)
    gp = inter_par.inter_grad(x)

    # The uncertainty function is defined using distance from x to xE
    # The estimated safety function is defined using distance from x to vertices of simplex.

    if exist == 0:  # All vertices of current simplex are evaluated.
        e = R2 - np.linalg.norm(x - xc) ** 2
        ge = - 2 * (x - xc)
    else:  # unevaluated boundary corner detected.
        e, ge, gge = SafeLearn.discrete_min_uncertainty(x, inter_par.xi, b, c)

    F[0] = p - K * e
    DM = gp - K * ge

    norm2_difference = np.dot((x - vertex).T, x - vertex)

    # G1: continuous search model function gradient
    G1 = DM.flatten()
    # G2: Safety function constraint gradient, flattened version, trick is size = M.
    # Notice that the safety functions are transformed due to the numerical difficulties.
    # Here we take L_safe^2 * ||x-vertex||^2_2 <= psi^2(vertex)
    G2 = np.tile((-2 * L_safe**2 * (x - vertex)).T[0], M)

    if n > 1 and unique_eval_vertex == 0:
        # nD data has n+1 simplex bounds.
        F[1 : 1 + (n + 1)] = (np.dot(A, x)).T[0]
        # M safety functions
        # Notice that SNOPT use c(x) >= 0, then transform it with negative sign
        # -L_safe^2 * || x - vertex ||^2_2 >= - psi ^ 2(vertex)
        F[1 + (n + 1) : 1 + (n + 1) + M] = - L_safe**2 * norm2_difference * np.ones(M)

        exist, eval_indicators = SafeLearn.unevaluated_vertices_identification(simplex, inter_par.xi)
        eval_indices = np.where(eval_indicators != 0)[0]
        i = np.where(np.linalg.norm(simplex - vertex, axis=0) == 0)[0]
        other_eval_vertices_index = np.where(eval_indices != i)[0]
        other_eval_vertices = simplex[:, other_eval_vertices_index]

        # Nearest neighbor bounds
        F[1 + (n + 1) + M:] = np.linalg.norm(x - other_eval_vertices, axis=0) ** 2 - np.linalg.norm(x - vertex) ** 2

        # G3: closest evaluated point constraint gradient
        G3 = 2 * (x - other_eval_vertices) - 2 * (x - vertex)
        G = np.hstack((G1, G2, G3.flatten()))

    elif n > 1 and unique_eval_vertex == 1:
        F[1 : 1 + (n + 1)] = (np.dot(A, x)).T[0]
        F[1 + (n + 1) : 1 + (n + 1) + M] = - L_safe**2 * norm2_difference * np.ones(M)

        G = np.hstack((G1, G2))

    else:
        # next line is the safe con
        F[1 : 1 + M] = - L_safe**2 * norm2_difference * np.ones(M)
        # Next line is the redundant row.
        F[-1] = np.sum(x)

        G = np.hstack((G1, G2))

    return F, G


def dogsobj(status, x, needF, F, needG, G):
    # G is the nonlinear part of the Jacobian
    F, G = constantk_search_cost_snopt(x)
    return status, F, G
