"""
This script computes the safe set and expansion set for SDOGS
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

import  numpy           as np
from    dogs            import Utils

__all__ = ['expansion_identifier', 'safe_expansion_set', 'grid_world', 'grid_points']

# ==================  Update Safe region Psi and Expansion set G  ==================


def expansion_identifier(x, safe_inter_query, sdogs):
    """
    Outline

    When you call this function inside safe_expansion_set, the query point x is guaranteed to be safe;
    This function determines whether or not x belongs to the expansion set;
    If x belongs to the expansion set, this function also computes its P_k^l(x) function values; O.w., pass

    ----------
    Parameters

    :param x                :       n-by-1 array (?); Query point
    :param safe_inter_query :       n-by-, array (?); The estimated safe radius of x
    :param sdogs            :       SafeDogs class object;

    ----------
    Output

    :return expand_sign :       boolean, 1 if x in expansion set, 0 o.w.
    :return expanders   :       int, non-negative; P_k^l(x) if x in expansion set, o.w. 0
    """
    # Initialization of outputs
    expand_sign         = False
    expander_children   = 0
    # Compute the safe rectangular of x with the estimated safe radius
    safe_init_refine, single_point, potentially_safe_grid_points = grid_points(x, safe_inter_query, sdogs)

    if safe_init_refine:
        # First criteria: If the estimated safe radius of the query point is too small,
        # smaller than the current mesh grid, then there is high probability that no unsafe points could
        # be classified as safe even though the query point is evaluated.
        # Also this indicates that query point is too close to the actually safe region boundary
        # (compared with current mesh points), then pass.

        # No current mesh points exist inside the estimated safe rectangular of x.
        # On the current mesh, this point does not provide safe info.
        # Do not expand this point at the current mesh.
        expand_sign         = False
        expander_children   = 0
    else:
        # TODO vectorize this piece of code!
        for i in range(potentially_safe_grid_points.shape[1]):
            query = potentially_safe_grid_points[:, i].reshape(-1, 1)

            # I - safe_estimate: Determine whether or not query is inside the current known to be safe region.
            # TODO fix the new safe estimate
            safe_estimate = np.min(sdogs.yS, axis=0) - sdogs.L_safe * np.linalg.norm(sdogs.xE - query, axis=0)
            query_safety  = False if (safe_estimate < 0).all() else True

            # II - Determine if the query is in the safe ball of x.
            # i.e., determine if the query could be safe based on safety function values at x
            query_in_safeball_x = True if np.min(safe_inter_query) - sdogs.L_safe * np.linalg.norm(query - x) > 0 else False

            # III - Determine if the query has the potential to expand other unsafe points based on current mesh
            query_safe_estimate = True if np.all(sdogs.safe_inter_par.SafeInter_val(query) / sdogs.L_safe > sdogs.mesh_size) else False

            # Judging criteria
            if not query_safety and query_in_safeball_x and query_safe_estimate:
                # query is unsafe, in the safe ball of x, and potentially to expand unsafe points
                expand_sign         = True
                expander_children  += 1
            else:
                pass

    return expand_sign, expander_children


def grid_world(limits, num_points):
    """
    Outline

    Generate all the discrete points of the discretization
    ----------
    Parameters

    :param limits       :   n-by-2 2d np.ndarray, first column stores lower bounds, second column stores upper bounds
    :param num_points   :   int, number of points for each dimension

    ----------
    Outputs

    :return points      :   n-by-(*) 2d np.ndarray,
    """
    num_points = np.broadcast_to(num_points[0], limits.shape[0]).astype(int)
    discrete_points = [np.linspace(low, up, n) for (low, up), n in zip(limits, num_points)]
    mesh = np.meshgrid(*discrete_points, indexing='ij')
    points = np.row_stack([col.ravel() for col in mesh])
    return points


def grid_points(x, y, sdogs):
    """
    Outline

    Compute the grid points that are in the hyper-rectangular centered at x with (estimated) safe radius.
    Not all the returned grid points are safe points

    ----------
    Parameters

    :param x    :       n-by-N, 2d array; n-dimension of input, N-number of data point
                                The array that stores the query point, could be evaluated data points,
                                or safe but have not yet been evaluated

    :param y    :       m-by-N, 2d array; n-number of safety functions, N-number of datapoint(CDC version!)
                                The array that stores the (estimated) safety function values at query points
    :param sdogs:       SafeDogs class object;

    ----------
    Output
    :return safe_init_refine:       Indicate if safe mesh refinement is needed at the beginning of optimization
    :return single_point        :       N-by-, np.ndarray; If the point x itself is the only safe point in its safe ball
                                        i.e. safe radius of x is less than mesh size
    :return _grid_points   :       n-by-(*), np.ndarray; Return the grid points that are in the ball
                                            centered at x with (estimated) safe radius
    """
    assert x.shape[0] == sdogs.n, "Query points x in function 'safe_grid_points' should have n as the first dimension. "
    assert x.shape[1] == y.shape[1], "There should be the same number of points x, with their corresponding safety " \
                                     "evaluations, i.e. x.shape[1] == y.shape[1]"
    N = x.shape[1]
    # Initialization of outputs
    _grid_points = np.empty(shape=[sdogs.n, 0])
    single_point = np.full(N, np.nan)

    # Main for loop
    for i in range(N):
        # reshape query to be n-by-1
        query = x[:, i].reshape(-1, 1)

        # compute the (estimated) safe radius at query, CDC version
        # TODO safe_radius might be change in future
        safe_radius = np.min(y[:, i], axis=0) / sdogs.L_safe
        if safe_radius < sdogs.mesh_size:
            # there is only one point x in its safe ball, safe radius < mesh size
            single_point[i] = True
            _grid_points = np.hstack((_grid_points, query))
            pass

        else:
            single_point[i] = False
            # find the number of intervals could go with each dimension on the current mesh
            # TODO num_mesh_interval might be change due to the change of safe_radius, now num_mesh_interval is 2d array
            num_mesh_interval = np.floor(safe_radius/sdogs.mesh_size).astype(int)

            # compute the lower bounds and upper bounds for each dimension
            lower_limits = np.maximum(np.zeros((sdogs.n, 1)), query - num_mesh_interval * sdogs.mesh_size)
            upper_limits = np.minimum(np.ones((sdogs.n, 1)), query + num_mesh_interval * sdogs.mesh_size)

            # concatenate lower and upper limits, compute the number of intervals for each dimension
            limits = np.hstack((lower_limits, upper_limits))
            # num_points is the number of intervals for each dim, at most to be sdogs.ms
            num_points = np.minimum(sdogs.ms, 2 * num_mesh_interval + 1)
            # reshape num_points to be a vector, each element represents how many intervals for each dimension
            num_points = np.broadcast_to(num_points, sdogs.n)

            # compute the grid points, not guaranteed to be safe, the corners of rectangular might be unsafe
            sub_grid_points = grid_world(limits, num_points)
            _grid_points = np.hstack((_grid_points, sub_grid_points))

    # uniqueness check on _grid_points, remove the repeated grid points
    _grid_points = Utils.unique_data(_grid_points)
    # transform single_point elements to boolean type, o.w. its float64
    single_point = single_point.astype(bool)
    safe_init_refine = True if np.all(single_point) else False
    return safe_init_refine, single_point, _grid_points


def safe_expansion_set(sdogs):
    """
    Outline

    Determine the safe set Psi and expansion set G. Determine whether or not the safe exploration stage has been
    exhausted; Also at the beginning of optimization, refine the mesh if needed. With CDC version, also computes the
    point with the largest value in function P (minimum value in surrogate if it is not unique).

    ----------
    Parameters

    :param sdogs            :   SafeDogs class object;

    ----------
    Output

    :return safe_expand_sign:   bool; True if there exists point to be evaluated so that safe set expanded
    :return safe_set        :   np.ndarray; The grid points that are known to be safe
    :return expansion_set   :   np.ndarray; The grid points that could potentially expand the safe set
    :return xhat            :   n-by-1, 2D np.ndarray; The point in the expansion set with the largest value in P
    :return ehat            :   scalar; The value of uncertainty function at xhat
    """
    # I: Determine the smallest rectangular that includes the safe region.
    # single_point: indicates whether or not each of xE is the only safe point in its safe ball on the current mesh
    # potentially_safe_grid_points: contains all the grid points in the safe rectangular of each of xE
    sdogs.safe_init_refine, sdogs.single_point, potentially_safe_grid_points = grid_points(sdogs.xE, sdogs.yS, sdogs)

    # safe_expand_sign: Check if there are points to be expanded as safe at this iteration.
    # safe_expand_sign == True: Safe exploration iteration;
    # safe_expand_sign == False: Exploitation iteration.
    sdogs.safe_expand_sign = None

    sdogs.xhat = np.empty(shape=[sdogs.n, 0])   # stores the maximizer of P
    sdogs.ehat = np.empty(shape=[1, 0]) # ehat: the uncertainty at the maximizer of P

    # II: Determine if the points belong to the expansion set.
    # - Calculate the uncertainty function value
    sdogs.safe_set      = np.empty(shape=[sdogs.n, 0])

    # III: The data structures 'expansion_set', 'expansion_set_uncertainty' and 'expansion_set_children_count'
    # are one-to-one correspondence.
    sdogs.expansion_set                = np.empty(shape=[sdogs.n, 0])
    sdogs.expansion_set_uncertainty    = np.empty(shape=[0, ])     # The uncertainty of each point in expansion set
    sdogs.expansion_set_children_count = np.empty(shape=[0, ])     # P_k^ell value of points in expansion set

    if sdogs.safe_init_refine:
        # On the current mesh grid, only the initial points are safe, refine the mesh grid
        # TODO fix xhat when initial refinement?
        sdogs.xhat             = None
        sdogs.safe_set         = np.copy(sdogs.xE)
        sdogs.safe_expand_sign = False
        sdogs.expansion_set = np.copy(sdogs.xE)

    else:
        # The Delaunay simplices for computing the uncertainty function values:
        # are computed at the beginning of each optimization iteration

        # The params b & c for uncertainty function outside of the Convex hull of the evaluated data points
        # is determined in the constant_surrogate_solver before the start of safe_expansion_set.
        # Because they are fixed once the Delaunay triangulation is determined.

        N = potentially_safe_grid_points.shape[1]
        for i in range(N):
            # 1: Determine it is safe or not, if it is unsafe, pass
            query         = potentially_safe_grid_points[:, i].reshape(-1, 1)
            # TODO: need a new safe criteria estimate
            safe_criteria = np.min(sdogs.yS, axis=0) - sdogs.L_safe * np.linalg.norm(query - sdogs.xE, axis=0)
            # Here the safe criteria is >= 0

            if (safe_criteria >= 0).any():
                # The query point is safe, add it to the safe_set.
                sdogs.safe_set = np.hstack((sdogs.safe_set, query))
                # 2: Query point have already been evaluated or not. If it has already been evaluated, pass.
                if np.min(np.linalg.norm(query - sdogs.xE, axis=0)) <= 1e-10:
                    pass
                else:
                    # 3: Determine it is inside expansion set or not.
                    # TODO Two kinds of safe_estimate (?):
                    # 1st - interpolation -> soft safe estimate for safe points in expansion set, estimate their safe R.
                    safe_inter_query = sdogs.safe_inter_par.SafeInter_val(query)

                    # If there exists unsafe point but could potentially be
                    # classified as safe once query is evaluated, then query point belongs to the expansion set.

                    expand_sign, expander_children = expansion_identifier(query, safe_inter_query, sdogs)
                    if expand_sign:
                        # The query point is inside the expansion set, add it to the expansion_set
                        sdogs.expansion_set = np.hstack((sdogs.expansion_set, query))

                        # 4: Determine the uncertainty function values at query
                        # at the query point that inside the expansion set
                        e = sdogs.uncertainty_eval(query)
                        # TODO I am not sure if the following concatenation about matrix size is correct
                        sdogs.expansion_set_uncertainty    = np.hstack((sdogs.expansion_set_uncertainty, e))
                        sdogs.expansion_set_children_count = np.hstack((sdogs.expansion_set_children_count, expander_children))

            else:
                # The query point is unsafe, pass.
                pass

        # 5: Determine the maximizer of uncertainty function inside the expansion set
        # If the expansion set is empty -> well searched the entire safe domain based on current mesh size,
        # If the max e(x) is less than epsilon -> well searched the entire safe domain
        # -> jump to min(surrogate)

        # 6: Dealing with the uniqueness issue of P_k^ell, pick the one with the minimum surrogate model values
        if sdogs.expansion_set.shape[1] == 0:
            sdogs.safe_expand_sign = False
        else:

            # Determine the point in expansion set that expanded the most number of unsafe points
            expanders_max = np.max(sdogs.expansion_set_children_count)

            # if the maximum uncertainty in the expansion set is less  than epsilon, safe expansion stage should end
            if np.max(sdogs.expansion_set_uncertainty) < sdogs.epsilon:
                # The max uncertainty function value in expansion set is too small, safe exploration is finished
                sdogs.safe_expand_sign = False
            else:
                sdogs.safe_expand_sign = True
                if len(sdogs.expansion_set_children_count - expanders_max == 0) == 1:
                    # The maximizer of expanders in expansion set is unique.
                    sdogs.xhat = sdogs.expansion_set[:, np.argmax(sdogs.expansion_set_children_count)].reshape(-1, 1)
                else:
                    # 6: Multiple maximizers, pick the one with the minimum surrogate function value.
                    max_expanders = sdogs.expansion_set[:, sdogs.expansion_set_children_count - expanders_max == 0]

                    # surrogate model value at expanders
                    expanders_surrogate = sdogs.surrogate_eval(max_expanders)
                    sdogs.xhat = max_expanders[:, np.argmin(expanders_surrogate)].reshape(-1, 1)
