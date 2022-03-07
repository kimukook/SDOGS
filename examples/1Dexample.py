import os
import inspect
import numpy as np
from dogs import interpolation
from dogs import OptionalParams
from dogs import safedogs
from dogs import SafeLearn
from dogs import plotting
from dogs import constantK_snopt_safelearning
from dogs import Utils

from TestFuncs import schwefel
from TestFuncs import sine
import matplotlib.pyplot as plt

# only print 4 digits
float_formatter = lambda x: "%.4f" % x
np.set_printoptions(formatter={'float_kind': float_formatter})

n = 2                   # Dimension of the input data
ub = np.ones((n, 1))    # upper bounds of decision variables
lb = np.zeros((n, 1))   # lower bounds of decision variables

TestFuncParams = {
    'ub': ub,
    'lb': lb,
    'm': n
}

fun = schwefel.Schwefel(TestFuncParams)
safe_fun = sine.Sine(TestFuncParams)

iter_max = 100          # Maximum number of iterations based on each mesh
MeshSize = 8            # Represents the number of mesh refinement that algorithm will perform
K = 3                   # parameter for constant k delta dogs

options = OptionalParams.SdogsOptions()
options.set_option('Constant surrogate', True)
options.set_option('Snopt solver', True)
options.set_option('Number of mesh refinement', 4)
options.set_option('Initial sites', .5 * np.ones((n, 1)))
options.set_option('Constant K', 3.0)
options.set_option('Plot display', True)


Ain = np.concatenate((np.identity(n), -np.identity(n)), axis=0)
Bin = np.concatenate((np.ones((n, 1)), np.zeros((n, 1))), axis=0)

sdogs = safedogs.SafeDogs(fun, safe_fun, options, Ain, Bin)



x = np.array([[.375, .625], [.25, .5]]).T
y = np.array([[4.5, 4.5], [4.5, 4.5]]).T
# x = np.array([[.375, .625]]).T
# y = np.array([[4.5, 4.5]]).T

num_iter = 0

# for kk in range(sdogs.mesh_refine):
#     for k in range(sdogs.iter_max):
num_iter += 1

# interpolation p(x) of the utility function
sdogs.inter_par = interpolation.InterParams(sdogs.xE)
sdogs.yp        = sdogs.inter_par.interpolateparameterization(sdogs.yE)

# interpolation q(x) of the safety function
sdogs.safe_inter_par = interpolation.SafeInterParams(sdogs.xE, sdogs.yS)
sdogs.yp_safe        = sdogs.safe_inter_par.update()

# local minimizer
sdogs.yp_min_ind = np.argmin(sdogs.yp)
sdogs.x_yp_min = sdogs.xE[:, sdogs.yp_min_ind].reshape(-1, 1)

# update the safe region and expansion set
# note that expansion set depends both on mesh size and iterations
# TODO use the new definition of safe radius
sdogs.safe_radius = np.min(sdogs.yS, axis=0) / sdogs.L_safe

sdogs.update_delaunay_triangulation()
sdogs.update_exterior_uncertainty()
SafeLearn.safe_expansion_set(sdogs)

# At the beginning, focus on safe exploration. As the algorithm proceeds, turn into exploitation mode at
# each level of grid size.
if sdogs.safe_expand_sign:
    # safe exploration stage: current evaluate point is the max P (with min surrogate value if not unique) in the expansion set
    xc_eval = np.copy(sdogs.xhat)
elif not sdogs.single_point:
    # expansion set empty, and not single safe point
    # exploitation stage: current evaluate point is the min surrogate model
    xc, yc, result, safe_estimate = constantK_snopt_safelearning.triangulation_search_bound_snopt(sdogs)
    # TODO fix safe_mesh_quantizer
    xc_eval = Utils.safe_mesh_quantizer(sdogs)

else:
    # TODO When could this condition happen?
    xc_eval = np.zeros((sdogs.n, 1))
    pass

if Utils.mindis(xc_eval, sdogs.xE)[0] < 1e-6 or sdogs.single_point:
    # TODO fix mesh refine step in safedogs: ms*=2, mesh_size update, K update
    Nm *= 2
    delta = 1 / Nm
    if not safe_expand_sign:
        # If this mesh refinement is incurred by safe exploration, do not increase K
        K *= 3
    print('===============  MESH Refinement  ===================')
    summary = {'alg': 'SafeLearn', 'xc_grid': xc_eval, 'xmin': xmin, 'Nm': Nm, 'y0': y0}
    plotting.print_summary(num_iter, k, kk, xE, yE, summary, 1)
else:
    xE = np.hstack((xE, xc_eval))
    yE = np.hstack((yE, func_eval(xc_eval)))
    y_safe = np.hstack((y_safe, safe_eval(xc_eval)))

    summary = {'alg': 'ori', 'xc_grid': xc_eval, 'xmin': xmin, 'Nm': Nm, 'y0': y0}
    plotting.print_summary(num_iter, k, kk, xE, yE, summary)

# import imp
# imp.reload(plotting)
# plotting.safe_expansion_set_plot(sdogs)

