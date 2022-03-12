import os
import inspect
import imp
import numpy as np
import  scipy.io as io
from dogs import interpolation
from dogs import OptionalParams
from dogs import safedogs
from dogs import SafeLearn
from dogs import plotting
from dogs import constant_snopt_min
from dogs import Utils
from dogs import exterior_uncertainty
from    optimize      import snopta, SNOPT_options

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
options.set_option('Optimization solver', 'snopt')

Ain = np.concatenate((np.identity(n), -np.identity(n)), axis=0)
Bin = np.concatenate((np.ones((n, 1)), np.zeros((n, 1))), axis=0)

sdogs = safedogs.SafeDogs(fun, safe_fun, options, Ain, Bin)

for kk in range(sdogs.mesh_refine):
    for k in range(sdogs.iter_max):

        # interpolation p(x) of the utility function
        sdogs.inter_par = interpolation.InterParams(sdogs.xE)
        sdogs.yp        = sdogs.inter_par.interpolateparameterization(sdogs.yE)

        # interpolation q(x) of the safety function
        sdogs.safe_inter_par = interpolation.SafeInterParams(sdogs.xE, sdogs.yS)
        sdogs.yp_safe        = sdogs.safe_inter_par.update()

        # local minimizer
        sdogs.yp_min_ind = np.argmin(sdogs.yp)
        sdogs.x_yp_min   = sdogs.xE[:, sdogs.yp_min_ind].reshape(-1, 1)

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
            sdogs.iter_type = 2
            xc_eval = np.copy(sdogs.xhat)

        elif not sdogs.safe_init_refine:

            sdogs.iter_type = 4
            # expansion set empty, and not single safe point
            # exploitation stage: current evaluate point is the min surrogate model
            constant_snopt_min.triangulation_search_bound_snopt(sdogs)
            # TODO fix safe_mesh_quantizer
            xc_eval = sdogs.safe_mesh_quantizer(sdogs)

        else:
            # safe mesh refinement invoked at the initial stage of optimization
            xc_eval = None
            pass

        sdogs.iter += 1

        if Utils.mindis(xc_eval, sdogs.xE)[0] < 1e-6 or sdogs.safe_init_refine:
            sdogs.iter_type = 3
            # TODO fix mesh refine step in safedogs class? ms*=2, mesh_size update, K update
            sdogs.ms *= 2
            sdogs.mesh_size = 1 / sdogs.ms
            if not sdogs.single_point:
                # Mesh refinement is invoked by exploitation, increase K
                sdogs.iter_type = 5
                K *= 3

        else:
            # TODO fix function evaluation in safedogs class?
            sdogs.xE = np.hstack((sdogs.xE, xc_eval))
            sdogs.yE = np.hstack((sdogs.yE, sdogs.func_eval(xc_eval)))
            sdogs.yS = np.hstack((sdogs.yS, sdogs.safe_eval(xc_eval)))

        plotting.summary_display(sdogs)

# import imp
# imp.reload(plotting)
# plotting.safe_expansion_set_plot(sdogs)

