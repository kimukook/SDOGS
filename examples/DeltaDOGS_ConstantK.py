import os
import inspect
import scipy.io as io
from scipy.spatial import Delaunay
from optimize import snopta, SNOPT_options
import numpy as np
from functools import partial
from dogs import interpolation
from dogs import Utils
from dogs import constant_snopt_min
from dogs import plotting
from dogs import SafeLearn

# only print 4 digits
float_formatter = lambda x: "%.4f" % x
np.set_printoptions(formatter={'float_kind': float_formatter})

# This script shows the main code of Delta DOGS(Lambda) - using Adaptive K continuous search function

n = 1              # Dimension of the input data
fun_arg = 2        # Type of objective function
safe_fun_arg = 1   # Type of safe constraint
iter_max = 100     # Maximum number of iterations based on each mesh
MeshSize = 8       # Represents the number of mesh refinement that algorithm will perform
K = 3              # parameter for constant k delta dogs

# plot class
plot_index = 1
original_plot = 0
illustration_plot = 0
interpolation_plot = 0
subplot_plot = 0
store_plot = 1     # The indicator to store ploting results as png.
nff = 1  # Number of experiments

# Algorithm choice:
surrogate          = {}
surrogate['type']  = 'c'
surrogate['param'] = K
sc = "ConstantK"   # The type of continuous search function
alg_name = 'SDOGS/'

# Calculate the Initial trinagulation points
Nm = 8             # Initial mesh grid size
num_iter = 0       # Represents how many iteration the algorithm goes

# Truth function
fun, lb, ub, y0, xmin, fname = Utils.test_fun(fun_arg, n)
func_eval                    = partial(Utils.fun_eval, fun, lb, ub)
# safe constraints
# The initial point must be on the mesh. Essential for generating expansion set.
safe_fun, x0, L_safe, safe_name = Utils.test_safe_fun(safe_fun_arg, n)
safe_eval                       = partial(Utils.fun_eval, safe_fun, lb, ub)
M                               = safe_eval(x0).shape[0]  # Dimension of safety functions

# params for safe expansion
delta   = 1 / Nm
epsilon = 1e-4

xU = Utils.bounds(np.zeros([n, 1]), np.ones([n, 1]), n)

for ff in range(nff):
    xE      = np.copy(x0)
    num_ini = xE.shape[1]
    yE      = np.zeros(xE.shape[1])       # yE stores the objective function value
    y_safe  = np.zeros((M, xE.shape[1]))  # y_safe stores the value of safe constraints.

    # initialization of the data points
    for ii in range(xE.shape[1]):
        yE[ii]     = func_eval(xE[:, ii])
        y_safe[:, ii] = safe_eval(xE[:, ii]).T[0]

    if plot_index:
        plot_parameters = {'store': store_plot, 'safe_cons_type': safe_fun_arg, 'safe_cons': 1}
        # TODO fix this old and out-of-date plot_class thing.
        plot_class = plotting.plot(plot_parameters, sc, num_ini, ff, fun_arg, alg_name)

    for kk in range(MeshSize):
        # if num_iter == 60:
        #     break
        for k in range(iter_max):
            # if num_iter == 60:
            #     break
            num_iter += 1

            # interpolation p(x) of the utility function
            inter_par = interpolation.InterParams(xE)
            yp        = inter_par.interpolateparameterization(yE)

            # interpolation q(x) of the safety function
            safe_inter = interpolation.SafeInterParams(xE, y_safe)
            yp_safe    = safe_inter.update()

            # local minimizer
            ypmin = np.amin(yp)
            ind_min = np.argmin(yp)

            # update the safe region and expansion set
            # note that expansion set depends both on mesh size and iterations
            # TODO use the new definition of safe radius
            safe_radius = np.min(y_safe, axis=0) / L_safe
            #
            safe_expand_sign, SinglePoint, xhat, ehat, expansion_set, uncertainty, safe_set = SafeLearn.safe_expansion_set(xE, y_safe, Nm, L_safe, safe_inter, inter_par, delta, epsilon, surrogate)

            # At the beginning, focus on safe exploration. As the algorithm proceeds, turn into exploitation mode at
            # each level of grid size.
            if safe_expand_sign:
                # current evaluate point is the max P (with min surrogate value if not unique) in the expansion set
                xc_eval = np.copy(xhat)
                if n == 2:
                    plotting.safe_set_expansion_set_2Dplot(xE, safe_radius, Nm, expansion_set, safe_set, xc_eval, xmin, safe_fun_arg)
                # SafeDOGSplot.safe_contour_uncertainty_2Dplot(xE, safe_eval, y_safe, L_safe, Nm)

                # If xc_eval discovered by expansion set is close to evaluated data points,
                # refine the mesh a little bit to get a closer position to the actually safe region.
            elif not SinglePoint:
                # current evaluate point is the min surrogate model
                xi, ind_min = cartesian_grid.add_sup(xE, xU, ind_min)
                xc, yc, result, safe_estimate = constant_snopt_min.triangulation_search_bound_snopt(inter_par, xi, K, ind_min, y_safe, L_safe)

                xc_eval = Utils.safe_mesh_quantizer(xc, Nm, xE, y_safe, L_safe)

                if n == 2:
                    plotting.safe_Delaunay_simplices_2D_plot(xE, xc_eval, safe_radius, Nm, xmin, safe_fun_arg)
                # SafeDOGSplot.safe_contour_utility_2Dplot(xE, func_eval, safe_eval, y_safe, L_safe, Nm, '2Dutility_contour')

            else:
                # Safe expansion failed, single point trigger!
                # put some arbitrary things in xc_eval
                xc_eval = np.zeros((n, 1))
                pass

            if n == 1:
                plotting.safe_continuous_constantK_search_1D_plot(xE, xU, func_eval, safe_eval, L_safe, K, xc_eval, Nm)

            if Utils.mindis(xc_eval, xE)[0] < 1e-6 or SinglePoint:
                # 1. xc_eval already exists, mesh refine.
                # 2. Only one single point exists in the safe set, mesh too coarse, refine the mesh.
                Nm *= 2
                delta = 1 / Nm
                if not safe_expand_sign:
                    # If this mesh refinement is incurred by safe exploration, do not increase K
                    K *= 3
                print('===============  MESH Refinement  ===================')
                summary = {'alg': 'SafeLearn', 'xc_grid': xc_eval, 'xmin': xmin, 'Nm': Nm, 'y0': y0}
                plotting.print_summary(num_iter, k, kk, xE, yE, summary, 1)
                break
            else:
                xE = np.hstack((xE, xc_eval))
                yE = np.hstack((yE, func_eval(xc_eval)))
                y_safe = np.hstack((y_safe, safe_eval(xc_eval)))

                summary = {'alg': 'ori', 'xc_grid': xc_eval, 'xmin': xmin, 'Nm': Nm, 'y0': y0}
                plotting.print_summary(num_iter, k, kk, xE, yE, summary)

    Alg = {'name': alg_name}
    plotting.save_data(xE, yE, inter_par, ff, Alg)
    plotting.save_results_txt(xE, yE, fname, y0, xmin, Nm, Alg, ff, num_ini)
    plotting.dogs_summary_plot(xE, yE, y0, ff, xmin, alg_name)
