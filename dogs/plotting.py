"""
Created on Tue Oct 31 15:45:35 2017

@author: mousse
"""
import shapely
import      os
import      inspect
import      shutil
import      scipy
from        scipy.spatial       import Delaunay
import      numpy               as np
import      matplotlib.pyplot   as plt
import      matplotlib.cm       as cm
from        matplotlib.ticker   import PercentFormatter
from        scipy               import io
import      math
from        functools           import partial
from        itertools           import combinations
from        matplotlib.patches  import Polygon
from        dogs                import Utils
from        dogs                import interpolation
from        dogs                import SafeLearn
import      shapely

'''
plotting.py is implemented to generate results for AlphaDOGS and DeltaDOGS containing the following functions:
    dogs_summary                 :   generate summary plots, 
                                           candidate points 
                                           for each iteration
    plot_alpha_dogs              :   generate plots for AlphaDOGS
    plot_delta_dogs              :   generate plots for DeltaDOGS
    plot_detla_dogs_reduced_dim  :   generate plots for Dimension reduction DeltaDOGS
'''


############################################ Summary ############################################
def summary_display(sdogs):
    '''
    Display the optimization info at each iteration
    :param sdogs:
    :return:
    '''
    pos_reltv_error = str(
        np.round(np.linalg.norm(sdogs.xmin - sdogs.xE[:, np.argmin(sdogs.yE)]) / np.linalg.norm(sdogs.xmin) * 100,
                 decimals=4)) + '%'
    val_reltv_error = str(np.round(np.abs(np.min(sdogs.yE) - sdogs.y0) / np.abs(sdogs.y0) * 100, decimals=4)) + '%'

    cur_pos_reltv_err = str(
        np.round(np.linalg.norm(sdogs.xmin - sdogs.xE[:, -1]) / np.linalg.norm(sdogs.xmin) * 100, decimals=4)) + '%'
    cur_val_reltv_err = str(np.round(np.abs(sdogs.yE[-1] - sdogs.y0) / np.abs(sdogs.y0) * 100, decimals=4)) + '%'

    if sdogs.iter_type == 2:
        iteration_name = 'Safe Exploration'
    elif sdogs.iter_type == 4:
        iteration_name = 'Exploitation'
    elif sdogs.iter_type == 3 or 5:
        iteration_name = 'Mesh refine iteration'
    else:
        raise ValueError('Iteration type should be integer 1 to 5. ')
    print('============================   ', iteration_name, '   ============================')

    print(' %40s ' % 'No. Iteration', ' %30s ' % sdogs.iter)
    print(' %40s ' % 'Mesh size'    , ' %30s ' % sdogs.ms)
    print(' %40s ' % 'X-min'        , ' %30s ' % sdogs.xmin.T[0])
    print(' %40s ' % 'Target Value' , ' %30s ' % sdogs.y0)
    print("\n")
    print(' %40s ' % 'Candidate point'                  , ' %30s ' % sdogs.xE[:, np.argmin(sdogs.yE)])
    print(' %40s ' % 'Candidate FuncValue'              , ' %30s ' % np.min(sdogs.yE))
    print(' %40s ' % 'Candidate Safety'                 , ' %30s ' % np.min(sdogs.yS[:, np.argmin(sdogs.yE)]))
    print(' %40s ' % 'Candidate Position RelativeError' , ' %30s ' % pos_reltv_error)
    print(' %40s ' % 'Candidate Value RelativeError'    , ' %30s ' % val_reltv_error)
    print("\n")
    print(' %40s ' % 'Current point'                    , ' %30s ' % sdogs.xE[:, -1])
    print(' %40s ' % 'Current FuncValue'                , ' %30s ' % format(sdogs.yE[-1], '.4f'))
    print(' %40s ' % 'Current Safety'                   , ' %30s ' % format(np.min(sdogs.yS[-1]), '.4f'))
    print(' %40s ' % 'Current Position RelativeError'   , ' %30s ' % cur_pos_reltv_err)
    print(' %40s ' % 'Current Value RelativeError'      , ' %30s ' % cur_val_reltv_err)
    print("\n")


def summary(sdogs):
    '''
    Summary when optimization completed:
    - Plot the convergence
    - Save the result into .mat
    - Summary the optm results in text file.
    :param sdogs:
    :return:
    '''
    summary_plot(sdogs)
    result_saver(sdogs)
    save_results_txt(sdogs)


def summary_plot(sdogs):
    '''
    This function generates the summary information of DeltaDOGS optimization
    :param sdogs.yE       :     The function values evaluated at each iteration
    :param sdogs.y0       :     The target minimum of objective function.
    :param sdogs.folder   :     Identify the folder we want to save plots. "DDOGS" or "DimRed".
    :param sdogs.xmin     :     The global minimizer of test function, usually presented in row vector form.
    :param sdogs.ff       :     The number of trial.
    '''
    N = sdogs.yE.shape[0]  # number of iteration
    yE_best = np.zeros(N)
    yE_reltv_error = np.zeros(N)
    for i in range(N):
        yE_best[i] = min(sdogs.yE[:i+1])
        yE_reltv_error[i] = (np.min(sdogs.yE[:i+1]) - sdogs.y0) / np.abs(sdogs.y0) * 100
    # Plot the function value of candidate point for each iteration
    fig, ax1 = plt.subplots()
    plt.grid()
    # The x-axis is the function count, and the y-axis is the smallest value DELTA-DOGS had found.
    ax1.plot(np.arange(N) + 1, yE_best, label='Function value of Candidate point', c='b')
    ax1.plot(np.arange(N) + 1, sdogs.y0 * np.ones(N), label='Global Minimum', c='r')
    ax1.set_ylabel('Function value', color='b')
    ax1.tick_params('y', colors='b')
    plt.xlabel('Number of Evaluated Datapoints')
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1))

    # Plot the relative error on the right twin-axis.
    ax2 = ax1.twinx()
    ax2.plot(np.arange(N) + 1, yE_reltv_error, 'g--', label=r'Relative Error=$\frac{f_{min}-f_{0}}{|f_{0}|}$')
    ax2.set_ylabel('Relative Error', color='g')

    ax2.yaxis.set_major_formatter(PercentFormatter())
    ax2.tick_params('y', colors='g')
    plt.legend(loc='upper right', bbox_to_anchor=(1, 0.8))
    # Save the plot
    plt.savefig(sdogs.plot_folder + "/Candidate_point.eps", format='eps', dpi=500)
    plt.close(fig)
    ####################   Plot the distance of candidate x to xmin of each iteration  ##################
    fig2 = plt.figure()
    plt.grid()
    xE_dis = np.zeros(N)
    for i in range(N):
        index = np.argmin(sdogs.yE[:i+1])
        xE_dis[i] = np.linalg.norm(sdogs.xE[:, index].reshape(-1, 1) - sdogs.xmin)
    plt.plot(np.arange(N) + 1, xE_dis, label="Distance with global minimizer")
    plt.ylabel('Distance value')
    plt.xlabel('Number of Evaluated Datapoints')
    plt.legend(loc='upper right', bbox_to_anchor=(1, 0.9))
    plt.savefig(sdogs.plot_folder + "/Distance.eps", format='eps', dpi=500)
    plt.close(fig2)


def result_saver(sdogs):
    data = {}
    data['xE'] = sdogs.xE
    data['yE'] = sdogs.yE
    data['yS'] = sdogs.yS
    data['ms'] = sdogs.ms
    data['L_safe'] = sdogs.L_safe
    data['test_func_index'] = sdogs.test_func_index
    data['mesh_refine'] = sdogs.mesh_refine

    data['inter_par_method'] = sdogs.inter_par.method
    data['inter_par_w'] = sdogs.inter_par.w
    data['inter_par_v'] = sdogs.inter_par.v
    data['inter_par_xi'] = sdogs.inter_par.xi
    io.savemat(sdogs.plot_folder + '/data.mat', data)
    return


def save_results_txt(sdogs):
    best_error = (np.min(sdogs.yE) - sdogs.y0) / np.linalg.norm(sdogs.y0)
    value_error = np.abs(sdogs.yE - sdogs.y0) / np.linalg.norm(sdogs.y0)
    if np.min(value_error) > 0.01:
        # relative error > 0.01, optimization performance not good.
        idx = '1% accuracy failed'
    else:
        idx = np.min(np.where(value_error-0.01 < 0)[0]) - 1
    name = sdogs.alg_name
    with open(sdogs.plot_folder + '/OptmResults.txt', 'w') as f:
        f.write('=====  ' + name + ': General information report  ' + str(sdogs.xE.shape[0]) + 'D' + ' ' + sdogs.fname +
                '=====' + '\n')
        f.write('%40s %10s' % ('Surrogate type', str(sdogs.surrogate_type)) + '\n')
        f.write('%40s %10s' % ('Number of Input Dimensions = ', str(sdogs.xE.shape[0])) + '\n')
        f.write('%40s %10s' % ('Total Number of Function evaluations = ', str(sdogs.xE.shape[1])) + '\n')
        f.write('%40s %10s' % ('Mesh size when terminated = ', str(sdogs.ms)) + '\n')
        f.write('%40s %10s' % ('Best Value Error when terminated = ', str(np.round(best_error, decimals=6)*100)+'%') + '\n')
        f.write('%40s %10s' % ('Position with Best Value Error  = ', str(np.round(sdogs.xE[:, np.argmin(sdogs.yE)], decimals=6))) + '\n')
        f.write('%40s %10s' % ('Evaluations Required for 1% accuracy = ', str(idx)) + '\n')
    return


def safe_constantK_plot1D(sdogs):
    '''
    Given evaluated points set xE, and the objective function.
    Plot the interpolation, uncertainty function, continuous search function and estimated safe region in the first plot.
    Plot the safety functions in the second plot.
    :param sdogs.xE       :   Evaluated data points.
    :param sdogs.xU       :   Vertices of box domain.
    :param sdogs.fun_eval :   Objective function evaluator.
    :param sdogs.safe_eval:   Safety function evaluator.
    :param sdogs.L_safe   :   Lipschitz constant upper bound of safety functions.
    :param sdogs.K        :   Parameter K of constant-K
    :param sdogs.xc_min   :   Minimizer at the current iteration.
    :param sdogs.ms       :   Mesh size.
    :return:
    '''
    N  = sdogs.xE.shape[1]

    x      = np.linspace(0, 1, 2000)
    y, yp  = np.zeros(x.shape), np.zeros(x.shape)

    for i in range(x.shape[0]):
        # Calculate the truth values at all sites of x
        y[i]  = sdogs.fun_eval(x[i])
        # Calculate the interpolation values at all sites of x
        yp[i] = sdogs.inter_par.inter_val(x[i])

    # Determine the triangulation of 1D data
    xi        = np.hstack((sdogs.xE, sdogs.xU))
    sx        = sorted(range(xi.shape[1]), key=lambda x: xi[:, x])
    tri       = np.zeros((xi.shape[1] - 1, 2))
    tri[:, 0] = sx[:xi.shape[1] - 1]
    tri[:, 1] = sx[1:]
    tri       = tri.astype(np.int32)

    num_plot_points = 2000
    xe_plot = np.zeros((tri.shape[0], num_plot_points))
    e_plot  = np.zeros((tri.shape[0], num_plot_points))
    sc_plot = np.zeros((tri.shape[0], num_plot_points))

    K = sdogs.K / 3
    # rescale the uncertainty function such that we have a better illustration on continuous search function.

    # -----------  Determine the uncertainty function and the constant K continuous search function   -----------
    for ii in range(len(tri)):
        temp_x = np.copy(xi[:, tri[ii, :]])

        # Determine if the boundary corner exists or not in simplex
        exist = SafeLearn.unevaluated_vertices_identification(xi[:, tri[ii, :]], sdogs.xE)[0]

        x_ = np.linspace(temp_x[0, 0], temp_x[0, 1], num_plot_points)
        R2, xc = Utils.circhyp(xi[:, tri[ii, :]], sdogs.n)
        for jj in range(len(x_)):
            p = sdogs.inter_par.inter_val(x_[jj])

            if exist == 0:
                e_plot[ii, jj] = (R2 - np.linalg.norm(x_[jj] - xc) ** 2)
            else:
                e_plot[ii, jj] = SafeLearn.discrete_min_uncertainty(x_[jj], sdogs.xE, sdogs.b, sdogs.c)[0]
            sc_plot[ii, jj] = p - K * e_plot[ii, jj]

        xe_plot[ii, :] = x_

    safe_bound = np.zeros((N, 2))
    for i in range(N):
        safe_bound[i, :] = map(lambda xmap, ymap: np.hstack((xmap - ymap, xmap + ymap)), sdogs.xE[:, i], sdogs.safe_radius[i])
    # ==================  First plot =================
    # truth function, uncertainty funciton, continuous search function, evaluated data point, continuous minimizer
    # estimated safe region

    fig = plt.figure()
    plt.subplot(2, 1, 1)
    ax = fig.add_subplot(2, 1, 1)
    ax.set_facecolor((0.773, 0.769, 0.769))
    ax.grid(color='white')

    # plot the essentials for DeltaDOGS
    plt.plot(x, y, c='k')
    plt.plot(x, yp, c='b')

    for i in range(len(tri)):

        amplify_factor = 50
        exist = SafeLearn.unevaluated_vertices_identification(xi[:, tri[i, :]], sdogs.xE)[0]
        if exist == 0:
            plt.plot(xe_plot[i, :], amplify_factor * e_plot[i, :] - 5.5, c='g')
        else:
            plt.plot(xe_plot[i, :], amplify_factor * e_plot[i, :] - 5.5, 'g--')
        plt.plot(xe_plot[i, :], sc_plot[i, :], c='r')

    # scatter plot evaluated points
    plt.scatter(sdogs.xE, sdogs.yE, c='b', marker='s')
    # scatter plot the minimizer of continuous search function
    yc_min = sc_plot.flat[np.abs(xe_plot - sdogs.xc).argmin()]
    plt.scatter(sdogs.xc, yc_min, c='r', marker='^')

    # plot the safe region in cyan
    safe_region_plot(sdogs, safe_bound, 'objective')

    plt.ylim(-6.5, 3.5)
    # Y-axis invisable is commented, otherwise the grid line is not showing up
    # plt.gca().axes.get_yaxis().set_visible(False)
    # plt.gca().axes.get_yaxis().set_ticks([])

    # ==================  Second plot =================
    # safety function, estimated safe region
    plt.subplot(2, 1, 2)
    ax = fig.add_subplot(2, 1, 2)
    ax.set_facecolor((0.773, 0.769, 0.769))
    plt.gca().xaxis.grid(True, color='white')
    plt.gca().yaxis.grid(True, color='white')
    ax.grid(True, which='both')

    # underlying safety function
    y_safe_all = np.zeros(x.shape)
    for i in range(x.shape[0]):
        y_safe_all[i] = sdogs.safe_eval(x[i])
    plt.plot(x, y_safe_all)

    # plot where the safety is zero
    zero_indicator = np.zeros(x.shape)
    plt.plot(x, zero_indicator, c='k')

    # scatter plot the evaluated data points
    x_scatter = np.hstack((sdogs.xE, sdogs.xc))
    y_scatter = np.zeros(x_scatter.shape[1])
    for i in range(x_scatter.shape[1]):
        ind = np.argmin(np.abs(x_scatter[:, i] - x))
        y_scatter[i] = y_safe_all[ind]

    # Scatter plot the evaluated data and next sampling point on safe region plot
    plt.scatter(x_scatter[:, :-1][0], y_scatter[:-1], c='b', marker='s')
    plt.scatter(x_scatter[:, -1], y_scatter[-1], c='r', marker='^')

    # plot the safe region with the vertical bounds for visualization
    safe_region_plot(sdogs, safe_bound, 'safe')

    # the range -1 to 2.2 could be different for different safety function
    plt.ylim(-1, 2.2)
    # plt.gca().axes.get_yaxis().set_visible(False)

    plt.savefig(sdogs.plot_folder + '/pic' + str(int(sdogs.iter)) + '.eps', format='eps', dpi=500)
    plt.close(fig)

    # ==================  Third plot =================
    # uncertainty plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor((0.773, 0.769, 0.769))
    ax.grid(color='white')

    for i in range(len(tri)):
        exist = SafeLearn.unevaluated_vertices_identification(xi[:, tri[i, :]], sdogs.xE)[0]
        if exist == 0:
            plt.plot(xe_plot[i, :], amplify_factor * e_plot[i, :] - 5.5, c='g')
        else:
            plt.plot(xe_plot[i, :], amplify_factor * e_plot[i, :] - 5.5, 'g--')
    plt.savefig(sdogs.plot_folder + '/uncertainty' + str(int(sdogs.iter)) + '.eps', format='eps', dpi=500)
    plt.close(fig)


def safe_region_plot(sdogs, safe_bound, safe_keyword):
    '''
    Plot the estimated safe region with lipschitz constant. 1D param and multiple safe con.
    :param sdogs        :   The class of safe - DOGS
    :param safe_bound   :   Contains the lower and upper bound for each evaluated data points
    :param safe_keyword :   Control the y-axis lower bound of vertical safe line
    :return:
    '''
    N = safe_bound.shape[0]
    inf = 1e+20
    length = 500
    # First check if all points are safe
    for i in range(N):
        hard_safe_estimate = np.min(sdogs.yS, axis=0) - sdogs.L_safe * np.linalg.norm(sdogs.xE - sdogs.xE[:, i].reshape(-1, 1), axis=0)
        hard_safe_estimate = np.delete(hard_safe_estimate, i, 0)
        if (hard_safe_estimate < 0).all():
            raise ValueError('Some point is unsafe!')
    # Second plot the safe region
    xlow = inf
    xupp = -inf
    for i in range(2):
        lower_bound = max(safe_bound[i, 0], 0)
        upper_bound = min(safe_bound[i, 1], 1)
        x_ = np.linspace(lower_bound, upper_bound, length)
        y_ = 2 * np.ones(length)
        line = plt.plot(x_, y_)
        plt.setp(line, color='cyan', linewidth=3)

        xlow = min(xlow, lower_bound)
        xupp = max(xupp, upper_bound)
    # Third plot the vertical safe bound
    if safe_keyword == 'objective':
        ylow_vertical = np.linspace(-2, 2, length)
        yupp_vertical = np.linspace(-2, 2, length)
    else:
        ylow_vertical = np.linspace(np.min(sdogs.safe_eval(xlow)), 2, length)
        yupp_vertical = np.linspace(np.min(sdogs.safe_eval(xupp)), 2, length)
    xlow_y_vertical = xlow * np.ones(length)
    xupp_y_vertical = xupp * np.ones(length)

    plt.plot(xlow_y_vertical, ylow_vertical, color='cyan', linestyle='--')
    plt.plot(xupp_y_vertical, yupp_vertical, color='cyan', linestyle='--')


# TODO fix me, can adaptive be incorporated into constant plot?
def safe_adaptiveK_plot1D(xE, xU, fun_eval, safe_eval, L_safe, y0, xc_min, Nm):
    '''
    Given evaluated points set xE, and the objective function.
    1st plot: The interpolation, uncertainty function and estimated safe region.
    2nd plot: Adaptive-K continuous search function
    3rd plot: Zoomed-in version of Adaptive-K continuous search function
    4th plot: Safety functions plot
    :param xE       :   Evaluated data points.
    :param xU       :   Vertices of box domain
    :param fun_eval :   Objective function evaluator.
    :param safe_eval:   Safety function evaluator.
    :param L_safe   :   Lipschitz upper bound of safety functions.
    :param y0       :   Target value of f(x).
    :param xc_min   :   Minimizer of continuous search function.
    :param Nm       :   Mesh size
    :return         :   Plots.
    '''
    inf = 1.0e+20
    N  = xE.shape[1]
    yE = np.zeros(N)
    for i in range(N):
        yE[i] = fun_eval(xE[:, i])
    inter_par = interpolation.InterParams(xE, yE)
    inter_par, _ = inter_par.interpolateparameterization()

    num_plot_points = 5000

    x      = np.linspace(0, 1, 5000)
    y, yp  = np.zeros(x.shape), np.zeros(x.shape)

    for i in range(x.shape[0]):
        y[i]  = fun_eval(x[i])
        yp[i] = inter_par.inter_val(x[i])

    xi        = np.hstack(( xE, xU))
    sx        = sorted(range(xi.shape[1]), key=lambda x: xi[:, x])
    tri       = np.zeros((xi.shape[1] - 1, 2))
    tri[:, 0] = sx[:xi.shape[1] - 1]
    tri[:, 1] = sx[1:]
    tri       = tri.astype(np.int32)

    xe_plot = np.zeros((tri.shape[0], num_plot_points))
    e_plot  = np.zeros((tri.shape[0], num_plot_points))
    sc_plot = np.zeros((tri.shape[0], num_plot_points))

    n = xE.shape[0]
    Rmax, max_dis = SafeLearn.max_circumradius_delauany_simplex(xi, xE, tri)
    b, c, status  = SafeLearn.uncertainty_parameter_solver(Rmax, max_dis)

    for ii in range(len(tri)):
        temp_x = np.copy(xi[:, tri[ii, :]])

        # Determine if the boundary corner exists or not in simplex
        exist = SafeLearn.unevaluated_vertices_identification(xi[:, tri[ii, :]], xE)[0]

        x_ = np.linspace(temp_x[0, 0], temp_x[0, 1], num_plot_points)
        R2, xc = Utils.circhyp(xi[:, tri[ii, :]], n)
        for jj in range(len(x_)):
            p = inter_par.inter_val(x_[jj])

            if exist == 0:
                e_plot[ii, jj] = (R2 - np.linalg.norm(x_[jj] - xc) ** 2)
            else:
                e_plot[ii, jj] = SafeLearn.discrete_min_uncertainty(x_[jj], xE, b, c)[0]

            if e_plot[ii, jj] > 1e-12:
                sc_plot[ii, jj] = (p - y0) / e_plot[ii, jj]
            else:
                sc_plot[ii, jj] = inf

        xe_plot[ii, :] = np.copy(x_)

    safe_plot = {}
    # plot the safe region
    for ii in range(xE.shape[1]):
        y_safe = safe_eval(xE[:, ii])

        safe_index = []
        y_safe_plot = []
        safe_eval_lip = lambda x: y_safe - L_safe * np.sqrt(np.dot((x - xE[:, ii]).T, x - xE[:, ii]))
        for i in range(x.shape[0]):
            safe_val = safe_eval_lip(x[i])
            y_safe_plot.append(safe_val[0])
            if safe_val > 0:
                safe_index.append(i)
        name = str(ii)
        safe_plot[name] = [safe_index, y_safe_plot]


    # ==================  First plot =================
    # truth function, uncertainty funciton, evaluated data point, continuous minimizer, estimated safe region

    fig = plt.figure()
    plt.subplot(4, 1, 1)
    # plot the essentials for DeltaDOGS
    plt.plot(x, y, c='k')
    plt.plot(x, yp, c='b')

    for i in range(len(tri)):

        amplify_factor = 50
        exist = SafeLearn.unevaluated_vertices_identification(xi[:, tri[i, :]], xE)[0]
        if exist == 0:
            plt.plot(xe_plot[i, :], amplify_factor * e_plot[i, :] - 5.5, c='g')
        else:
            plt.plot(xe_plot[i, :], amplify_factor * e_plot[i, :] - 5.5, 'g--')

    # scatter plot evaluated points
    plt.scatter(xE, yE, c='b', marker='s')
    # scatter plot the minimizer of continuous search function
    yp_xcmin = yp[np.abs(x - xc_min).argmin()]
    plt.scatter(xc_min, yp_xcmin, c='r', marker='^')

    # plot the safe region in cyan

    # TODO fix the following lines, inputs wrong
    # xlow, xupp   = safe_region_plot(sdogs, safe_bound)
    y_vertical   = np.linspace(-2, 2, 100)
    # xlow_y_vertical = xlow * np.ones(100)
    # xupp_y_vertical = xupp * np.ones(100)
    # plt.plot(xlow_y_vertical, y_vertical, color='cyan', linestyle='--')
    # plt.plot(xupp_y_vertical, y_vertical, color='cyan', linestyle='--')

    plt.ylim(-6.5, 2.5)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)

    # ==================  Second plot - overview picture =================
    # adaptive-k continuous search function
    plt.subplot(4, 1, 2)
    big_range = 10000
    low_bound = -200
    # determine where the continuous search function exceeds the 0-200 range.
    xc_dis = inf

    for i in range(len(tri)):
        dlt = []
        for j in range(num_plot_points):
            if sc_plot[i, j] > big_range:
                dlt.append(j)
        x_temp_sc_plot = scipy.delete(xe_plot[i, :], dlt, 0)
        temp_sc = scipy.delete(sc_plot[i, :], dlt, 0)

        if x_temp_sc_plot.shape[0] > 0:
            if np.min(np.abs(x_temp_sc_plot - xc_min)) < xc_dis:
                sc_xcmin = temp_sc[np.abs(x_temp_sc_plot - xc_min).argmin()]
                xc_dis = np.min(np.abs(x_temp_sc_plot - xc_min))

        exist = SafeLearn.unevaluated_vertices_identification(xi[:, tri[i, :]], xE)[0]
        if exist == 0:
            plt.plot(x_temp_sc_plot, temp_sc, 'r')
        else:
            plt.plot(x_temp_sc_plot, temp_sc, 'r--')

    plt.scatter(xc_min, sc_xcmin, c='r', marker='^')

    plt.ylim(low_bound, big_range)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)

    #  ==================  Third plot - zoomed in pic =================
    plt.subplot(4, 1, 3)
    small_range = 1000
    # determine where the continuous search function exceeds the 0-200 range.
    for i in range(len(tri)):
        dlt = []
        for j in range(num_plot_points):
            if sc_plot[i, j] > big_range:
                dlt.append(j)
        x_temp_sc_plot = scipy.delete(xe_plot[i, :], dlt, 0)
        temp_sc = scipy.delete(sc_plot[i, :], dlt, 0)
        exist = SafeLearn.unevaluated_vertices_identification(xi[:, tri[i, :]], xE)[0]
        if exist == 0:
            plt.plot(x_temp_sc_plot, temp_sc, 'r')
        else:
            plt.plot(x_temp_sc_plot, temp_sc, 'r--')

    plt.scatter(xc_min, sc_xcmin, c='r', marker='^')
    plt.ylim(low_bound, small_range)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)

    # ==================  Fourth plot =================
    # safety function, estimated safe region
    plt.subplot(4, 1, 4)
    # underlying safety function
    y_safe_all = np.zeros(x.shape)
    for i in range(x.shape[0]):
        y_safe_all[i] = safe_eval(x[i])
    plt.plot(x, y_safe_all)

    # plot where the safety is zero
    zero_indicator = np.zeros(x.shape)
    plt.plot(x, zero_indicator, c='k')

    # scatter plot the evaluated data points
    x_scatter = np.hstack((xE, xc_min))
    y_scatter = np.zeros(x_scatter.shape[1])
    for i in range(x_scatter.shape[1]):
        ind = np.argmin(np.abs(x_scatter[:, i] - x))
        y_scatter[i] = y_safe_all[ind]

    plt.scatter(x_scatter[:, :-1][0], y_scatter[:-1], c='b', marker='s')
    plt.scatter(x_scatter[:, -1], y_scatter[-1], c='r', marker='^')

    # plot the vertical bound of safe region for visualization
    xlow, xupp = safe_region_plot(x, safe_plot)
    low_idx = np.argmin(np.abs(xlow - x))
    upp_idx = np.argmin(np.abs(xupp - x))
    ylow_vertical = np.linspace(y_safe_all[low_idx], 2, 100)
    yupp_vertical = np.linspace(y_safe_all[upp_idx], 2, 100)
    xlow_y_vertical = xlow * np.ones(100)
    xupp_y_vertical = xupp * np.ones(100)
    plt.plot(xlow_y_vertical, ylow_vertical, color='cyan', linestyle='--')
    plt.plot(xupp_y_vertical, yupp_vertical, color='cyan', linestyle='--')

    # the range -1 to 2.2 could be different for different safety function
    plt.ylim(-1, 2.2)
    plt.gca().axes.get_yaxis().set_visible(False)

    current_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    plot_folder = current_path[:-5] + "/plot/DDOGS/0"

    num_iter = xE.shape[1] - 1 + math.log(Nm/8, 2)
    plt.savefig(plot_folder + '/pic' + str(int(num_iter)) + '.eps', format='eps', dpi=500)
    plt.close(fig)

    # uncertainty plot
    fig = plt.figure()
    for i in range(len(tri)):
        exist = SafeLearn.unevaluated_vertices_identification(xi[:, tri[i, :]], xE)[0]
        if exist == 0:
            plt.plot(xe_plot[i, :], amplify_factor * e_plot[i, :] - 5.5, c='g')
        else:
            plt.plot(xe_plot[i, :], amplify_factor * e_plot[i, :] - 5.5, 'g--')
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.savefig(plot_folder + '/uncertainty' + str(int(num_iter)) + '.eps', format='eps', dpi=500)
    plt.close(fig)

    return
# ==================================  quadrotor plot ================================


def quadrotor_trajectory_plot(trajectory, ind):
    N = trajectory.shape[1]
    fig = plt.figure()
    t = np.linspace(0, 5, 350)
    x_des, y_des = Utils.quadrotor_desired_trajectory()

    for i in range(N):
        if i != ind:
            plt.plot(t, trajectory[:, i], color="0.5", linestyle='--', linewidth=.2, label='Other experiments')

    plt.plot(t, trajectory[:, ind], 'b', label='Optimized', linewidth=2.0)

    plt.plot(t, trajectory[:, 0], 'g--', label='Initial', linewidth=2.0)

    plt.plot(t, x_des, c='r', label='Reference')

    wall_position = 3.2 * np.ones(350)
    plt.plot(t, wall_position, 'r--', label='Obstacle')
    plt.legend()
    plt.ylabel(' x - position')
    plt.xlabel(' Time')
    plt.show()
    current_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    plot_folder = current_path[:-5] + "/plot/DDOGS/0"
    plt.savefig(plot_folder + '/trajectory.eps', format='eps', dpi=500)
    plt.close(fig)
    return


def quadrotor_contour_plot(xE, y_safe, L_safe, lb, ub, mesh_size):
    # TODO fix this into utility function 2D plot!!!!
    x = np.linspace(lb[0], ub[0], mesh_size)
    y = np.linspace(lb[1], ub[1], mesh_size)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape, float)
    safe_estimate = np.zeros(X.shape, float)

    for i in range(mesh_size):
        for j in range(mesh_size):
            point = np.array([[X[i, j]], [Y[i, j]]])
            normalized_point = Utils.normalize_bounds(point, lb, ub)
            safe_estimate[i, j] = np.max(y_safe - L_safe * np.linalg.norm(normalized_point - xE, axis=0))
            cost, safe_cost = Utils.quadrotor_cost_function(point)
            if safe_estimate[i, j] >= 0:
                Z[i, j] = cost
            else:
                if safe_cost >= 0:
                    Z[i, j] = -1
                else:
                    Z[i, j] = -20

    fig, ax = plt.subplots()
    l = np.linspace(np.min(Z), np.max(Z), 10)
    cp = ax.contourf(X, Y, Z, levels=l)
    ax.contour(cp, colors='k')

    physical_xE = Utils.physical_bounds(xE, lb, ub)
    plt.scatter(physical_xE[0, :], physical_xE[1, :], c='r', s=1.)
    plt.ylim(lb[1], ub[1])
    plt.xlim(lb[0], ub[0])
    plt.xlabel(r'$k_1$')
    plt.ylabel(r'$k_2$')

    current_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    plot_folder = current_path[:-5] + "/plot/DDOGS/0"
    plt.savefig(plot_folder + '/quadrotor_contour.eps', format='eps', dpi=500)
    plt.close(fig)
    return
# ==================================  2D Safe Set and Expansion Set plot =====================================


def plot2D(sdogs):
    fig = plt.figure()
    ax = plt.axes(frameon=False)
    plt.axis('equal')
    ax.get_xaxis().tick_bottom()

    # TODO change 100->1000/500, just for test.
    mesh_size = 100
    x = y = np.linspace(0, 1, mesh_size)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape, float)

    diff = 3 - sdogs.y0
    if sdogs.xc is not None:
        yS_xc = sdogs.safe_eval(sdogs.xc)
    else:
        yS_xc = np.zeros((sdogs.M, 1))

    for i in range(mesh_size):
        for j in range(mesh_size):
            point = np.array([[X[i, j]], [Y[i, j]]])
            point_safe = sdogs.safe_eval(point)
            point_estimate_safe = np.min(sdogs.yS, axis=0) - sdogs.L_safe * np.linalg.norm(point - sdogs.xE, axis=0)
            if point_safe.min() > 0:  # underlying safe point
                if point_estimate_safe.max() > 0:  # known safe point
                    if sdogs.iter_type == 'explore' or 'initial':  # Safe exploration iteration
                        Z[i, j] = 0  # known safe
                    else:  # Exploitation
                        Z[i, j] = diff + sdogs.func_eval(point)
                else:  # Known to be unsafe: 1. newly expanded safe region; 2. known unsafe
                    if sdogs.iter_type != 'refine' and sdogs.iter_type != 'initial' and (yS_xc - sdogs.L_safe * np.linalg.norm(point - sdogs.xc)).min() > 0:
                        Z[i, j] = 1  # Newly expanded safe region
                    else:
                        Z[i, j] = 2  # Currently known to be unsafe but underlying safe
            else:
                Z[i, j] = 3  # Unsafe
    cmap = cm.get_cmap('Paired')
    plt.contour(X, Y, Z, levels=[2, 3], colors='r', linestyles='--', linewidths=1)
    cs = plt.contourf(X, Y, Z, alpha=0.75, cmap=cmap)

    # Set the safe region to be white
    cs.set_clim(0.1, np.max(Z))
    cs.cmap.set_under('w')
    # Determine the color and labels for each region
    # This is too hard, the color changes at each time script is called.
    # Fix the scatter plot
    plt.scatter(sdogs.xE[0, :], sdogs.xE[1, :], c='w', marker='s', s=12, edgecolor='k', label='Evaluated')
    plt.scatter(sdogs.xmin[0], sdogs.xmin[1], c='r', marker='*', s=12)
    if sdogs.xc is None:
        pass
    elif sdogs.iter_type == 'refine' and sdogs.SinglePoint:
        pass
    else:
        plt.scatter(sdogs.xc[0], sdogs.xc[1], c='r', marker='s', s=10, edgecolor='k', label='Sample point')

    if sdogs.iter_type == 'refine':
        title_ = 'Mesh refine'
    elif sdogs.iter_type == 'explore':
        title_ = 'Safe exploration'
    elif sdogs.iter_type == 'initial':
        title_ = 'Initialization'
    elif sdogs.iter_type == 'exploit':
        title_ = 'Exploitation'
    else:
        title_ = ' '
    plt.title(title_)

    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.savefig(sdogs.plot_folder + '/2Dplot' + str(sdogs.iter) + '.pdf', format='pdf', dpi=500, transparent=True)
    plt.close(fig)


def safe_expansion_set_plot(sdogs):
    """

    :param sdogs:
    :return:
    """
    assert sdogs.n == 2, 'This is 2D plot maker.'

    fig = plt.figure(figsize=[10, 10])
    ax = plt.axes(frameon=True)
    ax.get_xaxis().tick_bottom()

    # plot the boundary of box domain
    box_domain_boundary_2D_plot(ax)

    # plot the known to be safe region
    known_safe_region_plot(sdogs, ax)

    # plot the newly expanded safe region, plot the safe ball centered at sdogs.xhat
    # TODO fix the newly expanded safe set
    # newly_expanded_safe_region_plot(sdogs, ax)

    # Scatter plot the expansion set
    c1 = '#FFD700'

    scatter_point_size = 30
    scatter_point_zorder = 2

    # scatter plot the expansion set
    ax.scatter(sdogs.expansion_set[0, :], sdogs.expansion_set[1, :], c=c1, marker='s', s=scatter_point_size, alpha=1, edgecolors='k', zorder=scatter_point_zorder)

    # scatter plot the unique part of safe set
    unique_safe_set = np.empty(shape=[sdogs.n, 0])
    for i in range(sdogs.safe_set.shape[1]):
        query = sdogs.safe_set[:, i].reshape(-1, 1)
        if np.min(np.linalg.norm(sdogs.expansion_set - query, axis=0)) > 1e-10:
            unique_safe_set = np.hstack((unique_safe_set, query))

    if unique_safe_set.shape[1] > 0:
        c2 = '#228B22'
        ax.scatter(unique_safe_set[0, :], unique_safe_set[1, :], c=c2, marker='o', s=scatter_point_size, alpha=1, edgecolors='k', zorder=scatter_point_zorder)

    # plot the iteration result: safe exploration or exploitation from optimization
    if sdogs.iter_type == 1:
        # Safe exploration iteration, scatter plot the safe expander
        plt.scatter(sdogs.xhat[0], sdogs.xhat[1], c='r', marker='s', s=scatter_point_size, alpha=1, edgecolors='k', zorder=scatter_point_zorder)
    elif sdogs.iter_type == 2:
        #
        plt.scatter(sdogs.xc[0], sdogs.xc[1], c='r', marker='s', s=scatter_point_size, zorder=scatter_point_zorder)

    # Scatter plot the global minimizer
    ax.scatter(sdogs.xmin[0], sdogs.xmin[1], c='r', marker='*', s=scatter_point_size, zorder=scatter_point_zorder)

    ax.set_ylim(0-0.05, 1+0.05)
    ax.set_xlim(0-0.05, 1+0.05)
    ax.set_axis_off()

    # plt.savefig doesn't work after calling show because the current figure has been reset.
    if sdogs.plot_save:
        plt.savefig(os.path.join(sdogs.plot_folder, '2D_safe_expansion', str(sdogs.iter), '.pdf'), format='pdf', dpi=500, transparent=True)
    if sdogs.plot_display:
        plt.show()

    plt.close(fig)


def known_safe_region_plot(sdogs, axs):
    from shapely.ops import unary_union
    from shapely.geometry import Point
    from shapely.geometry.multipolygon import MultiPolygon

    # compute the union of each safe ball
    circles = []
    for i, point in enumerate(sdogs.xE.T):
        # TODO fix the safe radius later
        point_safe_radius = np.min(sdogs.yS[:, i], axis=0) / sdogs.L_safe
        circles.append(Point(point.tolist()).buffer(point_safe_radius))

    # if the union of safe ball is only one polygon, it would be Polygon type; We need a Multipolygon which has geoms attribute
    if type(circles) is MultiPolygon:
        known_safe_region = unary_union(circles)
    else:
        known_safe_region = MultiPolygon([unary_union(circles)])

    # plot the union of known safe region
    # https://gis.stackexchange.com/questions/353082/plotting-shapely-multipolygon-using-matplotlib
    for geom in known_safe_region.geoms:
        xs, ys = geom.exterior.xy
        axs.fill(xs, ys, alpha=0.5, ec='k', fc='w', zorder=1)

# def newly_expanded_safe_region_plot(sdogs, axs):





# ==================================  2D Utility and Uncertainty Contour plot ================================


def safe_contour_uncertainty_2Dplot(xE, safe_eval, y_safe, L_safe, Nm):
    n = xE.shape[0]
    xU = np.array([[0, 1, 1, 0], [0, 0, 1, 1]])
    xi = np.hstack((xE, xU))
    options = 'Qt Qbb Qc' if n <= 3 else 'Qt Qbb Qc Qx'
    DT = Delaunay(xi.T, qhull_options=options)

    Rmax, max_dis = SafeLearn.max_circumradius_delauany_simplex(xi, xE, DT.simplices)
    b, c, status = SafeLearn.uncertainty_parameter_solver(Rmax, max_dis)
    uncertainty_eval = partial(SafeLearn.uncertainty_calculator, DT, xi, xE, b, c)

    safe_contour_utility_2Dplot(xE, uncertainty_eval, safe_eval, y_safe, L_safe, Nm, '2Duncertainty_contour')

    return


def safe_contour_utility_2Dplot(xE, func_eval, safe_eval, y_safe, L_safe, Nm, save_name):
    '''
    - Notice that there is no need to call this function at EVERY iteration, the mesh size 500 * 500 is costful.
    - Plot the parameter sampling in 2D space. Distinguish the unsafe region and safe region;
    - Inside the known safe region plot the values of performance measurement as the background.
    - Also show the scatter plot of evaluated data points.
    :param xE       :   Evaluated data points.
    :param func_eval:   Objective function calculator.
    :param safe_eval:   Safety function calculator.
    :param y_safe   :   safety function values.
    :param L_safe   :   Lipschitz upper bound of safety functions.
    :param Nm       :   Current mesh size, for computing the iteration number
    :param save_name:   Indicate the name of the file that stores the figure.
    :return         :   Contour plot.
    '''
    mesh_size = 500
    x = y = np.linspace(0, 1, mesh_size)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape, float)

    for i in range(mesh_size):
        for j in range(mesh_size):
            point = np.array([[X[i, j]], [Y[i, j]]])
            safe_estimate = np.min(y_safe, axis=0) - L_safe * np.linalg.norm(point - xE, axis=0)
            f = func_eval(point)
            psi = safe_eval(point)
            if (safe_estimate > 0).any():
                # This point is known to be safe.
                Z[i, j] = np.copy(f)
            else:
                if psi >= 0:
                    # This point locates inside the safe region, but not yet known to be safe
                    Z[i, j] = -10
                else:
                    # This point locates inside the unsafe region.
                    Z[i, j] = -20

    fig, ax = plt.subplots()
    l = np.linspace(np.min(Z), np.max(Z), 10)
    cp = ax.contourf(X, Y, Z, levels=l)
    plt.scatter(xE[0, :], xE[1, :], c='r', s=1.)
    ax.contour(cp, colors='k')
    fig.colorbar(cp)

    num_iter = int(xE.shape[1] - 1 + math.log(Nm/8, 2))
    current_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    plot_folder = current_path[:-5] + "/plot/DDOGS/0"
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(plot_folder + '/' + save_name + str(num_iter) + '.eps', format='eps', dpi=500)
    plt.close(fig)
    return
# ==========================  plot 2D Delaunay simplices and 2 kinds of uncertainty function   ========================


def initial_Delaunay_simplices_2D_plot(xE):
    '''
    This is an illustration of the initial process of safe learning on 2D problem.
    This plot function is for presentation.
    1st plot    :   Scatter plot.
    2nd plot    :   Delaunay simplex plot.
    :param xE   :   Evaluated data points
    :return     :   2 plot.
    '''
    n = xE.shape[0]
    assert n == 2, 'This is 2D plot maker.'
    xU = np.array([[0, 1, 1, 0], [0, 0, 1, 1]])
    xi = np.hstack((xE, xU))

    # First plot, boundary and scatter plot of xE, xU.
    fig = plt.figure()
    # remove background frame lines boundary
    ax1 = plt.axes(frameon=False)
    plt.axis('equal')
    ax1.get_xaxis().tick_bottom()

    # scatter the vertices of domain and evaluated data points
    scatter_evaluated_plot(xE, xU)

    # plot the boundary of the box domain
    box_domain_boundary_2D_plot()

    plt.ylim(0-0.05, 1+0.05)
    plt.xlim(0-0.05, 1+0.05)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)

    current_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    plot_folder = current_path[:-5] + "/plot/DDOGS/0"
    plt.savefig(plot_folder + '/2D_Delaunay_plotinitial1.eps', format='eps', dpi=500, transparent=True)
    plt.close(fig)

    # Second plot, first Delaunay triangulation.
    fig = plt.figure()
    # remove background frame lines boundary
    ax1 = plt.axes(frameon=False)
    plt.axis('equal')
    ax1.get_xaxis().tick_bottom()

    # scatter the vertices of domain and evaluated data points
    scatter_evaluated_plot(xE, xU)

    Delaunay_simplices_boundary_plot(xi)

    # plot the boundaries of box domain
    box_domain_boundary_2D_plot()

    # make the contour plot of Delaunay simplices
    contour_interior_exterior_Delaunay_2D_plot(xE)

    plt.ylim(0-0.05, 1+0.05)
    plt.xlim(0-0.05, 1+0.05)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)

    plt.savefig(plot_folder + '/2D_Delaunay_plot_initial2.pdf', format='pdf', dpi=500, transparent=True)
    plt.close(fig)

    return


def safe_Delaunay_simplices_2D_plot(xE, xmin, radius, Nm, xstar, safe_fun_arg):
    '''
    - Show the Delaunay simplices boundary.
    - Show the color-filled in interior and exterior Delaunay simplices.
    - Show the known safe sphere of each evaluated data points.
    Notice that .eps file does not support transparency. Thus using .pdf file.
    :param xE:   :      Evalauted data points.
    :param xmin  :      The point of interest, either the maximizer of uncertainty in expansion set,
                        or the minimizer of surrogate model.
    :param radius:      safe region size at each evaluated points.
    :return:
    '''
    n = xE.shape[0]
    assert n == 2, 'This is 2D plot maker.'

    fig = plt.figure()
    ax1 = plt.axes(frameon=False)   # remove the frame
    plt.axis('equal')
    ax1.get_xaxis().tick_bottom()
    # TODO num_iter -> sdogs.iter?
    num_iter = int(xE.shape[1] - 1 + math.log(Nm/8, 2))

    xU = np.array([[0, 1, 1, 0], [0, 0, 1, 1]])
    Delaunay_simplices_plot_combination(xE, xU, radius, ax1, safe_fun_arg)

    # minimizer of search function within the estimated safe region
    plt.scatter(xmin[0], xmin[1], c='r', marker='s', s=15)

    # Scatter plot the global minimizer
    plt.scatter(xstar[0], xstar[1], c='r', marker='*', s=10)

    plt.ylim(0-0.05, 1+0.05)
    plt.xlim(0-0.05, 1+0.05)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)

    # save fig
    current_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    plot_folder = current_path[:-5] + "/plot/DDOGS/0"
    plt.savefig(plot_folder + '/2D_safe_expansion' + str(num_iter) + '.pdf', format='pdf', dpi=500, transparent=True)
    plt.close(fig)
    return


def Delaunay_simplices_plot_combination(xE, xU, radius, ax1, safe_fun_arg):
    xi = np.hstack((xE, xU))
    # scatter plot xE and xU
    scatter_evaluated_plot(xE, xU)

    # if need delaunay info:

    # plot the boundary of Delaunay simplices
    Delaunay_simplices_boundary_plot(xi)
    # make the contour plot of Delaunay simplices
    contour_interior_exterior_Delaunay_2D_plot(xE)

    # plot the known to be safe region, radius is the size of safe region
    safe_sphere_plot(xE, radius, ax1)

    # plot the boundaries of box domain
    box_domain_boundary_2D_plot()

    # plot the actually safe region
    if safe_fun_arg == 1:
        safe = np.array([[0.1, .9, .9, .1], [.1, .1, .9, .9]])
        plt.plot(safe[0, :], safe[1, :], 'r--')
        plt.plot(safe[0, [3, 0]], safe[1, [3, 0]], 'r--')
    elif safe_fun_arg == 4:
        quadratic_range = .5
        safe1 = np.array([[1, .9], [.9, 1]])
        plt.plot(safe1[0, :], safe1[1, :], 'r--')
        x = np.linspace(0, quadratic_range-0.00001, 1000)
        y = np.sqrt(quadratic_range**2 - x**2)
        plt.plot(x, y, 'r--')
    return


def scatter_evaluated_plot(xE, xU):
    '''
    Scatter plot the evaluated data xE and vertices of box boundary.
    :param xE:      Evaluated data points.
    :param xU:      Vertices of box domain boundary.
    :return:
    '''
    plt.scatter(xU[0, :], xU[1, :], c='k', marker='s', s=8)
    plt.scatter(xE[0, :], xE[1, :], c='b', marker='s', s=15)
    return


def safe_sphere_plot(xE, radius, ax1):
    '''
    plot the known safe region for each point with the radius in 'ms'.
    :param xE    : Evaluated data points.
    :param radius: Safe radius.
    :return:
    '''
    transparency = .15
    for i in range(xE.shape[1]):
        circle = plt.Circle(tuple(xE[:, i]), radius=radius[i], color='b', alpha=transparency)
        ax1.add_artist(circle)
    return


def contour_interior_exterior_Delaunay_2D_plot(xE):
    '''
    Color the inside of interior and exterior Delaunay simplices.
    Red   -> Interior D-simplex
    Green -> Exterior D-simplex
    :param xE:
    :return:
    '''
    n = xE.shape[0]
    assert n == 2, 'This is 2D plot maker.'
    xU = np.array([[0, 1, 1, 0], [0, 0, 1, 1]])
    xi = np.hstack((xE, xU))
    options = 'Qt Qbb Qc' if n <= 3 else 'Qt Qbb Qc Qx'
    DT = Delaunay(xi.T, qhull_options=options)
    transparency = .4
    axes = plt.gca()
    for ii in range(DT.simplices.shape[0]):
        simplex = xi[:, DT.simplices[ii]]
        exist = SafeLearn.unevaluated_vertices_identification(simplex, xE)[0]
        if exist == 0:
            axes.add_patch(Polygon([simplex[:, 0], simplex[:, 1], simplex[:, 2]],
                                   closed=True, facecolor='red', alpha=transparency))

        else:
            axes.add_patch(Polygon([simplex[:, 0], simplex[:, 1], simplex[:, 2]],
                                   closed=True, facecolor='green', alpha=transparency))
    return


def Delaunay_simplices_boundary_plot(xi):
    '''
    Plot the boundary of each Delaunay simplex.
    :param xi:      Evaluated data points xE and vertices of box domain xU.
    :return:
    '''
    n = xi.shape[0]
    options = 'Qt Qbb Qc' if n <= 3 else 'Qt Qbb Qc Qx'
    DT = Delaunay(xi.T, qhull_options=options)

    # plot the boundaries of Delaunay simplices
    nlist = np.arange(1, 3 + 1)
    C = np.array([list(c) for c in combinations(nlist, 2)])
    for ii in range(DT.simplices.shape[0]):
        simplex = xi[:, DT.simplices[ii]]
        for jj in range(C.shape[0]):
            vertex1 = simplex[:, C[jj, 0] - 1]
            vertex2 = simplex[:, C[jj, 1] - 1]
            plt.plot([vertex1[0], vertex2[0]], [vertex1[1], vertex2[1]], c='g', linewidth=1)
    return


def box_domain_boundary_2D_plot(ax):
    '''
    plot the boundaries of box domain, xU should be self written in the proper order.
    :param xU:  Box domain vertices.
    :return  :  Plot of box domain.
    '''
    boundary_points = np.array([[0, 0, 1, 1], [0, 1, 1, 0]])
    ax.plot(boundary_points[0, :], boundary_points[1, :], c='k')
    ax.plot(boundary_points[0, [3, 0]], boundary_points[1, [3, 0]], c='k')
    ax.fill(boundary_points[0], boundary_points[1], c='#DCDCDC', zorder=0)

