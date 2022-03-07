"""
This is the SafeDogs class that stores the main data structure used in SDOGS algorithm

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
import os
import inspect
import shutil
import numpy as np
from dogs import interpolation
from dogs import Utils
# from dogs import constantK_snopt_safelearning
# from dogs import adaptiveK_snopt_safelearning
from dogs import cartesian_grid
# from dogs import SafeDOGSplot
from dogs import SafeLearn
from dogs import exterior_uncertainty as extr_uncer

from    scipy.spatial   import Delaunay

# TODO
# __all__= []


class SafeDogs:
    """
    Outline
    stores the data structure of SDOGS optimization, including the objective & safe function evaluation handle,
    evaluated data and corresponding values, optional parameters

    """
    def __init__(self, obj_fun, safe_fun, optionalParams,
                 A, b):
        # Generate the test function caller
        self.func_eval = obj_fun.fun
        self.safe_eval = safe_fun.fun

        # Physical bounds, for function evaluation
        if hasattr(obj_fun, 'ub'):
            self.physical_ub = np.copy(obj_fun.ub)
        else:
            raise NameError("The property 'ub', upper bound of the objective function 'obj_fun', is not defined. ")

        if hasattr(obj_fun, 'lb'):
            self.physical_lb = np.copy(obj_fun.lb)
        else:
            raise NameError("The property 'lb', lower bound of the objective function 'obj_fun', is not defined. ")

        # Dimension of input parameters and number of safe functions
        self.n = self.physical_ub.shape[0]
        if hasattr(safe_fun, 'm'):
            self.m = np.copy(safe_fun.m)
        else:
            raise NameError("The property 'm', number of safe functions of 'safe_fun', is not defined. ")

        # Normalized vector bounds of parameter space, for generating bounds
        self.lb = np.zeros((self.n, 1))
        self.ub = np.ones((self.n, 1))

        # Take in the initial safe sites
        if optionalParams.get_option('Initial sites') is not None:
            self.x0 = optionalParams.get_option('Initial sites')
        else:
            raise NameError("The keyword 'Initial sites' is not defined in SdogsOptions. ")
        self.N = self.x0.shape[1]

        if obj_fun.xmin is not None:
            self.xmin = np.copy(obj_fun.xmin)
            self.y0   = np.copy(obj_fun.y0)

        self.fname      = obj_fun.func_name
        self.safe_fname = safe_fun.func_name

        self.safe_Lipschitz = np.zeros((self.m, 1)) # m-by-1 2D np.ndarray; L_{\psi_i} in the CDC paper
        self.L_safe         = np.zeros((1, 1))      # \bar{L} in the CDC paper
        # TODO modify how to feed in Lipschitz, maybe also have it as an attribute
        #  also, get_lipschitz is lower case in l, it could be misleading
        if callable(safe_fun.get_lipschitz):
            self.safe_Lipschitz = safe_fun.get_lipschitz()
            self.L_safe = np.max(self.safe_Lipschitz)
        else:
            raise NameError(f"Safe function {self.safe_fname} does not have method 'get_lipschitz'. ")

        # Mesh grid info, get the (initial) mesh size
        if optionalParams.get_option('Initial mesh size') is not None:
            # The number of intervals for each dimension
            self.initial_ms   = optionalParams.get_option('Initial mesh size')
            self.ms          = optionalParams.get_option('Initial mesh size')
        else:
            raise NameError("The keyword 'Initial mesh size' is not defined in SdogsOptions. ")

        # get the maximum number of mesh refinement
        if optionalParams.get_option('Number of mesh refinement') is not None:
            self.mesh_refine = optionalParams.get_option('Number of mesh refinement')
        else:
            raise NameError("The keyword 'Number of mesh refinement' is not defined in SdogsOptions. ")

        # Parameters for safe expansion
        self.mesh_size = 1 / self.ms
        self.epsilon   = 1e-4

        self.max_mesh          = self.ms * 2 ** self.mesh_refine
        # self.current_mesh_size = np.min(self.ub-self.lb) / self.ms

        # Iteration info
        self.iter      = 0
        self.iter_type = 1
        ''' possible values for iter_type:
        iter_type == 1 -> 'initial'
        iter_type == 2 -> 'safe_mesh_refinement'
        iter_type == 3 -> 'exploitation'
        iter_type == 4 -> 'exploiting_mesh_refinement'''

        # define the maximum number of iteration of optimization
        if optionalParams.get_option('Maximum iteration') is not None:
            self.iter_max = optionalParams.get_option('Maximum iteration')
        else:
            # if it is not defined, let it be 100 * number of mesh refinement
            self.iter_max = 50 * self.mesh_refine

        # Define the linear constraints, Ax <= b.
        # TODO fix the linear constraints
        self.Ain = A
        self.Bin = b

        # Record the Delaunay triangulation
        self.DT  = None
        self.tri = None

        # Parameters for exterior uncertainty function
        self.Rmax    = None
        self.max_dis = None
        self.b       = None
        self.c       = None

        self.safe_estimate_fun = None
        self.safe_radius_fun = None
        # TODO fix me
        # if optionalParams.get_option('Safe radius estimate CDC version'):
        #     self.safe_estimate_fun = self.
        #     self.safe_radius_fun = self.
        # elif optionalParams.get_option('Safe radius estimate Journal version'):
        #     self.safe_estimate_fun = self.
        #     self.safe_radius_fun = self.
        # else:
        #     ValueError("The safe estimate is not specified, change the option 'Safe radius CDC version' or 'Safe radius Journal version'. ")

        # Parameters generated for expansion set and safe set
        self.safe_expand_sign = None
        self.single_point     = None
        self.xhat             = None    # potential maximizer of function P (paper) in the expansion set
        self.ehat             = None    # the uncertainty function value e corresponding to xhat
        self.safe_set         = None

        # The data structures 'expansion_set', 'expansion_set_uncertainty' and 'expansion_set_children_count'
        # are one-to-one correspondence.
        self.expansion_set = None
        self.expansion_set_uncertainty = None       # The uncertainty of each point in expansion set
        self.expansion_set_children_count = None    # Essentially P_k^ell function in the paper

        # Support points and initial point
        self.xU = Utils.bounds(self.lb, self.ub, self.n)

        # Define the surrogate model
        self.constantK_surrogate = optionalParams.get_option('Constant surrogate')
        self.adaptiveK_surrogate = optionalParams.get_option('Adaptive surrogate')

        if not self.constantK_surrogate and not self.adaptiveK_surrogate:
            raise ValueError("Define one of the surrogate model in SdogsOptions, 'Constant surrogate' or "
                       "'Adaptive surrogate'. ")

        if self.constantK_surrogate:
            if optionalParams.get_option('Constant K') is not None:
                self.K = optionalParams.get_option('Constant K')
            else:
                raise TypeError("Constant K surrogate is declared without defining parameter K in the optionalParams 'Constant K'. ")

        if self.adaptiveK_surrogate:
            raise ValueError("Currently 'Adaptive surrogate' is not well-defined for SDOGS optimization. ")

        # TODO fix those 3
        # self.solver_type = optionalParams.get_option('')
        # self.surrogate_type = surrogate_type
        # self.surrogate_solver, self.surrogate_eval = self.solver_generator(y0, K)

        # initialization of the data set
        self.xE = np.copy(self.x0)
        self.yE = np.zeros(self.xE.shape[1])
        self.yS = np.empty(shape=[self.m, 0])

        # TODO I think we could also do a vector input for function evaluation handle,
        #  but this is not that important, cuz every iteration we only have 1 data point to be evaluated
        for i in range(self.xE.shape[1]):
            self.yE[i] = self.func_eval(self.xE[:, i])
            self.yS = np.hstack((self.yS, self.safe_eval(self.xE[:, i])))
        # At the beginning of each iteration of the main script, will call SafeLearn.delaunay_triangulation
        # there will also concatenate xU and xE
        self.xi = np.hstack((self.xU, self.xE))

        # Define the interpolation for f(x)
        self.inter_par = None
        self.yp = None
        # Store the local minimizer index of yp
        self.yp_min_ind = None
        self.x_yp_min = None

        # Define the interpolation for psi(x)
        self.safe_inter_par = None
        self.yp_safe = None

        # Define the safe radius for each evaluated data points
        self.safe_radius = None

        '''Define the minimizer of continuous search function, parameter to be evaluated xc, and its corresponding
        search function value yc, which is the optimization result from (global/local/safe)
        and the estimated safe conditions at xc.'''
        self.xc = None
        self.yc = None
        self.optm_result   = None
        '''optm_result: indicate the xc is the minimizer of the surrogate inside which type of simplex 
        1 -> global simplex
        2 -> global and safe simplex
        3 -> local simplex
        '''
        self.xc_safe_est = None    # the safe estimate at xc

        '''
        The following is about the Plotting part, put the plot part at the last section in __init__ function.
        '''
        # Although it is stochastic(?), just display the behavior instead of the actual data to show the trend.
        # Define the name of directory to store the figures

        # Plot control
        self.plot_display = optionalParams.get_option('Plot display')
        self.plot_save    = optionalParams.get_option('Plot save')

        # Define the range of plot
        # TODO fix plot range
        self.plot_ylow = None
        self.plot_yupp = None

        # No matter you call self.current_path in an example script, or in the debug mode, it all returns SDOGS/dogs
        self.current_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        # generate the images folder under SDOGS
        self.plot_folder = os.path.join(os.path.dirname(self.current_path), 'images')
        if os.path.isdir(self.plot_folder):
            files = os.listdir(self.plot_folder)
            png_files = [file for file in files if file.endswith(".png")]
            for file in png_files:
                path_to_file = os.path.join(self.plot_folder, file)
                os.remove(path_to_file)
        else:
            os.makedirs(self.plot_folder)
        # TODO delete this?
        self.func_path = None

        # TODO fix plot
        # self.plot_initialize()
        self.func_cost = None
        self.func_name = None
        self.initial_prior = None
        self.initial_need = True

        self.alg_name = "SDOGS"
        # TODO
        # create two new functions, one compute the (estimated) safe radius using CDC version
        # another compute the (estimated) safe radius using new version

        # TODO fix plot! define them in another class object, dont mess it with SafeDogs, here only data structure
        # if self.n == 1:
        #     if self.surrogate_type == 'constant':
        #         self.plot_func = SafeDOGSplot.safe_constantK_plot1D
        #     else:
        #         self.plot_func = SafeDOGSplot.safe_adaptiveK_plot1D
        # elif self.n == 2:
        #     self.plot_func = SafeDOGSplot.plot2D
        # else:
        #     self.plot_func = None

        # Generate the plot of test function
        # TODO fix plot! define them in another class object, dont mess it with SafeDogs, here only data structure
        # self.initial_visual_plot1D()


        # TODO I think test_func & safe_func part are done, safe delete them once you believe this
        # Define the test function index number
        # self.test_func_index = test_func_info['test_func_index']
        # self.safe_func_index = test_func_info['safe_func_index']
        # if test_func_info['type'] == 'test':
        #     # Define the physical bounds for objective function and safety functions
        #     self.physical_ub = None
        #     self.physical_lb = None
        #     # Define the dimension of safety functions
        #     self.M = None
        #     # Define the safe initial point
        #     self.x0 = None
        #     # Initialize the test function info, global minimizer 'xmin', target value 'y0', function name 'fname'
        #     # AND: The upper bound of Lipschitz constant for Safety functions.
        #     self.xmin = None
        #     self.y0 = None
        #     self.fname = None
        #     self.safe_name = None
        #     self.L_safe = None
        #
        # else:
        #     # ==================
        #     # UNDER PERPARATION!
        #     # ==================
        #     # Physical bounds of parameter space
        #     self.physical_ub = test_func_info['physical_ub']
        #     self.physical_lb = test_func_info['physical_lb']
        #     # Define the dimension of safety functions
        #     self.M = test_func_info['M']
        #     # Define the safe initial point
        #     self.x0 = test_func_info['x0']
        #     # Initialize the test function info, global minimizer 'xmin', target value 'y0', function name 'fname'
        #     # AND: The upper bound of Lipschitz constant for Safety functions.
        #     self.xmin = test_func_info['xmin']
        #     self.y0 = test_func_info['y0']
        #     self.fname = test_func_info['fname']
        #     self.safe_name = test_func_info['safe_name']
        #     self.L_safe = test_func_info['L_safe']


    def update_delaunay_triangulation(self):
        """
        Outline

        Generate the Delaunay triangulation based on the current evaluated data points xE together with the support points
        xU

        The update of Delaunay triangulation is fixed at the beginning of each optimization iteration
        ----------
        Parameters

        :param self:   SafeDogs class object;

        ----------
        Output

        :return sdogs.DT:
        :return sdogs.tri:
        """
        # Update the total data points, evaluated data xE and support points xU, find their union whose points are unique
        self.xi = Utils.unique_data(np.hstack((self.xU, self.xE)))
        # Construct the Denlaunay triangulation
        if self.n == 1:
            sx = sorted(range(self.xi.shape[1]), key=lambda x: self.xi[:, x])
            tri = np.zeros((self.xi.shape[1] - 1, 2))
            tri[:, 0] = sx[:self.xi.shape[1] - 1]
            tri[:, 1] = sx[1:]
            self.tri = tri.astype(np.int32)
            self.DT = np.copy(tri)

        else:
            options = 'Qt Qbb Qc' if self.n <= 3 else 'Qt Qbb Qc Qx'
            self.DT = Delaunay(self.xi.T, qhull_options=options)
            tri = self.DT.simplices
            keep = np.ones(len(tri), dtype=bool)
            for i, t in enumerate(tri):
                if abs(np.linalg.det(np.hstack((self.xi.T[t], np.ones([1, self.n + 1]).T)))) < 1E-15:
                    keep[i] = False  # Point is coplanar, we don't want to keep it
            self.tri = np.copy(tri[keep])

    def update_exterior_uncertainty(self):
        self.Rmax, self.max_dis = extr_uncer.max_circumradius_delaunay_simplex(self)
        # Determine the parameters b and c for exterior uncertainty function.
        self.b, self.c, status = extr_uncer.exterior_uncertainty_parameter_solver(self.Rmax, self.max_dis)

    def uncertainty_eval(self, query):
        """
        Calculate the uncertainty function value at the query point

        :param query:  n-by-1 2D np.ndarray; Query point
        :return e   :  1-by-, 1D np.ndarray; Uncertainty function value at query point
        """
        assert query.shape[0] == self.n and len(query.shape) == 2, "Query should be n-by-(*) array. "
        if self.n > 1:
            query = query.reshape(-1, 1)
            simplex_index = self.DT.find_simplex(query.T)
            DT_vertices_index = self.DT.simplices[simplex_index]
            simplex = self.xi[:, DT_vertices_index[0]]
            # At the first few iterations, e is number but later on e becomes array
            # Both uncertainty function calculates e(x) as a (1,) array, here only returns a number instead of array

        else:
            simplex = self.xi[:, np.argsort(np.linalg.norm(query - self.xi, axis=0))[:2]]

        unevaluated_exist, evaluated_indices = extr_uncer.unevaluated_vertices_identification(simplex, self.xE)
        if unevaluated_exist:
            e = extr_uncer.exterior_uncertainty(query, self)[0]
        else:
            R2, xc = Utils.circhyp(simplex, self.n)
            e = (R2 - np.linalg.norm(query - xc) ** 2)
        return e

    def surrogate_eval(self, query):
        """
        Outline
        Evaluate the constant K surrogate model values at given query points

        ----------
        Parameters

        :param query:    n-by-N, 2d np.ndarray; n-dimension of input, N-number of data point;

        ----------
        Output
        :return s:      N-by-, 1d np.ndarray;   Stores the surrogate values for each of query point;

        """
        # The difficulty in vectorizing this function lies in NPS interpolation
        assert query.shape[0] == self.n and len(query.shape) == 2, "Query should be n-by-(*) array. "
        N = query.shape[1]
        s = np.zeros(N)
        for ii in range(N):
            x = query[:, ii].reshape(-1, 1)
            s[ii] = self.inter_par.inter_val(x)
            s[ii] -= self.K * self.uncertainty_eval(x)
        return s

    # TODO get the safe radius, with the naive CDC version and vectorized new version
    def get_safe_radius_CDC(self, x):
        """
        Outline

        Compute the (estimated) safe radius (r_hat) r for query points x.
        If x is evaluated, r = yS(x) / sdogs.L_safe,
        if x has not been evaluated, r_hat = sdogs.safe_inter_par.SafeInter_eval(x) / sdogs.L_safe

        ----------
        Parameters

        :param x:   n-by-(*), 2D np.ndarray; The query point

        ----------
        Outputs

        :return safe_radius:
        """
        assert x.shape[0] == self.n and len(x.shape) == 2, "Query x should be n-by-(*) array"
        N = x.shape[1]
        r = np.zeros(N)
        # TODO vectorize r, by adding another argument y in the function input, so that you dont need if statement below
        for ii in range(N):
            query = x[:, ii].reshape(-1, 1)
            val, idx, x_nn = Utils.mindis(query, self.xE)
            if val < 1e-10:
                # query has been evaluated
                r[ii] = np.min(self.yS[:, ii]) / self.L_safe
            else:
                # query has not yet been evaluated
                r[ii] = np.min(self.safe_inter_par.SafeInter_val(query)) / self.L_safe
        return r

    # TODO get the safe estimate at the current x, CDC version and vectorized new version
    def get_safe_estimate_CDC(self, query, x=None, y=None):
        """
        Outline

        Compute the (estimated) safe estimation for query points x, verify x is safe or not.
        More accurately, verify x is in the (estimated) safe ball based on y or not.

        shat = min(y) - L * ||query - x||

        If query is safe but unevaluated, y = sdogs.yS, x = sdogs.xE
        If query is unsafe, verify if it is in the estimated safe ball of x, which is a safe but unevaluated point,
            y is the values of interpolant of psi at x.

        ----------
        Parameters

        :param query:   n-by-(*), 2D np.ndarray; The query point
        :param x    :   n-by-(*), 2D np.ndarray;
        :param y    :   m-by-(*), 2D np.ndarray; The safety function values, or its interpolant, at x.
        ----------
        Outputs

        :return safe_radius:
        """
        assert query.shape[0] == self.n and len(query.shape) == 2, "Query should be n-by-(*) array"
        if x is None:
            x = self.xE
        if y is None:
            y = self.yS

        # TODO vectorize shat, is this correct?
        shat = np.min(y, axis=0) - self.L_safe * np.linalg.norm(query - x)
        return shat

    # TODO journal version get safe radius and estimate
    # def get_safe_radius_journal(self, x, y):

    # def get_safe_estimate_journal(self, x, y):

    # TODO safe delete the test_func once you believe we dont need this anymore
    # def test_func(self):
    #     fun, self.physical_lb, self.physical_ub, self.y0, self.xmin, self.fname = Utils.test_fun(self.test_func_index, self.n)
    #     self.func_eval = partial(Utils.fun_eval, fun, self.physical_ub, self.physical_ub)
    #     safe_fun, self.x0, self.L_safe, self.M, self.safe_name = Utils.test_safe_fun(self.safe_func_index, self.n)
    #     self.safe_eval = partial(Utils.fun_eval, safe_fun, self.physical_lb, self.physical_ub)
    #     return self.func_eval, self.safe_eval

    # TODO We need another class that defines the optimization process, here SafeDogs is only about the data structure
    # def sdogs_optimizer(self):
    #     '''
    #     Main optimization function of S-DOGS.
    #     :return:
    #     '''
    #     for kk in range(self.mesh_refine):
    #         for k in range(self.iter_max):
    #             self.surrogate_solver(self)
    #             SafeDOGSplot.summary_display(self)
    #             if self.iter_type == 'refine':
    #                 break
    #             else:
    #                 pass
    #     SafeDOGSplot.summary(self)

    # def solver_generator(self, y0, K):
    #     if self.surrogate_type == 'constant':
    #         self.K = K
    #         self.surrogate_solver = constantK_snopt_safelearning.constant_surrogate_solver
    #         self.surrgate_eval = constantK_snopt_safelearning.surrogate_eval
    #     else:
    #         self.y0 = y0
    #         # self.surrogate_solver = adaptiveK_snopt_safelearning.adaptiveK_surrogate_solver
    #         # self.surrgate_eval = adaptiveK_snopt_safelearning.surrogate_eval
    #     return self.surrogate_solver, self.surrgate_eval
    #
    # def folder_path(self):
    #     '''
    #     Determine the folder path
    #     :return:
    #     '''
    #     self.current_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    #     self.plot_folder = self.current_path + "/plot"
    #
    # def plot_initialize(self):
    #     '''
    #     Generate the folder path for saving plots.
    #     :return:
    #     '''
    #     if os.path.exists(self.plot_folder):
    #         # if the folder already exists, delete that folder to restart the experiments.
    #         shutil.rmtree(self.plot_folder)
    #     os.makedirs(self.plot_folder)
    #
    # def initial_visual_plot1D(self):
    #     self.plot_func(self)
    #
    # def remove_optimize_file(self):
    #     if self.surrogate_type == 'constant':
    #         file_path = self.current_path + '/opt_info_ck.mat'
    #     else:
    #         file_path = self.current_path + '/opt_info_ak.mat'
    #     os.remove(file_path)


if __name__ == '__main__':
    # TODO rework this
    n = 2
    surrogate_ = 'constant'
    solver_ = 'snopt'
    Nm = 8
    MESH = 4
    Ain = np.concatenate((np.identity(n), -np.identity(n)), axis=0)
    Bin = np.concatenate((np.ones((n, 1)), np.zeros((n, 1))), axis=0)

    # sdogs = SafeDogs(n, test_func_info_, surrogate_, solver_, Nm, MESH, Ain, Bin)
    # sdogs.sdogs_optimizer()
    # while sdogs.iter < 40:
    #     sdogs.surrogate_solver(sdogs)
