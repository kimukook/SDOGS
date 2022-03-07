"""
This is a script that characterize the general test functions
In the class of SafeDogs, there are lots of calling on test functions with a vector n-by-,
=====================================
Author  :  Muhan Zhao
Date    :  Feb. 24, 2022
Location:  UC San Diego, La Jolla, CA
=====================================


"""
import numpy as np


class TestFunc:
    """
    The base class for test functions

    Parameters
    ----------
    :param params: dict, required

    :param ub       :    n-by-1 vector
    :param lb       :    n-by-1 vector
    :param n        :    dimension of the data
    :param extraArgs:    extra arguments for function evaluation
    :param func_name:    the name of the function handle

    -------
    :return : N/A
    """
    def __init__(self, params):

        # upper bound and lower bound, should be n-by-1 array
        if 'ub' in params.keys():
            self.ub = np.atleast_2d(params['ub'])
        else:
            KeyError("Input params has no attribute upper bound 'ub'.")

        if 'lb' in params.keys():
            self.lb = np.atleast_2d(params['lb'])
        else:
            KeyError("Input params has no attribute upper bound 'ub'.")

        # number of the dimension of the parameter space
        if 'n' in params.keys():
            self.n = params['n']
        else:
            self.n = self.ub.shape[0]

        if 'm' in params.keys():
            self.m = params['m']
        else:
            pass

        if 'extraArgs' in params.keys():
            self.extraArgs = params['extraArgs']
        else:
            pass

        if 'func_name' in params.keys():
            self.func_name = params['func_name']
        else:
            pass

        self.xmin = None
        self.y0 = None

    # def get_normalized_data(self):

    def get_physical_data(self, x):
        """
        Transform the normalized decision variables back to the physical range for function evaluation
        :param x: n-by-1 array, or n-by-m array, columnwise
        :return:
        """
        if x.shape == (1, self.n):
            x_transform = x.T

        elif x.shape == (self.n,):
            x_transform = np.atleast_2d(x).T

        elif x.shape[0] == self.n and x.shape[1] > 1:
            x_transform = np.copy(x)

        else:
            x_transform = np.copy(x)

        if x.shape[0] != self.n:
            ValueError('Each data point should be columnwise.')
        return self.lb + x_transform * (self.ub - self.lb)

