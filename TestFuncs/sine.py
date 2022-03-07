"""
sine test function script
=====================================
Author  :  Muhan Zhao
Date    :  Feb. 24, 2022
Location:  UC San Diego, La Jolla, CA
=====================================
"""

# I have tested for Schwefel, np.array([[1, 2]]) / np.array([[1], [2]]) / np.array([1, 2]) all work with the function
# evaluation
from testFunc import TestFunc
import numpy as np


class Sine(TestFunc):
    def __init__(self, params):
        super().__init__(params)
        self.xmin = np.ones((self.n, 1))                        # x = 0 / 1
        self.y0 = -0.382683 * np.ones((self.n, 1))              # y0 = -0.382683
        self.func_name = 'sine'

    def fun(self, x):
        x_phy = self.get_physical_data(x)
        y = np.sin((x_phy - 0.1) * 5 / 4 * np.pi)
        # y is a self.n-by-1 array
        return y

    def get_lipschitz(self):
        """
        The Lipschitz constant characterizes the upper bound of the variation of the
        function over unit change of multi-variables.

        Parameters
        ----------
        :param self: class

        -------

        :return lip_const: m-by-1 vector
        """
        # must be m-by-1 vector
        return 4 * np.ones((self.m, 1))
