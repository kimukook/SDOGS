"""
Schwefel test function script
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


class Schwefel(TestFunc):
    def __init__(self, params):
        super().__init__(params)
        self.xmin = .8419 * np.ones((self.n, 1))    # within [-1, 1] of each decision variable
        self.y0 = -1.6759316 * self.n               # within [-1, 1] of each decision variable
        self.func_name = 'schwefel'

    def fun(self, x):

        # transform x into the physical bound
        x_phy = self.get_physical_data(x)

        # make sure that the following line could work for multiple columnwise input data
        y = -sum(np.multiply(500 * x_phy, np.sin(np.sqrt(abs(500 * x_phy))))) / 250
        return y
