"""
This is a script that stores the optional parameters needed for the S-DOGS optimization process.
=====================================
Author  :  Muhan Zhao
Date    :  Feb. 24, 2022
Location:  UC San Diego, La Jolla, CA
=====================================


"""
import numpy as np


class OptionsClass:
    """
    Options Class
    """

    def __init__(self):
        self.options = None
        self.solverName = 'None'

    def set_option(self, key, value):
        try:
            if type(value) is self.options[key][2]:
                self.options[key][0] = value
            else:
                print(f"The type of value for the keyword '{key}' should be '{self.options[key][2]}'.")
        except:
            raise ValueError('Incorrect option keyword or type: ' + key)

    def get_option(self, key):
        try:
            value = self.options[key][0]
            return value
        except:
            raise ValueError('Incorrect option keyword: ' + key)

    def reset_options(self, key):
        try:
            self.options[key] = self.options[key][1]
        except:
            raise ValueError('Incorrect option keyword: ' + key)


class SdogsOptions(OptionsClass):
    """
    :param SdogsOptions:    Stores the general parameters for optimization, including mesh info, solver type
                            surrogate type and plotting options

    """
    def __init__(self):
        OptionsClass.__init__(self)
        self.setup()
        self.solver_name = 'S-DOGS'

    def setup(self):
        self.options = {
            # [Current value, default value, type]
            # TODO remove the next 1 lines when you are assertive
            'y0': [None, None, float],

            # TODO i dont think we need this 1 line below, Mar. 2
            'Support points': [None, None, np.ndarray],
            'Algorithm Name': ['SDOGS', 'SDOGS', str],

            'Maximum iteration': [None, 500, int],
            'Convergence threshold': [None, 10e-3, float],

            # TODO in future have the 'journal' option that implements the new estimate
            'Safe radius estimate CDC version': [False, False, bool],
            'Safe radius estimate Journal version': [False, False, bool],

            'Constant surrogate': [False, False, bool],
            'Constant K': [None, None, float],

            'Adaptive surrogate': [False, False, bool],
            'Target value': [None, None, float],

            'Scipy solver': [False, False, bool],
            'Snopt solver': [False, False, bool],

            'Initial mesh size': [8, 8, int],
            'Number of mesh refinement': [8, 8, int],
            'Unit interval grid': [None, None, float],

            'Initial sites': [None, None, np.ndarray],
            'Initial function values': [None, None, np.ndarray],

            # TODO how to incorporate next two lines into test func?
            #  now in SafeDogs I am using info from testFuncs
            'Global minimizer known': [False, False, bool],
            'Global minimizer': [None, None, np.ndarray],

            'Function range known': [False, False, bool],
            'Function range': [None, None, np.ndarray],

            'Function evaluation cheap': [True, True, bool],
            'Function prior file path': [None, None, str],

            'Plot save': [False, False, bool],
            'Plot display': [False, False, bool],
            'Figure format': ['png', 'png', str],

            'Candidate distance summary': [False, False, bool],
            'Candidate objective value summary': [False, False, bool],
            'Iteration summary': [False, False, bool],
            'Optimization summary': [False, False, bool]
        }
