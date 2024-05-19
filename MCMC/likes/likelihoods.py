import numpy as np
from cobaya.likelihood import Likelihood
from cobaya.theory import Theory

#############################################
class Chi2Like(Likelihood):
    data = None # expect 1-d array
    invcov_mat = None # expect 2-d array
    #########################################
    def initialize(self):
        # various checks on input data and inverse covariance matrix
        if self.data is None:
            raise Exception('data must be specified in Chi2Like.')
        if self.invcov_mat is None:
            raise Exception('invcov_mat must be specified in Chi2Like.')
        
        if len(self.data.shape) != 1:
            raise Exception('data should be 1-d array in Chi2Like.')
        if len(self.invcov_mat.shape) != 2:
            raise Exception('invcov_mat should be 2-d array in Chi2Like.')
            
        if self.invcov_mat.shape != (self.data.size,self.data.size):
            raise Exception('Incompatible inverse covariance and data in Chi2Like.')        
    #########################################

    #########################################
    def get_requirements(self):
        """ Theory code should return model array. """
        return {'model': None}
    #########################################

    #########################################
    def logp(self,**params_values_dict):
        model = self.provider.get_model()
        residual = self.data - model
        chi2 = np.dot(residual,np.dot(self.invcov_mat,residual))
        return -0.5*chi2
    #########################################


#############################################
class StraightLineTheory(Theory):
    xvals = None # expect 1-d array of same size as data input to likelihood
    #########################################
    def initialize(self):
        if self.xvals is None:
            raise Exception('xvals must be specified in StraightLine.')
        if len(self.xvals.shape) != 1:
            raise Exception('xvals should be 1-d array in StraightLine.')
    #########################################
    
    #########################################
    def calculate(self,state, want_derived=False, **param_dict):
        # ensure ordering a0,a1
        a0,a1 = param_dict.values()
        model = a0 + a1*self.xvals
        state['model'] = model
    #########################################

    #########################################
    def get_model(self):
        return self.current_state['model']
    #########################################

    #########################################
    def get_allow_agnostic(self):
        return True
    #########################################
