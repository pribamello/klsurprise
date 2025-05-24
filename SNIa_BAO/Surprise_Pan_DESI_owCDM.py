
import numpy as np
import os
import jax 

import SNIa_likelihood_pantheon as SNIa
import BAO_likelihood_DESI as BAO


# pathdir = '../'
pathdir = '.'
import sys
sys.path.append(pathdir)
import klsurprise as kls

date_id = '20250520'
domain = np.array([[0.3, 1.0], # h
                   [0.05, 1.0], # Om
                   [-1.0, 1.0], # OK
                   [-3.0, -0.4]]) # w

## necessary for KLD MCMC mode
@jax.jit
def logL_1(x):
    # owCDM
    return BAO.logP_BAO_marginalized(x)

def data_2_model_fun(x):
    # owCDM
    return SNIa.model_SNIa(x)

data_2_covariance = SNIa.covariance_pantheon
data_2_vector = SNIa.mu_SNIa

############ define the loglikelihood function used to create mock data from PPD samples ############
@jax.jit
def logL_mock(theta, data):
    return SNIa.logposterior_owCDM(theta, mu_SNIa=data)


sup = kls.surprise_statistics(logL_1, data_2_model_fun, covariance_matrix_2=data_2_covariance,
                               domain=domain, data_2=data_2_vector,
                               data_1_name = "{}_DESI+BBN_owCDM.pkl".format(date_id),
                               data_2_name = "{}_Pantheon+SH0ES_owCDM.pkl".format(date_id),
                               init_NS = True)#, Nppd = 4) # check this parameter...

Nkld = 2000
# Call the main method
resultados = sup.surprise_function_call(Nkld = Nkld, result_path = "{}_Surprise_Pan_DESI.hdf5", n_jobs = 64)