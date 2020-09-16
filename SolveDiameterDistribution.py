# -*- coding: utf-8 -*-

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

import pymc3 as pm
import theano.tensor as tt
import arviz as az

class DiaDistResult:

    def __init__(self, MdlsData, method='NNLS'):
        '''
        Mdls means multi-angle DLS
        MdlsData is a instance of multiAngleDls object in MultiAngleDls.py
        
        supported methods are:
        NNLS, BayesianInference
        '''

        self.data = MdlsData
        self.method = method
        self.d = self.data.d

        if method == 'NNLS':
            self.solveNnls()
        elif method == 'BayesianInference':
            self.solveBayesianInference()

    
    def solveNnls(self):
        data = self.data
        g1_R, G_R = data.g1_R, data.G_R
        g1_R = g1_R.reshape(g1_R.size)  # array shape for nnls method
        N, rnorm = optimize.nnls(G_R, g1_R, maxiter=30*G_R.shape[1])
        self.nnls_result = (N, rnorm)
        print('NNLS calculation ended.')
        return N, rnorm

    def solveBayesianInference(self):
        data = self.data
        
        # 3 numbers that may be used directly
        n = data.d.size                       # d number
        R = data.angleNum                     # angle number

        g1square_theta_list = data.g1square_theta_list

        # second derivative operator matrix
        # for prior use
        L2 = np.zeros((n, n))
        for j in range(L2.shape[0]-2):
            L2[j, j], L2[j, j+1], L2[j, j+2] = 1, -2, 1

        def prior(N):
            # N.shape = (n, 1)

            #N = N.reshape((N.size, 1))
            # 这里可能需要使用tt里的矩阵乘法
            # 目前还未解决分段函数的问题
            temp = tt.sum(tt.dot(L2, N)**2)
            return tt.exp(-1*temp)


        # for likelihood use
        C_theta_list = data.C_theta_list
        F_theta_list = data.F_theta_list

        def likelihood(N):
            # N.shape = (n, 1)

            def _logp(g1square_exp_list):
                output = 1
                for r in range(R):
                    #N = N.reshape((N.size, 1))
                    g1square_exp = g1square_exp_list[r]
                    M = g1square_exp.size                # tau number
                    C = C_theta_list[r]
                    F = F_theta_list[r]
                    #C = C.reshape((C.size, 1))
                    CN = C*N
                    g1square_theo = ( (1/CN.sum())*tt.dot(F, N) )**2
                    chi = tt.sum( (g1square_exp - g1square_theo)**2 )
                    output = output * tt.power(chi, -1*M/2)
                return output

            return _logp

        model = pm.Model()
        with model:
            # prior distribution
            N = pm.DensityDist('N', prior, shape=(n,1))
            # likelihood function
            like = pm.DensityDist('like', likelihood(N), observed=g1square_theta_list)

        # beggin MCMC
        with model:
            step = pm.Metropolis()
            trace = pm.sample(draws=1000, step=step, cores=1, chains=4, tune=10000, discard_tuned_samples=True)
        
        #az.summary(trace)
        print(trace['N'].shape)
        Nd = np.sum(trace['N'], axis=0)
        plt.plot(data.d, Nd)
        plt.show()
