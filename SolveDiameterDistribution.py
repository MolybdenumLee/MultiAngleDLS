# -*- coding: utf-8 -*-

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

import pymc3 as pm
import theano.tensor as tt
import arviz as az
print("Running on PyMC3 v{}".format(pm.__version__))


class DiaDistResult:

    def __init__(self, MdlsData, method='BayesianInference', mcmc_method='NUTS', auto=True, *args, **kwargs):
        '''
        Mdls means multi-angle DLS
        MdlsData is a instance of multiAngleDls object in MultiAngleDls.py
        
        supported methods are:
        NNLS, BayesianInference
        '''

        self.data = MdlsData
        self.method = method
        self.d = self.data.d

        if auto:
            if method == 'NNLS':
                self.solveNnls()
            elif method == 'BayesianInference':
                self.solveBayesianInference(mcmc_method=mcmc_method, *args, **kwargs)

    
    def solveNnls(self):
        data = self.data
        g1_R, G_R = data.g1_R, data.G_R
        g1_R = g1_R.reshape(g1_R.size)  # array shape for nnls method
        N, rnorm = optimize.nnls(G_R, g1_R, maxiter=30*G_R.shape[1])
        self.N = N
        print('NNLS calculation ended.')
        return N, rnorm

    def solveBayesianInference(self, mcmc_method='NUTS', *args, **kwargs):
        data = self.data
        self.mcmc_method = mcmc_method
        
        # 2 numbers that may be used directly
        n = data.d.size                       # d number
        R = data.angleNum                     # angle number
        # constant list may be used directly
        M = [dlsData.tau.size for dlsData in data.dlsDataList]  # tau or g1 number in each angle

        # second derivative operator matrix
        # for prior use
        L2 = np.zeros((n, n))
        for j in range(L2.shape[0]-2):
            L2[j, j], L2[j, j+1], L2[j, j+2] = 1, -2, 1

        ################## 构建 prior ###################
        # 先验分布，此处使用的是一个指数分布
        # 包含信息：粒径分布必须大于等于0，且尽量光滑
        from pymc3.distributions.continuous import BoundedContinuous
        from pymc3.distributions.dist_math import bound

        class prior(BoundedContinuous):

            def __init__(self, lower=np.zeros((n, 1)), L=L2, *args, **kwargs):
                self.lower = lower = tt.as_tensor_variable(lower)
                self.L = L = tt.as_tensor_variable(L)
                super().__init__(lower=lower,  *args, **kwargs)

            def logp(self, value):
                # 概率分布函数的对数
                # value here is N
                lower = self.lower
                L = self.L
                return bound(-1*tt.sum(tt.dot(L, value)**2), value >= lower)
        ##################################################


        ################# 构建 likelihood #################
        # 似然函数
        from pymc3.distributions import Continuous

        # 生成testval
        tau = data.dlsDataList[0].tau * 1e-6
        Gamma = np.linspace(100, 1000, num=R)
        testg1square_list = [np.exp(-1*gamma*tau)**2 for gamma in Gamma]

        class likelihood(Continuous):

            def __init__(self, g1square_theo_list=testg1square_list, *args, **kwargs):
                self.g1square_theo_list = g1square_theo_list = tt.as_tensor_variable(g1square_theo_list)
                super().__init__(*args, **kwargs)

            def logp(self, value_list):
                # value here is g1square_exp_list
                g1square_theo_list = self.g1square_theo_list

                result = tt.as_tensor_variable(0)
                for r in range(R):
                    result += -M[r]/2 * tt.log(tt.sum((value_list[r] - g1square_theo_list[r])**2))
                return result

        ###################################################

        # 初始化模型
        model = pm.Model()
        with model:
            # prior distribution
            N = prior('N', lower=np.zeros((n, 1)), L=L2, shape=(n,1), testval=np.ones((n, 1)))

            # likelihood function
            g1square_exp_list = data.g1square_theta_list
            g1square_theo_list = []
            F = data.F_theta_list
            C = data.C_theta_list
            for r in range(R):
                g1square_theo_list.append( (1/tt.sum(C[r]*N) * tt.dot(F[r], N))**2 )

            # shape参数没法写，因为考虑到不同角度M可能不相等
            like = likelihood(
                'like', 
                g1square_theo_list=g1square_theo_list,
                testval=testg1square_list, 
                observed=g1square_exp_list
                )

        # beggin MCMC
        with model:
            if mcmc_method == 'NUTS':
                step = pm.NUTS(target_accept=0.95)
            elif mcmc_method == 'HamiltonianMC':
                step = pm.HamiltonianMC()
            elif mcmc_method == 'Slice':
                step = pm.Slice()
            elif mcmc_method == 'Metropolis':
                step = pm.Metropolis()
            else:
                step = pm.NUTS()
            trace = pm.sample(step=step, *args, **kwargs)
            #trace = pm.sample(draws=10000, step=step, cores=8, chains=8, tune=100000, discard_tuned_samples=True)
        
        #az.summary(trace)
        print(trace['N'].shape)
        N_result = np.sum(trace['N'], axis=0) / (trace['N'].shape[0] + 1)
        self.N = N_result
        self.trace = trace
        return model
