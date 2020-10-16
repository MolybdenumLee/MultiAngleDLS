# -*- coding: utf-8 -*-

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

import pymc3 as pm
import theano.tensor as tt
import arviz as az
print("Running on PyMC3 v{}".format(pm.__version__))


class DiaDistResult:

    def __init__(self, MdlsData, method='BayesianInference', alpha=1, beta=1, mcmc_method='NUTS', auto=True, *args, **kwargs):
        '''
        Mdls means multi-angle DLS
        MdlsData is a instance of multiAngleDls object in MultiAngleDls.py
        
        supported methods are:
        NNLS, BayesianInference

        *args, **kwargs transfer to pm.sample() directly
        '''

        self.data = MdlsData
        self.method = method
        self.d = self.data.d
        self.params = {
            'NNLS': {},
            'BayesianInference': {}
        }

        if auto:
            if method == 'NNLS':
                self.solveNnls()
            elif method == 'BayesianInference':
                self.solveBayesianInference(alpha=alpha, beta=beta, mcmc_method=mcmc_method, *args, **kwargs)

    
    def solveNnls(self):
        data = self.data
        g1_R, G_R = data.g1_R, data.G_R
        g1_R = g1_R.reshape(g1_R.size)  # array shape for nnls method
        N, rnorm = optimize.nnls(G_R, g1_R, maxiter=30*G_R.shape[1])
        self.N = N.reshape((N.size, 1))
        print('NNLS calculation ended.')
        return N, rnorm

    def solveBayesianInference(self, alpha=1, beta=1, mcmc_method='NUTS', *args, **kwargs):
        data = self.data
        self.params['BayesianInference']['alpha'] = alpha
        self.params['BayesianInference']['beta'] = beta
        self.params['BayesianInference']['mcmc_method'] = mcmc_method

        # 2 numbers that may be used directly
        n = data.d.size                       # d number
        R = data.angleNum                     # angle number
        
        # second derivative operator matrix
        # for prior use
        L2 = np.zeros((n, n))
        for j in range(L2.shape[0]-2):
            L2[j, j], L2[j, j+1], L2[j, j+2] = 1, -2, 1


        '''
        ### 目前所用的贝叶斯推断各分布的说明 ###
        1. prior包含两个部分，即 p(N) 和 p(sigma**2)
        其中 p(N) 使用一指数分布，并且 N>=0，来包含先验信息即N为非负且尽量光滑（二阶导尽量小）
        而我们认为sigma遵循 Halfnormal 分布，那么可以推导出 sigma**2 服从 Gamma分布
        然后，并不像那篇文章里将sigma做边缘化处理，我们照样把sigma作为参数算出来
        因为sigma的个数和角度的数量是一样的，相比N的数量会少很多，并不会显著增加参数的个数，影响应该不大
        2. likelihood
        直接使用正态分布，g1square_obs 应该服从理论值为中心sigma**2为方差的正态分布
        '''


        ################## 构建 p(N) ###################
        # 先验分布，此处使用的是 指数分布 以及 伽马分布
        # 由于指数分布实际上是伽马分布的特殊情况（alpha=1），因此就只保留一个就够了
        # 包含信息：粒径分布必须大于等于0，且尽量光滑（二阶导尽可能小）
        from pymc3.distributions.continuous import BoundedContinuous, PositiveContinuous
        from pymc3.distributions.dist_math import bound, gammaln, logpow
        from pymc3.theanof import floatX
        '''
        class pN_Exponential(BoundedContinuous):

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
        '''
        class pN_Gamma(PositiveContinuous):

            def __init__(self, alpha=1, beta=1, L=L2, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.alpha = alpha = tt.as_tensor_variable(floatX(alpha))
                self.beta = beta = tt.as_tensor_variable(floatX(beta))
                self.L = L = tt.as_tensor_variable(L)

                #assert_negative_support(alpha, "alpha", "Gamma")
                #assert_negative_support(beta, "beta", "Gamma")

            def logp(self, value):
                # value here is N
                alpha = self.alpha
                beta = self.beta
                L = self.L
                return bound(
                    #-gammaln(alpha) + logpow(beta, alpha) - beta * tt.sum(tt.dot(L, value)**2) + logpow(tt.sum(tt.dot(L, value)**2), alpha - 1),
                    -gammaln(alpha) + alpha*tt.log(beta) - beta * tt.sum(tt.dot(L, value)**2) + (alpha - 1)*tt.log(tt.sum(tt.dot(L, value)**2)),
                    value >= 0,
                    alpha > 0,
                    beta > 0,
                )
        ##################################################

        '''
        废弃的方法
        这里的方法是实现那篇文献里的方法 (DOI: 10.1109/SSP.2014.6884650)
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
        '''

        # 初始化模型
        model = pm.Model()
        with model:
            #### prior distribution ####
            # testval的选择也比较重要，目前来看对于我们选择的分布，需要满足：
            # 不能有0，如果alhpa!=1的情况下不能全为一样的值
            # 否则会报错：bad initial energy
            testval = 1e-5 * np.ones((n, 1))
            testval[int(n/2), 0] = 2 * testval[int(n/2), 0]
            N = pN_Gamma('N', alpha=alpha, beta=beta, L=L2, shape=(n,1), testval=testval)  # 这里testval用np.zeros就会报错 bad initial energy
            
            # assume sigma~N(0, s^2), so s is the std. deviation of sigma distribution
            s = 0.05
            beta1 = 1 / (2*s**2)
            sigma = pm.Gamma('sigma', alpha=0.5, beta=beta1, shape=R)
            ############################

            #### likelihood function ###
            g1square_obs = []
            for r in range(R):
                M = data.g1square_theta_list[r].size
                C = data.C_theta_list[r]
                C = C.reshape((C.size, 1))
                F = data.F_theta_list[r]
                g1square_theo = ( (1/tt.sum(C*N))*tt.dot(F, N) )**2

                #like = likelihood('like', g1square_theo=g1square_theo, M=M, shape=(M,1), testval=testg1square, observed=data.g1square_theta_list[0])
                g1square_obs.append(
                    pm.Normal('g1square_obs_{}'.format(r), mu=g1square_theo, sigma=sigma[r], shape=(M, 1), observed=data.g1square_theta_list[r])
                )
            #############################

        # beggin MCMC
        with model:
            target_accept = 0.95
            if mcmc_method == 'NUTS':
                step = pm.NUTS(target_accept=target_accept)
            elif mcmc_method == 'HamiltonianMC':
                step = pm.HamiltonianMC(target_accept=target_accept)
            elif mcmc_method == 'Slice':
                step = pm.Slice(target_accept=target_accept)
            elif mcmc_method == 'Metropolis':
                step = pm.Metropolis(target_accept=target_accept)
            else:
                step = pm.NUTS(target_accept=target_accept)
            trace = pm.sample(step=step, *args, **kwargs)
            #trace = pm.sample(draws=10000, step=step, cores=8, chains=8, tune=100000, discard_tuned_samples=True)
        
        #az.summary(trace)
        #print(trace['N'].shape)
        N_result = np.sum(trace['N'], axis=0) / (trace['N'].shape[0] + 1)
        self.N = N_result
        self.trace = trace
        return N_result, trace, model
