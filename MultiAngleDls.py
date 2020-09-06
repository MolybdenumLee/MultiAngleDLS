# -*- coding: utf-8 -*-

import os
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import PyMieScatt as ps

import DlsDataParser


# !!! for this version, we assume that all the data are collected
# under same optical settings and same sample.

'''
尚待解决的问题：
1. 如果拟合g1，g1在数据大tau处会很接近零，Ctau开根号后导致那块地方数据波动相当剧烈，这样会导致拟合在低tau处的权重下降。
   但是如果考虑拟合g1^2也就大概是Ctau的话，那就会丧失线性关系这一个优点了。
2. 数据点分布其实比较不均匀，使得不同位置的权重差别较大
3. 实际上来说，实验测得的intensity与sumCN有着一定的差别，因此使用intensity来得到不同角度的光散射数据会使得拟合的结果
   并不能完美吻合实验数据，即使使用的是实际上已知的粒径分布。因此后面考虑直接使用sumCN进行归一化，直接拟合g1而不是g1_star.
   不过产生的问题是这样拟合的函数就不再是线性函数了。
'''


'''
### 所用缩写说明 ###
d = diameter
num = number
Int = intensity
'''


class multiAngleDls:

    def __init__(self, filelist, filetype='brookhaven dat file', d_min=1, d_max=1e3, d_num=50, log_d=False):
        # d_min, d_max in nanometer

        # read the multi-angle DLS data from a series of dat files
        self.dlsDataList = []
        for file in filelist:
            self.dlsDataList.append(DlsDataParser.DlsData(filename=file))
        self.angleNum = len(self.dlsDataList)
        # generate diameter array
        self.log_d = log_d
        if log_d:
            self.d = np.logspace(d_min, d_max, num=d_num, base=10)
        else:
            self.d = np.linspace(d_min, d_max, num=d_num)

    def plotOriginalData(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(len(self.dlsDataList)):
            dlsData = self.dlsDataList[i]
            tau = dlsData.tau
            g1 = dlsData.g1
            ax.plot(tau, g1, '.', label='{}° exp'.format(int(dlsData.angle)))
        ax.axhline(y=0, color='black')
        ax.legend()
        ax.set_xscale('log')
        plt.show()

    def _calcMieScatt(self, angle, m_particle, wavelength, diameter, nMedium, polarization='perpendicular'):
        # angle in radian
        # wavelength in nanometer
        # diameter in nanometer
        # polarization: 'perpendicular', 'parallel', 'unpolarized'

        # DON'T use normalization !
        '''
        # this method calculate too much content that we don't need
        # angle in degree
        angles, SL, SR, SU = ps.ScatteringFunction(m_particle, wavelength, diameter, nMedium=nMedium, angularResolution=1, angleMeasure='degree')
        index = round(angle)
        if polarization == 'perpendicular':
            return SL[index]
        elif polarization == 'parallel':
            return SR[index]
        elif polarization == 'unpolarized':
            return SU[index]
        '''
        mu = np.cos(angle)
        S11, S12, S33, S34 = ps.MatrixElements(m_particle, wavelength, diameter, mu, nMedium=nMedium)
        SL = S11 - S12
        SR = S11 + S12
        SU = 0.5 * (SL + SR)
        if polarization == 'perpendicular':
            return SL
        elif polarization == 'parallel':
            return SR
        elif polarization == 'unpolarized':
            return SU



    def _singleAngleProcess(self, dlsData):
        kb = 1.38064852e-23                     # Boltzmann Constant

        d = self.d * 1e-9                       # in meter
        theta = dlsData.angle / 180 * np.pi     # rad
        Lambda = dlsData.wavelength * 1e-9      # in meter
        T = dlsData.temperature                 # in Kelvin
        eta = dlsData.viscosity * 1e-3          # viscosity, in Pa.s
        n =dlsData.RI_liquid
        Int = dlsData.intensity                 # intensity, should be cps

        g1 = dlsData.g1
        g1 = g1.reshape((len(g1), 1))           # shape=(m, 1)
        tau = dlsData.tau * 1e-6                # delay time, in second
        tau = tau.reshape((len(tau), 1))         # shape=(m, 1)

        g1_star_theta = Int * g1

        Gamma0 = (16 * np.pi * n**2 * kb * T)/(3 * Lambda**2 * eta) * np.sin(theta/2)**2

        temp1 = -1 * Gamma0 * tau
        temp2 = 1 / d
        temp1 = temp1.reshape((len(tau), 1))
        temp2 = temp2.reshape((1, len(d)))
        Exp = np.exp(np.matmul(temp1, temp2))

        C_theta = [] # intensity from Mie theory
        # the parameters below only used in mie scattering calculations
        d_nano = self.d                   # in nanometer
        angle = theta                     # in rad
        m_particle = complex(dlsData.RI_particle_real, dlsData.RI_particle_img)
        wavelength = dlsData.wavelength   # in nanometer
        for di in d_nano:
            mieInt = self._calcMieScatt(angle, m_particle, wavelength, di, n, polarization='perpendicular')
            C_theta.append(mieInt)
        C_theta = np.array(C_theta) 
        
        '''
        # 以下使用的方法算出来的F是错误的！
        # 原因应该是python中的坑， 总之不再使用以下方法
        F_theta = Exp
        for i in range(len(C_theta)):
            F_theta[:,i] = F_theta[:,i] * C_theta[i]
        '''
        F_theta = np.ones_like(Exp)
        for j in range(Exp.shape[0]):
            for i in range(Exp.shape[1]):
                F_theta[j,i] = Exp[j,i] * C_theta[i]
        
        return g1_star_theta, F_theta, Exp, C_theta

    def _multiAngleProcess(self):
        g1_star_theta_list = []
        F_theta_list = []
        Exp_list = []
        C_list = []
        for dlsData in self.dlsDataList:
            g1_star_theta, F_theta, Exp, C = self._singleAngleProcess(dlsData)
            g1_star_theta_list.append(g1_star_theta)
            F_theta_list.append(F_theta)
            Exp_list.append(Exp)
            C_list.append(C)
        g1_star_R = np.vstack(g1_star_theta_list)
        F_R = np.vstack(F_theta_list)
        self.g1_star_theta_list = g1_star_theta_list
        self.F_theta_list = F_theta_list
        self.Exp_list = Exp_list
        self.C_list = C_list
        self.g1_star_R = g1_star_R
        self.F_R = F_R
        return g1_star_R, F_R

    def solveNnls(self):
        self._multiAngleProcess()
        g1_star_R = self.g1_star_R.reshape(len(self.g1_star_R))  # array shape for nnls method
        N_star, rnorm = optimize.nnls(self.F_R, g1_star_R, maxiter=None)
        self.N_star = N_star
        self.rnorm_nnls = rnorm
        return N_star, rnorm

    # 这个方法存在很大问题
    def solveDualAnnealing(self):
        def func(N_star, F_R, g1_star_R):
            N_star = N_star.reshape((N_star.size,1))
            r =  np.matmul(F_R, N_star) - g1_star_R
            return np.sum(abs(r))
        
        self._multiAngleProcess()
        bounds = np.zeros((self.d.size, 2))
        bounds[:,1] = np.array([1e9]*self.d.size)
        result = optimize.dual_annealing(func, bounds=bounds, args=(self.F_R, self.g1_star_R))
        self.N_star = result.x


    def plotResult(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        plot_g1_star = False
        if not plot_g1_star:
            for i in range(self.angleNum):
                dlsData = self.dlsDataList[i]
                tau = dlsData.tau
                g1 = dlsData.g1
                ax1.plot(tau, g1, '.', label='{}° exp'.format(int(dlsData.angle)))
                g1_star = np.matmul(self.F_theta_list[i], self.N_star)
                # test
                # 用sumCN来归一化能够保证归一化的效果，但是其实实际拟合中并不是这么归一化的
                # 实际中是使用的intensity归一化的
                sumCN = np.sum(self.C_list[i]*self.N_star)
                g1_fit = g1_star / sumCN
                # 实际使用以下语句
                #g1_fit = g1_fit = g1_star / dlsData.intensity
                ax1.plot(tau, g1_fit, 'k-')
        else:
            for i in range(self.angleNum):
                dlsData = self.dlsDataList[i]
                tau = dlsData.tau
                ax1.plot(tau, self.g1_star_theta_list[i], '.', label='{}° exp'.format(int(dlsData.angle)))
                g1_star = np.matmul(self.F_theta_list[i], self.N_star)
                #g1_fit = g1_star / dlsData.intensity
                g1_fit = g1_star
                ax1.plot(tau, g1_fit, 'k-')

        ax1.set_xscale('log')
        ax1.legend()

        ax2.plot(self.d, self.N_star)
        if self.log_d:
            ax2.set_xscale('log')
        ax2.set_yscale('log')

        plt.show()

    def plotResult2(self):
        fig = plt.figure(figsize=(16,9), dpi=100)
        ax1 = fig.add_subplot(111)
        plot_g1_star = False
        if not plot_g1_star:
            for i in range(self.angleNum):
                dlsData = self.dlsDataList[i]
                tau = dlsData.tau
                g1 = dlsData.g1
                ax1.plot(tau, g1, '.', label='{}° exp'.format(int(dlsData.angle)))
                g1_star = np.matmul(self.F_theta_list[i], self.N_star)
                # test
                # 用sumCN来归一化能够保证归一化的效果，但是其实实际拟合中并不是这么归一化的
                # 实际中是使用的intensity归一化的
                sumCN = np.sum(self.C_list[i]*self.N_star)
                #g1_fit = g1_star / sumCN
                # 实际使用以下语句
                g1_fit = g1_fit = g1_star / dlsData.intensity
                ax1.plot(tau, g1_fit, 'k-')
        else:
            for i in range(self.angleNum):
                dlsData = self.dlsDataList[i]
                tau = dlsData.tau
                ax1.plot(tau, self.g1_star_theta_list[i], '.', label='{}° exp'.format(int(dlsData.angle)))
                g1_star = np.matmul(self.F_theta_list[i], self.N_star)
                #g1_fit = g1_star / dlsData.intensity
                g1_fit = g1_star
                ax1.plot(tau, g1_fit, 'k-')

        ax1.set_xscale('log')
        ax1.legend()

        plt.show()



if __name__ == "__main__":
    filelist = ['test_data/PS_80-200-300nm=2-1-1_{}.dat'.format(i+1) for i in range(6,13)]
    #filelist = ['test_data/PS_100nm_90degree.dat']
    data = multiAngleDls(filelist, d_min=10, d_max=300)
    data.d = np.array([100, 220, 360])
    #data.plotOriginalData()
    data.solveNnls()
    
    #data.solveDualAnnealing()
    #print([a.intensity for a in data.dlsDataList])

    # 实际的粒径分布情况
    data.N_star = 15000*np.array([50, 2, 1])
    #data.N_star = 18000*np.array([105, 3, 1])
    data.plotResult2()


    '''
    data.N_star_nnls = 3000*np.array([50, 2, 1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for dlsData in data.dlsDataList:
        ax.plot(dlsData.tau, dlsData.g1, '.')
    for i in range(len(data.dlsDataList)):
        tau = data.dlsDataList[i].tau
        C = data.C_list[i]
        Exp = data.Exp_list[i]
        F = data.F_theta_list[i]
        N = data.N_star_nnls
        CN = C * N
        k = 1 / np.sum(CN)
        CN = CN.reshape((CN.size, 1))
        N = N.reshape((N.size, 1))
        g1 = k * np.matmul(F, N)
        ax.plot(tau, g1, '-')
    ax.set_xscale('log')
    plt.show()
    '''



    


