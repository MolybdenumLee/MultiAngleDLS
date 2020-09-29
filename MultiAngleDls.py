# -*- coding: utf-8 -*-

import os
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import PyMieScatt as ps

import DlsDataParser
from SolveDiameterDistribution import DiaDistResult


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
4. 目前使用后缀 _g1 的方法效果暂时是最好的，用的也是NNLS方法
'''


'''
### 所用缩写说明 ###
d = diameter
num = number
Int = intensity
'''


class multiAngleDls:

    def __init__(self, filelist, filetype='brookhaven dat file', d_min=1, d_max=1e3, d_num=20, log_d=False, auto_process=True):
        # d_min, d_max in nanometer

        # read the multi-angle DLS data from a series of dat files
        self.dlsDataList = []
        for file in filelist:
            self.dlsDataList.append(DlsDataParser.DlsData(filename=file))
        self.angleNum = len(self.dlsDataList)
        # generate diameter array
        self.log_d = log_d
        if log_d:
            log_d_min = np.log10(d_min)
            log_d_max = np.log10(d_max)
            self.d = np.logspace(log_d_min, log_d_max, num=d_num, base=10)
        else:
            self.d = np.linspace(d_min, d_max, num=d_num)
        
        if auto_process:
            self._multiAngleProcess()

    def plotOriginalData(self, plot='g1square'):
        plt.style.use('seaborn')
        fig = plt.figure(figsize=(12,4))
        ax = fig.add_subplot(111)
        for i in range(len(self.dlsDataList)):
            dlsData = self.dlsDataList[i]
            tau = dlsData.tau
            g1 = dlsData.g1
            g1square = dlsData.g1square
            if plot == 'g1':
                ax.plot(tau, g1, '.', label='{}° exp'.format(int(dlsData.angle)))
            elif plot == 'g1square':
                ax.plot(tau, g1square, '.', label='{}° exp'.format(int(dlsData.angle)))
                ax.set_title('$|g^{(1)}|^2$')
        ax.axhline(y=0, color='black')
        ax.legend()
        ax.set_xscale('log')
        ax.set_xlabel(r'$\tau \; (ms)$')
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

     
    def _singleAngleProcess(self, dlsData, Int1):
        kb = 1.38064852e-23                     # Boltzmann Constant

        d = self.d * 1e-9                       # in meter
        theta = dlsData.angle / 180 * np.pi     # rad
        Lambda = dlsData.wavelength * 1e-9      # in meter
        T = dlsData.temperature                 # in Kelvin
        eta = dlsData.viscosity * 1e-3          # viscosity, in Pa.s
        n =dlsData.RI_liquid
        Int = dlsData.intensity                 # intensity, should be cps

        g1 = dlsData.g1
        g1square = dlsData.g1square

        g1 = g1.reshape((g1.size, 1))                   # shape=(m, 1)
        g1square = g1square.reshape((g1square.size, 1)) # shape=(m, 1)
        tau = dlsData.tau * 1e-6                        # delay time, in second
        tau = tau.reshape((len(tau), 1))                # shape=(m, 1)

        g1_theta = g1
        g1square_theta = g1square

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
        C_theta = C_theta.reshape((C_theta.size, 1)) 
        
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

        k_star_theta = Int1 / Int

        G_theta = k_star_theta * F_theta
        
        return g1_theta, g1square_theta, G_theta, F_theta, C_theta, k_star_theta

    def _multiAngleProcess(self):
        g1_theta_list = []
        g1square_theta_list = []
        G_theta_list = []
        F_theta_list = []
        C_theta_list = []
        k_star_theta_list = []
        Int1 = self.dlsDataList[0].intensity
        for dlsData in self.dlsDataList:
            g1_theta, g1square_theta, G_theta, F_theta, C_theta, k_star_theta = self._singleAngleProcess(dlsData, Int1)
            g1_theta_list.append(g1_theta)
            g1square_theta_list.append(g1square_theta)
            G_theta_list.append(G_theta)
            F_theta_list.append(F_theta)
            C_theta_list.append(C_theta)
            k_star_theta_list.append(k_star_theta)

        g1_R = np.vstack(g1_theta_list)
        G_R = np.vstack(G_theta_list)
        
        self.g1_theta_list = g1_theta_list
        self.g1square_theta_list = g1square_theta_list
        self.G_theta_list = G_theta_list
        self.F_theta_list = F_theta_list
        self.C_theta_list = C_theta_list
        self.k_star_theta_list = k_star_theta_list

        self.g1_R = g1_R
        self.G_R = G_R

        return g1_R



    def solveDiaDist(self, method='BayesianInference', mcmc_method='NUTS', *args, **kwargs):
        if method == 'NNLS':
            self.result = DiaDistResult(self, method='NNLS')
            self.N = self.result.nnls_result[0]
            return self.result.nnls_result
        elif method == 'BayesianInference':
            self.result = DiaDistResult(self, method='BayesianInference', mcmc_method='NUTS', *args, **kwargs)
            return 1


    def plotResult_g1star(self):
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
                # TEST
                # 用sumCN来归一化能够保证归一化的效果，但是其实实际拟合中并不是这么归一化的
                # 实际中是使用的intensity归一化的
                #sumCN = np.sum(self.C_list[i]*self.N_star)
                #g1_fit = g1_star / sumCN
                # END TEST

                # 实际使用以下语句
                g1_fit = g1_star / dlsData.intensity
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


    def plotResult_g1square(self):
        fig = plt.figure(figsize=(12,4))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        N = self.result.N
        for i in range(self.angleNum):
            dlsData = self.dlsDataList[i]
            tau = dlsData.tau
            g1square = dlsData.g1square
            ax1.plot(tau, g1square, '.', label='{}° exp'.format(int(dlsData.angle)))
        for i in range(self.angleNum):
            k = 1 / np.sum(self.C_theta_list[i]*N)
            g1square_fit = (k * np.matmul(self.F_theta_list[i], N))**2
            ax1.plot(tau, g1square_fit, 'k-')
            

        ax1.set_xscale('log')
        ax1.legend()

        ax2.plot(self.d, N)
        if self.log_d:
            ax2.set_xscale('log')
        #ax2.set_yscale('log')

        plt.show()


    def plotResult_g1(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        N = self.N.reshape((self.N.size, 1))
        for i in range(self.angleNum):
            dlsData = self.dlsDataList[i]
            tau = dlsData.tau
            g1 = dlsData.g1
            ax1.plot(tau, g1, '.', label='{}° exp'.format(int(dlsData.angle)))

            g1_fit = np.matmul(self.G_theta_list[i], N)
            # TEST
            #g1_fit = g1_fit / g1_fit[0]
            # END TEST 

            ax1.plot(tau, g1_fit, 'k-')

        ax1.set_xscale('log')
        ax1.legend()

        ax2.plot(self.d, N)
        if self.log_d:
            ax2.set_xscale('log')
        ax2.set_yscale('log')

        plt.show()



if __name__ == "__main__":
    filelist = ['test_data/PS_80-200-300nm=2-1-1_{}.dat'.format(i+1) for i in range(6,13)]
    #filelist = ['test_data/PS_100nm_90degree.dat']
    data = multiAngleDls(filelist, d_min=10, d_max=500)
    #data.d = np.array([100, 220, 360])
    #data.plotOriginalData()
    data.solveDiaDist(method='BayesianInference')
    #data.undeterminedMethod()

    # 实际的粒径分布情况
    #data.N = 3 * np.array([50, 3, 1])
    #data.N_star = 18000*np.array([105, 3, 1])
    #data.plotResult_g1()




    


