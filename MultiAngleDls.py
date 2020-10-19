# -*- coding: utf-8 -*-

import os
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from cycler import cycler # matplotlib内部对象，用于使用内置 colormap 生成循环
import PyMieScatt as ps
import json
import arviz as az

import DlsDataParser
from SolveDiameterDistribution import DiaDistResult


# !!! for this version, we assume that all the data are collected
# under same optical settings and same sample.

'''
尚待解决的问题：
1. 略慢...不过现在发现MCMC不用计算那么多步就能得到还不错的结果了，大概半小时就差不多了。
   目前来看 draws=500, cores=8, chains=8, tune=2000 的设置就能得到比较满意的结果；
   draws=100, step=step, cores=8, chains=8, tune=500 就能够快速的得到一个和最终结果非常接近的结果了，大概计算6分钟左右。
2. 在小直径处总会有很严重的上翘（0-200nm），可以理解，因为小直径粒子在光强上的贡献非常之小，因此数量上看起来很严重的上翘其实光强贡献可以忽略不计。
   因此，对于200nm以上的粒径计算结果非常好，然而有200nm以下粒径的结果就会被这个上翘完全掩盖。也就是说，远离瑞利近似的粒径区间就能够得到比较满意的结果。
   想到的解决方案：
   a. 使用NNLS或者CONTIN计算得到的结果作为MCMC计算（ pm.sampling() ）的start参数.
      结果并没有用
   b. 添加一个限制条件，就是N[0]越小越好，这个限制条件可以通过在prior中使用一个矩阵 E0*N + ||L2*N|| （*为矩阵乘法）代替之前单纯的 ||L2*N|| 实现。
      结果这种实现方式可以使得N[0]趋近于0，但是并不能解决200nm以内结果不准确的问题，以前的单纯上翘变成了一个很大的峰
   c. 最后结果输出的时候不仅提供数量的结果，还提供和CONTIN相同的强度为纵轴的结果(即 Gamma*G(Gamma) )，这样至少按照传统的作图方式看起来就没问题了。


未来计划:
(划掉)1. 能不能使用 NNLS 甚至 CONTIN 得到的结果来作为MCMC的起始值(start参数)
     !(得到的结果并没有什么区别，不过可以作为一个和以往研究的区别，可以考虑加进去)
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
            self.dlsDataList.append(DlsDataParser.DlsData(file))
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
        #plt.style.use('seaborn')
        fig = plt.figure(figsize=(12,4))
        ax = fig.add_subplot(111)
        
        ##### 自定义cycler #####
        # 使用 rainbow colormap
        angleNum = self.angleNum
        custom_cycler = cycler("color", plt.cm.rainbow(np.linspace(0,1,angleNum)))
        ax.set_prop_cycle(custom_cycler)
        #######################

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



    def solveDiaDist(self, method='BayesianInference', alpha=1, beta=1, mcmc_method='NUTS', *args, **kwargs):
        if method == 'NNLS':
            self.result = DiaDistResult(self, method='NNLS')
        elif method == 'BayesianInference':
            self.result = DiaDistResult(self, method='BayesianInference', alpha=1, beta=1, mcmc_method='NUTS', *args, **kwargs)
        
        #### 计算90°或者最接近90°的光强权重的粒径分布，也就是CONTIN得到的那种 ####
        N = self.result.N
        Int = np.zeros_like(N)
        C_theta_list = self.C_theta_list
        angleList = np.array([dlsdata.angle for dlsdata in self.dlsDataList])
        # 得到最接近90°的数据的索引
        angleList = np.abs(angleList - 90)
        i = np.where(angleList == angleList.min())[0][0]

        C = C_theta_list[i]
        Int = C * N
        self.result.intensity_weighted_dist = Int
        ##################################################################

        return self.result

    def plotResult(self, show=True, figname=None, title=None):
        fig = plt.figure(figsize=(12,4))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        N = self.result.N

        ##### 自定义cycler #####
        # 使用 rainbow colormap
        angleNum = self.angleNum
        custom_cycler = cycler("color", plt.cm.rainbow(np.linspace(0,1,angleNum)))
        ax1.set_prop_cycle(custom_cycler)
        #######################

        for i in range(self.angleNum):
            dlsData = self.dlsDataList[i]
            tau = dlsData.tau
            g1square_exp = dlsData.g1square
            ax1.plot(tau, g1square_exp, '.', label='{}° exp'.format(int(dlsData.angle)))

            C = self.C_theta_list[i]
            F = self.F_theta_list[i]
            k = 1 / np.sum(C*N)
            g1square_fit = ( k * np.matmul(F, N) )**2
            ax1.plot(tau, g1square_fit, 'k-')

        ax1.set_title(r'$|g^{(1)}|^2$')
        ax1.set_xlabel(r'$\tau \, (ms)$')
        ax1.set_ylabel(r'$|g^{(1)}|^2$')
        ax1.set_xscale('log')
        ax1.legend(frameon=False)
        
        d = self.d

        # 画 intensity weighted distribution
        ax3 = ax2.twinx()
        ax3.plot(d, self.result.intensity_weighted_dist, 'g--', label='intensity')
        ax3.set_ylabel('Intensity', color='g')

        # 画 number distribution
        ax2.plot(d, N, 'b-', label='number')
        ax2.set_xlabel('d (nm)')
        ax2.set_ylabel('Number', color='b')

        ax2.set_title('Number Distribution / Intensity weighted')

        if title:
            fig.suptitle(title)

        if show:
            plt.show()
        if figname:
            fig.savefig(figname)

        self.result_fig = fig

        return self.result_fig

    def saveResult(self, dirname, summary=True, trace=True, posterior=False, figtitle=None, figformat='svg'):
        os.mkdir(dirname)
        name = os.path.basename(dirname)

        #### gether all the info and data in 1 dictionary ####
        result_dict = {}
        result_dict['d'] = self.d.tolist()
        result_dict['result.N'] = self.result.N.tolist()
        result_dict['result.intensity_weighted_dist'] = self.result.intensity_weighted_dist.tolist()
        result_dict['result.method'] = self.result.method
        result_dict['result.params'] = self.result.params
        result_dict['g1_theta_list'] = [a.tolist() for a in self.g1_theta_list]
        result_dict['g1square_theta_list'] = [a.tolist() for a in self.g1square_theta_list]
        result_dict['G_theta_list'] = [a.tolist() for a in self.G_theta_list]
        result_dict['F_theta_list'] = [a.tolist() for a in self.F_theta_list]
        result_dict['C_theta_list'] = [a.tolist() for a in self.C_theta_list]
        result_dict['k_star_theta_list'] = [a.tolist() for a in self.k_star_theta_list]
        result_dict['g1_R'] = self.g1_R.tolist()
        result_dict['G_R'] = self.G_R.tolist()

        dlsDataList = []
        for dlsData in self.dlsDataList:
            dic = dlsData.__dict__
            for key in dic.keys():
                if type(dic[key]) == np.ndarray:
                    dic[key] = dic[key].tolist()
            dlsDataList.append(dic)
        result_dict['dlsDataList'] = dlsDataList
        ########################################################

        ### save data: json file ###
        filename = name + '.json'
        filepath = os.path.join(dirname, filename)
        jsontext = json.dumps(result_dict, indent=1)
        with open(filepath, 'w') as f:
            f.write(jsontext)

        ### save result plot ###
        filename = name + '.' + figformat
        filepath = os.path.join(dirname, filename)
        if figtitle:
            self.plotResult(show=False, figname=filepath, title=figtitle)

        ### save summary ###
        if summary:
            filename = name + '_summary.csv'
            filepath = os.path.join(dirname, filename)
            df = az.summary(self.result.trace)
            df.to_csv(filepath)

        ### save trace plot ###
        if trace:
            filename = name + '_trace.' + figformat
            filepath = os.path.join(dirname, filename)
            az.plot_trace(self.result.trace)
            plt.savefig(filepath)

        ### save posterior ###
        if posterior:
            filename = name + '_posterior.' + figformat
            filepath = os.path.join(dirname, filename)
            az.plot_posterior(self.result.trace)
            plt.savefig(filepath)
        
        return 'data saved in {}'.format(dirname)

    
    def loadResult(self, filename):
        with open(filename, 'r') as f:
            jsontext = f.read()
        result_dict = json.loads(jsontext)
        '''
        to be continued !!!
        '''
            

if __name__ == "__main__":
    filelist = ['test_data/PS_80-200-300nm=2-1-1_{}.dat'.format(i+1) for i in range(6,13)]
    #filelist = ['test_data/PS_100nm_90degree.dat']
    data = multiAngleDls(filelist, d_min=10, d_max=500)
    #data.d = np.array([100, 220, 360])
    #data.plotOriginalData()
    data.solveDiaDist(method='NNLS')
    data.plotResult(figname='test.png')
    # method='BayesianInference' or 'NNLs'

    # 实际的粒径分布情况
    #data.N = 3 * np.array([50, 3, 1])
    #data.N_star = 18000*np.array([105, 3, 1])
    #data.plotResult_g1()




    


