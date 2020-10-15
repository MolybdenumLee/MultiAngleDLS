# -*- coding: utf-8 -*- 

import numpy as np
import PyMieScatt as ps
import matplotlib.pyplot as plt
import os

import DlsDataParser
import SaveBrookhavenDatFile

'''
### 所用缩写说明 ###
dia = diameter
dist = distribution
num = number
sim = simulation
'''

class DlsDataSim:

    def __init__(
        self, 
        d,
        N,
        save=True,
        filename='Simulated_DLS.dat',
        sampleID='Simulated_DLS',
        operatorID='limu',
        tau_datfile = None,
        tau_min=1, 
        tau_max=1e6, 
        tau_num=200,
        angle=90,                 # degree
        wavelength=633,           # nanometer
        temperature=298,          # Kelvin
        viscosity=0.89,           # cP
        RI_liquid=1.331,
        RI_particle_real=1.5875,
        RI_particle_img=0,
        noise_factor=0.002,       # applied on g1square
        beta=0.53,
        baseline=53364332
    ):
        self.d = d
        self.N = N
        if tau_datfile:
            template = DlsDataParser.DlsData(tau_datfile)
            tau = template.tau                # microsecond
            self.tau = tau.reshape(tau.size)
        else:
            self.tau = np.logspace(np.log10(tau_min), np.log10(tau_max), num=tau_num, base=10) # microsecond
        self.angle = angle # degree
        self.wavelength = wavelength # nanometer
        self.temperature = temperature
        self.viscosity = viscosity # cP
        self.RI_liquid = RI_liquid
        self.RI_particle_real = RI_particle_real
        self.RI_particle_img = RI_particle_img
        self.noise_factor = noise_factor
        self.beta = beta
        self.baseline = baseline

        self.filename = filename
        self.sampleID = sampleID
        self.operatorID = operatorID

        self.genG1()
        self.genG1square()
        self.genG2()

        if save:
            self.saveData()



    def _calcMieScatt(self, angle, m_particle, wavelength, diameter, nMedium, polarization='perpendicular'):
        # angle in radian
        # wavelength in nanometer
        # diameter in nanometer
        # polarization: 'perpendicular', 'parallel', 'unpolarized'

        # DON'T use normalization !
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

    def genG1(self):
        kb = 1.38064852e-23                     # Boltzmann Constant

        d = self.d * 1e-9                       # in meter
        tau = self.tau * 1e-6                   # in second

        theta = self.angle / 180 * np.pi        # in rad
        Lambda = self.wavelength * 1e-9      # in meter
        T = self.temperature
        eta = self.viscosity * 1e-3          # viscosity, in Pa.s
        n =self.RI_liquid

        Gamma0 = (16 * np.pi * n**2 * kb * T)/(3 * Lambda**2 * eta) * np.sin(theta/2)**2
        temp1 = -1 * Gamma0 * tau
        temp2 = 1 / d
        temp1 = temp1.reshape((tau.size, 1))
        temp2 = temp2.reshape((1, d.size))
        Exp = np.exp(np.matmul(temp1, temp2))

        C = [] # intensity from Mie theory
        # the parameters below only used in mie scattering calculations
        d_nano = self.d                   # in nanometer
        angle = theta                     # in rad
        m_particle = complex(self.RI_particle_real, self.RI_particle_img)
        wavelength = self.wavelength   # in nanometer
        for di in d_nano:
            mieInt = self._calcMieScatt(angle, m_particle, wavelength, di, n, polarization='perpendicular')
            C.append(mieInt)
        C = np.array(C)
        CN = C * self.N
        k = 1 / np.sum(CN)
        CN = CN.reshape((CN.size, 1))
        g1 = k * np.matmul(Exp, CN)
        g1 = g1.reshape(g1.size)          # convert g1 to 1d array, same as tau
        self.Exp = Exp
        self.C = C
        self.intensity = np.sum(CN)
        self.g1 = g1
        return self.g1

    def genG1square(self):
        self.g1square = self.g1**2
        self.g1square_with_noise = self.g1square + self.noise_factor * np.random.randn(self.g1.size)
        return self.g1square, self.g1square_with_noise

    def genG2(self):
        beta = self.beta
        B = self.baseline
        self.g2 = B * (1 + beta*self.g1square)
        self.g2_with_noise = B * (1 + beta*self.g1square_with_noise)
        return self.g2, self.g2_with_noise

    def plotG1(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.tau, self.g1, '.')
        ax.set_xscale('log')
        plt.show()

    def saveData(self):
        SaveBrookhavenDatFile.save(self, self.filename, sampleID=self.sampleID, operatorID=self.operatorID)


if __name__ == "__main__":
    #d = np.array([400, 600, 1019])
    #N = np.array([74, 19, 7])
    d = np.array([100])
    N = np.array([1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for angle in range(30, 150, 10):
        data = DlsDataSim(d, N, angle=angle, filename='test_data/simulated data at {} degree.dat'.format(angle), tau_datfile='test_data/PS_80-200-300nm=2-1-1_1.dat')
        data.genG1()
        ax.plot(data.tau, data.g1square_with_noise, '.', label=str(angle)+'degree')
        ax.plot(data.tau, data.g1square, 'k-')
    ax.set_xscale('log')
    ax.legend()
    fig.savefig('test.png')
    #plt.show()