# -*- coding: utf-8 -*- 

import numpy as np
from scipy.linalg import lstsq
#
# G2 = B * (1 + beta * g1^2) = B * (1 + Ctau)
# Ctau = beta * g1^2
#



class DlsData:

    def __init__(self, filename=None, filetype='brookhaven dat file', baseline='default', auto=True):
        self.sampleInfo = {
            'SampleID': 'foo',
            'OperatorID': 'foo',
            'Date': 'foo',
            'Time': 'foo'
        }
        self.first_delay = 0 # in microseconds
        self.last_delay = 0 # in microseconds
        self.time_unit = 'microsecond'
        self.angle = 0 # in degrees
        self.wavelength = 0 # in nanometer
        self.temperature = 0 # in Kelvin
        self.viscosity = 0 # in centipoise
        self.RI_liquid = 0
        self.RI_particle_real = 0
        self.RI_particle_img = 0
        self.baseline_calculated = 0
        self.baseline_measured = 0
        self.baseline = 0
        self.intensity = 1

        self.tau = np.array([])
        self.G2 = np.array([])

        self.Ctau = np.array([])
        self.g1 = np.array([])

        if auto:
            if filetype == 'brookhaven dat file':
                self.readBrookhavenRawFile(filename, baseline=baseline)
            self.calcCtau()
            self.calcG1()

    def readBrookhavenRawFile(self, filename, baseline='default'):
        '''
        read params and data from .dat raw file generated by Brookhaven BI-200SM light scattering instrument
        '''
        with open(filename, 'r') as f:
            alldata = list(f.readlines())
        self.sampleInfo = {
            'SampleID': alldata[-4][:-1],
            'OperatorID': alldata[-3][:-1],
            'Date': alldata[-2][:-1],
            'Time': alldata[-1][:-1]
        }
        self.first_delay = float(alldata[5][:-1]) # in microseconds
        self.last_delay = float(alldata[23][:-1]) # in microseconds
        self.time_unit = 'microsecond'
        self.angle = float(alldata[8][:-1]) # in degree
        self.wavelength = float(alldata[9][:-1]) # in nanometer
        self.temperature = float(alldata[10][:-1]) # in degrees Kelvin
        self.viscosity = float(alldata[11][:-1]) # in centipoise
        self.RI_liquid = float(alldata[13][:-1])
        self.RI_particle_real = float(alldata[14][:-1])
        self.RI_particle_img = float(alldata[15][:-1])
        self.baseline_calculated = float(alldata[21][:-1])
        self.baseline_measured = float(alldata[22][:-1])
        if baseline == 'default':
            if int(float(alldata[7][:-1])) == 1:
                self.baseline = self.baseline_calculated
            else:
                self.baseline = self.baseline_measured
        elif baseline == 'calculated':
            self.baseline = self.baseline_calculated
        elif baseline == 'measured':
            self.baseline = self.baseline_measured

        #load raw correlation function (G2)
        tau_list = []
        G2_list = []
        for line in alldata[37:-4]:
            if ',' in line:
                temp = line.strip().split(', ') # for new version of Brookhaven dls software
            else:
                temp = line.split() # for old version of Brookhaven dls software
            tau_list.append(float(temp[0]))
            G2_list.append(float(temp[1]))
        self.tau = np.array(tau_list)
        self.G2 = np.array(G2_list) # intensity auto-correlation function

        self.intensity = np.sqrt(self.baseline)
        # <I> = sqrt( G^{(2)}_\infin )
        # refer to: Vega, J. R.; Gugliotta, L. M.; Gonzalez, V. D. G.; Meira, G. R., Latex particle size distribution by dynamic light scattering: novel data processing for multiangle measurements. J. Colloid Interface Sci. 2003, 261 (1), 74-81.

    def calcCtau(self, baseline='measured'):
        '''
        # C(tau) is the normalized raw correlation function
        # C(tau) = G2(tau)/Baseline - 1 = beta * exp(-2*gamma*tau) = beta * g1(tau)**2
        '''
        B = self.baseline
        self.Ctau = self.G2 / B - 1
        return self.Ctau

    def calcG1(self):
        Ctau = self.Ctau

        # determine beta by fitting first 5 values with Y=beta*exp(-alpha*X)
        # lnY = -alpha*X + ln beta
        X, Y = self.tau[:10], Ctau[:10]
        lnY = np.log(Y)
        A = np.array([[1, xi] for xi in X])
        lnBeta = lstsq(A, lnY)[0][0]
        beta = np.exp(lnBeta)

        self.g1square = self.Ctau / beta
        self.g1 = np.sign(self.Ctau) * np.sqrt(np.abs(self.Ctau / beta))
        return self.g1

    
