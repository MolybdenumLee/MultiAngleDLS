# -*- coding: utf-8 -*- 

import datetime


# dls is an DlsSimulation object in DlsDataSimulator.py
def save(dls, filename):
    textList = ['' for i in range(37)]

    textList[0]  = '1'    # Run Number 
    textList[1]  = '-1'   # Total Counts, A Input 
    textList[2]  = '-1'   # Total Counts, B Input 
    textList[3]  = '1'    # Number of Samples
    textList[4]  = '0'    # not used
    textList[5]  = dls.tau[0]
    textList[6]  = dls.tau.size
    textList[7]  = 0      # Determines whether the ISDA programs will prompt the user for which baseline to use: 0 = will prompt
    textList[8]  = dls.paramsDict['angle']
    textList[9]  = dls.paramsDict['wavelength']
    textList[10] = dls.paramsDict['temperature']
    textList[11] = dls.paramsDict['viscosity']
    textList[12] = 0      # not used
    textList[13] = dls.paramsDict['RI liquid']
    textList[14] = dls.paramsDict['RI particle real']
    textList[15] = dls.paramsDict['RI particle img']
    textList[16] = 37     # One less than the number of parameters preceding the x and y data in this file. As of the date of this document, the number of parameters is 37. 
    textList[17] = dls.tau[0]
    textList[18] = -2     # Time delay mode: -2 = Constant ratio spacing • -1 = Spacing from delay file • 1 = Linear spacing 
    textList[19] = 2      # Analysis mode: • 2 = Autocorrelation  • 3 = Cross correlation  • 4 = Test 
    textList[20] = 4      # Number of extended baseline channels 
    textList[21] = dls.paramsDict['baseline']
    textList[22] = dls.paramsDict['baseline']
    textList[23] = dls.tau[-1]
    textList[24] = '-1'   # Sampling time used to generate number of samples (µs)
    textList[25] = '-1'   # First delay used from High speed section
    textList[26] = '-1'   # Number of High speed channels used 
    textList[27] = '-1'   # Middle speed sampling time (µs) 
    textList[28] = '-1'   # Number of Middle speed channels 
    textList[29] = '-1'   # Low speed sampling time (µs) 
    textList[30] = '-1'   # Number of Low speed channels 
    textList[31] = '0'    # not used
    textList[32] = '0'    # not used
    textList[33] = '0'    # not used
    textList[34] = '-1'   # First measured baseline channel number 
    textList[35] = '-1'   # Last measured baseline channel number 
    textList[36] = '0'    # not used
    
    for i in range(len(dls.tau)):
        line = '{} {}'.format(str(dls.tau[i]), str(dls.G2_exp[i]))
        textList.append(line)
    
    textList.append(
        dls.paramsDict['sample ID']
    )
    textList.append(
        dls.paramsDict['operator ID']
    )

    today = datetime.date.today()
    date = '{}/{}/{}'.format(str(today.month), str(today.day), str(today.year))
    textList.append(date)

    now = datetime.datetime.now().time()
    time = '{}:{}:{}'.format(str(now.hour), str(now.minute), str(now.second))
    textList.append(time)

    textList = [str(line)+'\n' for line in textList]

    with open(filename, 'w') as f:
        f.writelines(textList)

    return 'done'