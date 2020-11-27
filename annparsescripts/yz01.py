import numpy as np
import pandas as pd

def AM1_AM3_AM5_ann():
    x = np.zeros(28*7*7, int)
    y = np.zeros(28*7*7, int)
    ref = list()
    
    ct = 0
    for i in range(496): #Y
        for j in range(496): #X
            if((i%81) in (1, 8)):
                if((j%81) in range(1, 9)):
                    x[ct] = j
                    y[ct] = i
                    ct += 1
                    if((j%81) in (4, 5)):
                        ref.append('AM3')
                    else:
                        ref.append('AM1')
                    
            if((i%81) in range(2, 8)):
                if((j%81) in (1, 8)):
                    x[ct] = j
                    y[ct] = i
                    ct += 1
                    if((i%81) in (4, 5)):
                        ref.append('AM3')
                    else:
                        ref.append('AM1')
    
    banff_AM_ann = pd.DataFrame()
    banff_AM_ann['x'] = x
    banff_AM_ann['y'] = y
    banff_AM_ann['ref'] = np.array(ref)
    
    ############################
    
    x = np.zeros(28*15*15, int)
    y = np.zeros(28*15*15, int)
    ref = list()
    
    ct = 0
    for i in range(1424): #Y
        for j in range(1424): #X
            if((i%101) in (1, 8)):
                if((j%101) in range(1, 9)):
                    x[ct] = j
                    y[ct] = i
                    ct += 1
                    if((j%101) in (4, 5)):
                        ref.append('AM5')
                    else:
                        ref.append('AM1')
                    
            if((i%101) in range(2, 8)):
                if((j%101) in (1, 8)):
                    x[ct] = j
                    y[ct] = i
                    ct += 1
                    if((i%101) in (4, 5)):
                        ref.append('AM5')
                    else:
                        ref.append('AM1')
    
    yz01_AM_ann = pd.DataFrame()
    yz01_AM_ann['x'] = x
    yz01_AM_ann['y'] = y
    yz01_AM_ann['ref'] = np.array(ref)
    
    AM_ann = dict()
    AM_ann['banff'] = banff_AM_ann
    AM_ann['yz01'] = yz01_AM_ann
    
    return AM_ann