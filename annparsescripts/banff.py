import json
import numpy as np
import pandas as pd
import utils.path as sipath
import utils.config as siconfig

AM_cell_num = 28

jchip = siconfig.get_jchip("banff")
marker_row_num = siconfig.get_chip_marker_num(jchip)["x"]
marker_col_num = siconfig.get_chip_marker_num(jchip)["y"]

def polyt_chrome_ann():
        
    banff_chip = np.ones((496, 496), dtype = int) - 2
    
    Edge = ([2]*10)
    Edge.extend([2, 0, 0, 0, 0, 0, 0, 0, 0, 2]*8)
    Edge.extend([2]*10)
    # i = '47'
    for i in list(banff_marker_start_pos.keys()):
        c = (np.array(banff_marker_start_pos[i]) * 81)[0] + 2 #col
        r = (np.array(banff_marker_start_pos[i]) * 81)[1] + 2 #row
        M = np.array(marker_detail[i]).reshape(6, 6) + 1 #marker
        E = np.array(Edge).reshape(10, 10)
        banff_chip[r:r+M.shape[0], c:c+M.shape[1]] += M
        c -= 2; r -= 2
        banff_chip[r:r+E.shape[0], c:c+E.shape[1]] += E
    
    yz01_chip = np.ones((1424, 1424), dtype = int) - 2    

    # i = '47'
    for i in list(yz01_marker_start_pos.keys()):
        c = (np.array(yz01_marker_start_pos[i]) * 101)[0] + 2 #col
        r = (np.array(yz01_marker_start_pos[i]) * 101)[1] + 2 #row
        M = np.array(marker_detail[i]).reshape(6, 6) + 1 #marker
        E = np.array(Edge).reshape(10, 10)
        yz01_chip[r:r+M.shape[0], c:c+M.shape[1]] += M
        c -= 2; r -= 2
        yz01_chip[r:r+E.shape[0], c:c+E.shape[1]] += E
        
        
    banff_bg_ann = pd.DataFrame()
    
    x = list(np.where(banff_chip == 1)[1])
    y = list(np.where(banff_chip == 1)[0])
    ref = ['Chrome']*len(np.where(banff_chip == 1)[1])
    x.extend(list(np.where(banff_chip == 0)[1]))
    y.extend(list(np.where(banff_chip == 0)[0]))
    ref.extend(['PolyT']*len(np.where(banff_chip == 0)[1]))
    
    banff_bg_ann['x'] = x
    banff_bg_ann['y'] = y
    banff_bg_ann['ref'] = ref
    del banff_chip
    
    yz01_bg_ann = pd.DataFrame()
    
    x = list(np.where(yz01_chip == 1)[1])
    y = list(np.where(yz01_chip == 1)[0])
    ref = ['Chrome']*len(np.where(yz01_chip == 1)[1])
    x.extend(list(np.where(yz01_chip == 0)[1]))
    y.extend(list(np.where(yz01_chip == 0)[0]))
    ref.extend(['PolyT']*len(np.where(yz01_chip == 0)[1]))
    
    yz01_bg_ann['x'] = x
    yz01_bg_ann['y'] = y
    yz01_bg_ann['ref'] = ref
    del yz01_chip
    
    bg_ann = dict()
    bg_ann["banff"] = banff_bg_ann
    bg_ann["yz01"] = yz01_bg_ann
    
    return bg_ann

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