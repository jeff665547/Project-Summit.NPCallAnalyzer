import json
import numpy as np
import pandas as pd
import utils.path as sipath
import utils.config as siconfig

AM_cell_num_1marker = 28

chip_name = "yz01"
jchip = siconfig.get_jchip(chip_name)
marker_num_r = siconfig.get_chip_marker_num(jchip)["r"]
marker_num_c = siconfig.get_chip_marker_num(jchip)["c"]
probe_num_r = siconfig.get_chip_probe_num(jchip)["r"]
probe_num_c = siconfig.get_chip_probe_num(jchip)["c"]

marker_size_r_probe = siconfig.get_marker_size_probeunit(jchip)["r"]
marker_size_c_probe = siconfig.get_marker_size_probeunit(jchip)["c"]
marker_stride_r_probe = siconfig.get_marker_stride_probeunit(jchip)["r"]
marker_stride_c_probe = siconfig.get_marker_stride_probeunit(jchip)["c"]

marker_edge = ([2]*marker_size_c_probe)
marker_edge.extend([2, 0, 0, 0, 0, 0, 0, 0, 0, 2]*(marker_size_r_probe - 2))
marker_edge.extend([2]*marker_size_c_probe)

aruco_db = siconfig.get_jarucodb()
yz01_aruco_id_map = siconfig.get_chip_arucoid(jchip)


def polyt_chrome_ann():
        
    yz01_chip = np.ones((probe_num_r, probe_num_c), dtype = int) - 2
    
    # i = '47'
    for i in list(yz01_aruco_id_map.keys()):
        marker_pos_c = (np.array(yz01_aruco_id_map[i]) * marker_stride_c_probe)[0] # col  (horizontal)
        marker_pos_r = (np.array(yz01_aruco_id_map[i]) * marker_stride_r_probe)[1] # row  (vertical)
        code_pos_c = marker_pos_c + 2                                               # ArUco code start col
        code_pos_r = marker_pos_r + 2                                               # ArUco code start row
        code = np.array(aruco_db[i]).reshape(6, 6) + 1                              # ArUco code content
        edge = np.array(marker_edge).reshape(10, 10)                                # Edge of the marker
        
        yz01_chip[code_pos_r:code_pos_r + code.shape[0], code_pos_c:code_pos_c + code.shape[1]] += code
        yz01_chip[marker_pos_r:marker_pos_r + edge.shape[0], marker_pos_c:marker_pos_c + edge.shape[1]] += edge
    
    yz01_ann = pd.DataFrame()
    
    x = list(np.where(yz01_chip == 1)[1])  # horizontal
    y = list(np.where(yz01_chip == 1)[0])  # vertical
    ref = ['Chrome']*len(np.where(yz01_chip == 1)[1])
    x.extend(list(np.where(yz01_chip == 0)[1]))
    y.extend(list(np.where(yz01_chip == 0)[0]))
    ref.extend(['PolyT']*len(np.where(yz01_chip == 0)[1]))
    
    yz01_ann['x'] = x
    yz01_ann['y'] = y
    yz01_ann['ref'] = ref
    del yz01_chip

    return yz01_ann


def AM1_AM3_AM5_ann():
    col = np.zeros(AM_cell_num_1marker*marker_num_r*marker_num_c, int)
    row = np.zeros(AM_cell_num_1marker*marker_num_r*marker_num_c, int)
    ref = list()
    
    ct = 0
    for i in range(probe_num_r):     # i th row (vertical)
        for j in range(probe_num_c): # j th column (horizontal)
            if((i%marker_stride_c_probe) in (1, 8)):           # i th row (vertical)
                if((j%marker_stride_r_probe) in range(1, 9)):  # j th column (horizontal)
                    col[ct] = j
                    row[ct] = i
                    ct += 1
                    if((j%marker_stride_r_probe) in (4, 5)):
                        ref.append('AM5')
                    else:
                        ref.append('AM1')
                    
            if((i%marker_stride_c_probe) in range(2, 8)): # i th row (vertical)
                if((j%marker_stride_r_probe) in (1, 8)):  # j th column (horizontal)
                    col[ct] = j
                    row[ct] = i
                    ct += 1
                    if((i%marker_stride_c_probe) in (4, 5)):
                        ref.append('AM5')
                    else:
                        ref.append('AM1')
    
    yz01_AM_ann = pd.DataFrame()
    yz01_AM_ann['x'] = col # horizontal
    yz01_AM_ann['y'] = row # vertical
    yz01_AM_ann['ref'] = np.array(ref)

    return yz01_AM_ann

def probe_ann():

    chip_shape = siconfig.get_chip_probe_num(jchip)
    chip_shape = chip_shape['r'], chip_shape['c']

    with open(sipath.settings_jfile) as f:
        opts = json.load(f)
        info = opts["annotation"][chip_name]
        columns = ["probe_id", "x", "y"] + list(info['cols'].keys())
        annot = pd.read_csv(info["path"], usecols = columns)
        annot = annot.rename(columns = info["cols"])
        annot.y = chip_shape[1] - annot.y - 1
        return annot

def NP_probe_ann(probe_ann: pd.core.frame.DataFrame):

    annot_np = probe_ann.loc[probe_ann.probe_id.str.contains('CEN-NP')].copy(True)
    annot_np.loc[(annot_np.ref == 'C') | (annot_np.ref == 'G'), 'actual'] = 'CG'
    annot_np.loc[(annot_np.ref == 'A') | (annot_np.ref == 'T'), 'actual'] = 'AT'

    return annot_np
