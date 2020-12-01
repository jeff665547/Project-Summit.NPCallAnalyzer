import utils.path as sipath
import json

def get_jarucodb():
    with open(sipath.arucodb_jfile) as jfile:
        jfile = json.load(jfile)
        jmarker_dict = jfile['DICT_6X6_250']['bitmap_list']
    return jmarker_dict

def get_jchip(chipname: str):
    with open(sipath.chip_jfile) as jfile:
        jfile = json.load(jfile)
        for i in range(len(jfile)):
            if(jfile[i]['name'] == chipname):
                jchip = jfile[i]
    return jchip

def get_chip_arucoid(jchip: dict):
    return jchip["aruco_marker"]['id_map']

def get_chip_marker_num(jchip: dict):
    chip_marker_num = dict()
    chip_marker_num.update({"r": jchip["shooting_marker"]['position_cl']['row']})
    chip_marker_num.update({"c": jchip["shooting_marker"]['position_cl']['col']})
    return chip_marker_num

def get_chip_probe_num(jchip: dict):
    chip_probe_num = dict()
    chip_probe_num.update({"r": jchip["h_cl"]})
    chip_probe_num.update({"c": jchip["w_cl"]})
    return chip_probe_num

def get_marker_size_probeunit(jchip: dict):
    marker_size = dict()
    marker_size.update({"r": jchip["shooting_marker"]['position_cl']["h"]})
    marker_size.update({"c": jchip["shooting_marker"]['position_cl']["w"]})
    return marker_size

def get_marker_stride_probeunit(jchip: dict):
    marker_stride = dict()
    marker_stride.update({"r": jchip["shooting_marker"]['position_cl']["h_d"]})
    marker_stride.update({"c": jchip["shooting_marker"]['position_cl']["w_d"]})
    return marker_stride