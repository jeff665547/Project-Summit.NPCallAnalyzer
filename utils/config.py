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
    chip_marker_num.update({"x", jchip["shooting_marker"]['position_cl']['col']})
    chip_marker_num.update({"y", jchip["shooting_marker"]['position_cl']['row']})
    return chip_marker_num

def get_chip_