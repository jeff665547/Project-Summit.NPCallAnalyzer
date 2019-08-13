# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 11:58:42 2019

@author: Chao-Hsi Lee
"""
import pandas as pd
from pathlib import Path


data1_path = Path(r'C:\Git\data\20190402_1NIF880-23-08\46_20190402133503\grid\channels\CY3\heatmap.csv')
data2_path = Path(r'C:\Git\data\20190402_1NIF880-23-08\46_20190402133503\grid\channels\CY5\heatmap.csv')
annot_path = Path(r'C:\Git\data\Cent_M015_C88_annot.tab')

data1 = pd.read_csv(str(data1_path), usecols = ['x', 'y', 'mean'])
data2 = pd.read_csv(str(data2_path), usecols = ['x', 'y', 'mean'])
annot = pd.read_csv(str(annot_path), usecols = ['probe_id', 'x', 'y', 'ref', 'alt'])
annot.y = 495 - annot.y

df = pd.merge(data1, data2, on = ['x', 'y'], suffixes = ('_g', '_r'))
df = pd.merge(annot, df   , on = ['x', 'y'], how = 'outer')
del data1, data2

#%%

np_probes = df.probe_id.str.contains('CEN-NP')
df.loc[np_probes & ((df.ref == 'A') | (df.ref == 'T')), 'type'] = 'AT'
df.loc[np_probes & ((df.ref == 'C') | (df.ref == 'G')), 'type'] = 'CG'



