# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 17:48:51 2019

@author: Chao-Hsi Lee
"""

from pathlib import Path
import matplotlib.pyplot as pt
import numpy as np
import pandas as pd

KEYS     = ['x', 'y']
USECOLS1 = ['probe_id', 'probe_seq_5p', 'ref', 'x', 'y']
USECOLS2 = ['x', 'y', 'mean']
CSS_AT   = {'color': 'green', 'alpha': 0.2 }
CSS_CG   = {'color': 'red'  , 'alpha': 0.2 }
CHANNELS = ['mean_g', 'mean_r']
LABEL_AT = 'AT'
LABEL_CG = 'CG'

annot_path = r'C:\Git\data\Cent_M015_C88_annot.tab'

annot = pd.read_csv(str(annot_path), usecols = USECOLS1)
annot.y = 495 - annot.y
annot = annot.loc[annot.probe_id.str.contains('CEN-NP')]
annot.loc[(annot.ref == 'A') | (annot.ref == 'T'), 'label'] = 'AT'
annot.loc[(annot.ref == 'C') | (annot.ref == 'G'), 'label'] = 'CG'


import_dirs = [
    r'C:\Git\data\20190402_1NIF880-23-08\46_20190402133503',
    r'C:\Git\data\20190402_1NIF880-23-08\51_20190402133823',
    r'C:\Git\data\20190402_1NIF880-23-08\61_20190402134142',
    r'C:\Git\data\20190402_1NIF880-23-08\195_20190402135139',
    r'C:\Git\data\20190402_1NIF880-23-08\349_20190402140818',
]

df = []
usecols = ['x', 'y', 'mean_g', 'mean_r', 'decision']
for import_dir in map(Path, import_dirs):
    path = import_dir / 'npcall' / 'ps.csv'
    print('load', str(path))
    df.append(pd.read_csv(str(path), usecols = usecols))
df = pd.concat(df)

#%%

N = len(import_dirs)
yrng = np.unique(annot['y'].values)
cnts = N - df.groupby(['x','y'])['decision'].sum()
matx = np.ones((496,496), int) * -1
for (x, y), c in cnts.iteritems():
    matx[y, x] = c
    

#%%

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# https://plot.ly/ipython-notebooks/color-scales/

fig = make_subplots(
    rows = 3,
    cols = 1,
    vertical_spacing = 0.05,
)
for row, rng in enumerate([ yrng[:13], yrng[13:26], yrng[26:] ], 1):
    ymin, ymax = min(rng), max(rng) + 1
    fig.add_trace(
        go.Heatmap(
            z = matx[ymin:ymax, :],
            x = np.arange(496),
            y = np.arange(ymin, ymax),
            text = np.array([ 'a{}'.format(i) for i in range(13 * 496) ])
        ),
        row = row,
        col = 1,
    )
    fig.update_layout(**{
        'xaxis{}'.format(row): dict(side = "bottom"),
        'yaxis{}'.format(row): dict(autorange = 'reversed')
    })
fig.update_layout(
    autosize = False,
    width    = 500 * 10,
    height   = 700,
)
fig.write_html(
    'failure_map.html',
    auto_open = True,
)




    
    