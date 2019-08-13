import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from itertools import product
from pathlib import Path

import plotly.graph_objects as go
from imgproc_utils import auto_contrast

colorscale = {
    'green': [
        [0.0, 'rgb(0,  0,0)'],
        [1.0, 'rgb(0,255,0)'],
    ],
    'red': [
        [0.0, 'rgb(  0,0,0)'],
        [1.0, 'rgb(255,0,0)'],
    ],
    'dual': [
        [(g + r * 256) / 65535, 'rgb({},{},0)'.format(r, g)]
        for r in range(256) for g in [0, 255]
    ]
}

def draw_dual_topography(
    result_path = 'topography.html',
    auto_thres = 5000,
    auto_open = True,
    **traces
):
    global colorscale
    data = []
    for name, values in traces.items():
        vmax = auto_contrast(values, auto_thres)[1]
        levels = np.uint16(np.round(np.maximum(0, np.minimum(1, values / vmax)) * 255))
        data += [
            go.Heatmap(
                z = levels[0] + 256 * levels[1],
                zmin = 0,
                zmax = 65535,
                colorscale = colorscale['dual'],
                showscale  = False,
                visible    = len(data) == 0,
            ),
            go.Heatmap(
                z = values[0],
                zmin = 0,
                zmax = vmax,
                colorscale = colorscale['green'],
                showscale  = False,
                visible    = False,
            ),
            go.Heatmap(
                z = values[1],
                zmin = 0,
                zmax = vmax,
                colorscale = colorscale['red'],
                showscale  = False,
                visible    = False,
            ),
        ]
    fig = go.Figure(
        data = data,
        layout = go.Layout(
            width  = 816,
            height = 800,
            xaxis  = dict(side = "top"),
            yaxis  = dict(autorange = "reversed"),
            updatemenus = [
                go.layout.Updatemenu(
                    type = 'buttons',
                    active = 0,
                    xanchor = 'left',
                    yanchor = 'top',
                    x = 1.02,
                    buttons = [
                        dict(
                            label = '{}/{}'.format(key, channel),
                            method = 'update',
                            args = [{'visible': [i == j for i in range(3 * len(traces))]}],
                        )
                        for j, (key, channel) in enumerate(product(
                            traces.keys(), ['both', 'green', 'red']
                        ))
                    ]
                )
            ]
        )
    )
    fig.write_html(
        str(result_path),
        auto_open = auto_open,
    )
        
    


#def draw_topography(g_values, r_values, result_path, thres = 10000, auto_open = True):
#    global colorscale
#    dual = 'x: {}<br />y: {}<br />z: {}, {}'
#    vmax = max(
#        auto_contrast(g_values, thres)[1],
#        auto_contrast(r_values, thres)[1]
#    )
#    rescaled_g = np.uint16(np.round(np.minimum(1, g_values / vmax) * 255))
#    rescaled_r = np.uint16(np.round(np.minimum(1, r_values / vmax) * 255))
#    
#    fig = go.Figure(data = [
#        go.Heatmap(
#            z = rescaled_g + rescaled_r * 256,
#            zmin = 0,
#            zmax = 65536,
#            text = [
#                [ dual.format(x, y, g, r) for x, (g, r) in enumerate(zip(gs, rs))]
#                for y, (gs, rs) in enumerate(zip(g_values, r_values))
#            ],
#            colorscale = colorscale['dual'],
#            showscale  = False,
#            visible    = True,
#            hoverinfo  = 'text',
#        ),
#        go.Heatmap(
#            z = g_values,
#            zmin = 0,
#            zmax = vmax,
#            colorscale = colorscale['green'],
#            showscale  = False,
#            visible    = False,
#        ),
#        go.Heatmap(
#            z = r_values,
#            zmin = 0,
#            zmax = vmax,
#            colorscale = colorscale['red'],
#            showscale  = False,
#            visible    = False,
#        )
#    ])  
#    fig.update_layout(
#        width  = 800,
#        height = 800,
#        xaxis  = dict(side = "top"),
#        yaxis  = dict(autorange = "reversed"),
#        updatemenus = [
#            go.layout.Updatemenu(
#                type = 'buttons',
#                direction = 'left',
#                active = 0,
#                x = 0.25,
#                y = 1.12,
#                buttons = [
#                    dict(
#                        label = text,
#                        method = 'update',
#                        args = [{'visible': [i == j for i in range(3)]}],
#                    )
#                    for j, text in enumerate(['both', 'green', 'red'])
#                ]
#            )
#        ]
#    )
#    fig.write_html(
#        str(result_path),
#        auto_open = auto_open,
#    )


#%%

shape = (496,496)

samples = [
    '46_20190402133503',
    '51_20190402133823',
    '61_20190402134142',
    '195_20190402135139',
    '349_20190402140818',
]
paths = map(Path,[
    r'C:\Git\data\20190402_1NIF880-23-08\{}\indiv\topography.csv'.format(d)
    for d in samples
])
keys = ['y', 'x']
cols = ['mean_g', 'mean_r']
df = pd.concat([ pd.read_csv(str(path)) for path in paths ])
gp = df.groupby(keys)

#%%

s0 = np.round(gp.median())[cols].values.reshape((496,496,-1)).transpose(2,0,1)
s1 = np.round(gp.mean()  )[cols].values.reshape((496,496,-1)).transpose(2,0,1)
s2 = np.round(gp.std()   )[cols].values.reshape((496,496,-1)).transpose(2,0,1)
draw_dual_topography(
    median = s0,
    mean = s1,
    stdev = s2,
    cv = s2 / s1 * 100
)



#%%

draw_topography(s1[0], s1[1], 'topography_mean.html' , auto_open = True)

#%%
draw_topography(s2[0] / s1[0], s2[1] / s1[1], 'topography_stdev.html', auto_open = True)
