import os
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
import plotly.io as pio

#%% Marker Violin Plots
def chip_violin(data, x_classification, y_value, plot_title, filename):
    
    plot_data = []
    for i in range(0, len(pd.unique(x_classification))):
        trace = {
                "type": 'violin',
                "x": x_classification[x_classification == pd.unique(x_classification)[i]],
                "y": y_value[x_classification == pd.unique(x_classification)[i]],
                "name": pd.unique(x_classification)[i],
                "box": {
                    "visible": True
                },
                "meanline": {
                    "visible": True
                }
            }
        plot_data.append(trace)
        
    fig = {
        "data": plot_data,
        "layout" : {
            "title": plot_title,
            "yaxis": {
                "zeroline": False,
            }
        }
    }
        
    plotly.offline.plot(fig, 
                        filename = '{}.html'.format(filename), 
                        validate = False,
                        auto_open = False)

#%% Draw plots for each chip
def each_chip_plots(All, path_to_output, chip_name):
    
    ch2 = All.loc[:, ['x', 'y', 'mean_g', 'ref']]
    if(chip_name == "banff"):
        ch2_matrix = np.zeros((496, 496))
    elif(chip_name == "yz01"):
        ch2_matrix = np.zeros((1424, 1424))
        
    for x, y, z in ch2.loc[:, ['x', 'y', 'mean_g']].values:
        ch2_matrix[int(y)][int(x)] = z
    
    ch4 = All.loc[:, ['x', 'y', 'mean_r', 'ref']]
    if(chip_name == "banff"):
        ch4_matrix = np.zeros((496, 496))
    elif(chip_name == "yz01"):
        ch4_matrix = np.zeros((1424, 1424))
    for x, y, z in ch4.loc[:, ['x', 'y', 'mean_r']].values:
        ch4_matrix[int(y)][int(x)] = z
    
    # Violin plots
    ch2 = ch2[(All.ref == 'Chrome') | (All.ref == 'PolyT') | (All.ref == 'AM1') |(All.ref == 'AM3') | (All.ref == 'AM5')]
    ch4 = ch4[(All.ref == 'Chrome') | (All.ref == 'PolyT') | (All.ref == 'AM1') |(All.ref == 'AM3') | (All.ref == 'AM5')]
    
    chip_violin(data = ch2_matrix, 
                x_classification = ch2['ref'],
                y_value = ch2['mean_g'],
                plot_title = '<b>Violin Plots for green channel Markers</b>',
                filename = str(path_to_output /'marker_violin_green'))
    
    chip_violin(data = ch4_matrix, 
                x_classification = ch4['ref'],
                y_value = ch4['mean_r'],
                plot_title = '<b>Violin Plots for red channel Markers</b>',
                filename = str(path_to_output / 'marker_violin_red'))

#%% Compute statistics for each chip
def each_chip_statistics(All, path_to_output, chip_name):
    
    # marker csv
    marker = All[(All.ref == "AM1") | (All.ref == "AM3") | (All.ref == "PolyT") | (All.ref == "Chrome") | (All.ref == 'AM5')]

    if(chip_name == "banff"):
        ch2_matrix = np.empty((496, 496))
    elif(chip_name == "yz01"):
        ch2_matrix = np.empty((1424, 1424))
    ch2_matrix[:] = np.nan
    if(chip_name == "banff"):
        ch4_matrix = np.empty((496, 496))
    elif(chip_name == "yz01"):
        ch4_matrix = np.empty((1424, 1424))
    ch4_matrix[:] = np.nan
    
    res = marker.groupby(["ref"])['mean_g', 'mean_r'].agg([("Mean", "mean"), 
                                                           ("Std", "std"),
                                                           ("CV", lambda x: x.std()/x.mean())])
    res = res.rename(columns = {"mean_g": "green", "mean_r": "red"})
    res.to_csv(str(path_to_output / "analysis_results.csv"))
    
    
    # for x, y, m2, m4 in marker.loc[:, ['x', 'y', 'mean_g', 'mean_r']].values:
    #    ch2_matrix[int(y)][int(x)] = round(m2, 2)
    #    ch4_matrix[int(y)][int(x)] = round(m4, 2)
    # 
    # np.savetxt(str(path_to_output / "marker_green.csv"), ch2_matrix, fmt = '%.4f', delimiter = ',')
    # np.savetxt(str(path_to_output / "marker_red.csv"), ch4_matrix, fmt = '%.4f', delimiter = ',')

