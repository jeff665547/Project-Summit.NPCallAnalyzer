import os
import json
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
import plotly.io as pio
import sys

#%% Quantile Normalization
def quantile_norm(data):
    temp = data.argsort(axis = 0)
    rank = np.empty_like(temp)
    for i in range(data.shape[1]):
        rank[temp[:,i],i] = np.arange(data.shape[0])
    return np.sort(data, axis = 0).mean(axis = 1)[rank]

#%% Annotation files for marker (PloyT & Chrome)
def polyt_chrome_ann(file_path):
        
    with open('{}/aruco_default_dict_6x6_250.json'.format(file_path)) as inner_marker:
        inner = json.load(inner_marker)
        marker_detail = inner['marker_list']
    
    with open('{}/chip.json'.format(file_path)) as marker_pos:
        chip = json.load(marker_pos)
        for i in range(len(chip)):
            if(chip[i]['name'] == 'banff'):
                marker_start_pos = chip[i]["aruco_marker"]['id_map']
    
    all_chip = np.ones((496, 496), dtype = int) - 2
    
    Edge = ([2]*10)
    Edge.extend([2, 0, 0, 0, 0, 0, 0, 0, 0, 2]*8)
    Edge.extend([2]*10)
    # i = '0'
    for i in list(marker_start_pos.keys()):
        c = (np.array(marker_start_pos[i]) * 81)[0] + 2 #col
        r = (np.array(marker_start_pos[i]) * 81)[1] + 2 #row
        M = np.array(marker_detail[i]).reshape(6, 6) + 1 #marker
        E = np.array(Edge).reshape(10, 10)
        all_chip[r:r+M.shape[0], c:c+M.shape[1]] += M
        c -= 2; r -= 2
        all_chip[r:r+E.shape[0], c:c+E.shape[1]] += E
    
    bg_ann = pd.DataFrame()
    
    ref = list()
    x = list(np.where(all_chip == 1)[1])
    y = list(np.where(all_chip == 1)[0])
    ref = ['Chrome']*len(np.where(all_chip == 1)[1])
    x.extend(list(np.where(all_chip == 0)[1]))
    y.extend(list(np.where(all_chip == 0)[0]))
    ref.extend(['PolyT']*len(np.where(all_chip == 0)[1]))
    
    bg_ann['x'] = x
    bg_ann['y'] = y
    bg_ann['ref'] = ref
    del all_chip
    
    return bg_ann

#%% Annotation files for marker (AM1 & AM3)
def AM1_AM3_ann():
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
    
    AM_ann = pd.DataFrame()
    AM_ann['x'] = x
    AM_ann['y'] = y
    AM_ann['ref'] = np.array(ref)
    
    return AM_ann

#%% Annotation files for marker position
def marker_pos_ann(data):
    marker = data[data.x.map(lambda x: (x % 81) in np.linspace(0, 9, 10, dtype = int)) & \
                  data.y.map(lambda x: (x % 81) in np.linspace(0, 9, 10, dtype = int))]
    pos = pd.DataFrame({"marker_pos": list(zip((marker.x // 81).values, (marker.y // 81).values))})
    pos.index = marker.index
    data = pd.concat([data, pos], axis = 1)    
    return data

#%% Heatmap Plots
def chip_heatmap(data, plot_title, filename,
                 colorscale = [[0, 'rgba(0, 0 ,0, 1)'], 
                               [0.25, 'rgba(255, 255 ,0, 1)'],
                               [0.5, 'rgba(0, 255, 222, 1)'],
                               [0.75, 'rgba(255, 185, 255, 1)'],
                               [1, 'rgba(255, 0, 0, 1)']], 
                 zmax = 3000, 
                 width = 1210, 
                 height = 1200):
    
    data_matrix = data
    plot_data = go.Heatmap(z = data_matrix, colorscale = colorscale, zmin = 0,
                           zmax = zmax)
    
    data = [plot_data]
    
    layout = go.Layout(title = plot_title, width = width, height = height,
                       xaxis = dict(side = "top"),
                       yaxis = dict(autorange = "reversed"))
    
    fig = go.Figure(data = data, layout = layout)
    
    plotly.offline.plot(fig, 
            filename = '{}.html'.format(filename), 
            auto_open = False)
    
    pio.write_image(fig = fig, file = '{}.pdf'.format(filename), 
                    width = 297*3, height = 297*3, scale = 2)
    
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
    
    pio.write_image(fig = fig, file = '{}.pdf'.format(filename), 
                    width = 297*3, height = 297*3, scale = 2)


#%% Draw plots for each chip
def each_chip_plots(All, path_to_output):
    os.chdir(path_to_output)
    
    # Heatmaps
    ch2 = All.loc[:, ['x', 'y', 'mean_2', 'ref']]
    ch2_matrix = np.zeros((496, 496))
    for x, y, z in ch2.loc[:, ['x', 'y', 'mean_2']].values:
        ch2_matrix[int(y)][int(x)] = z
    
    ch4 = All.loc[:, ['x', 'y', 'mean_4', 'ref']]
    ch4_matrix = np.zeros((496, 496))
    for x, y, z in ch4.loc[:, ['x', 'y', 'mean_4']].values:
        ch4_matrix[int(y)][int(x)] = z
    
    chip_heatmap(data = ch2_matrix, 
                 plot_title = '<b>Heatmap for ch1 Data</b>',
                 filename = 'Heatmap_ch1')
    
    chip_heatmap(data = ch4_matrix, 
                 plot_title = '<b>Heatmap for ch2 Data</b>',
                 filename = 'Heatmap_ch2')
    
    # Violin plots
    ch2 = ch2[(All.ref == 'Chrome') | (All.ref == 'PolyT') | (All.ref == 'AM1') |(All.ref == 'AM3')]
    ch4 = ch4[(All.ref == 'Chrome') | (All.ref == 'PolyT') | (All.ref == 'AM1') |(All.ref == 'AM3')]
    
    chip_violin(data = ch2_matrix, 
                x_classification = ch2['ref'],
                y_value = ch2['mean_2'],
                plot_title = '<b>Violin Plots for ch1 Markers</b>',
                filename = 'Violin_ch1')
    
    chip_violin(data = ch4_matrix, 
                x_classification = ch4['ref'],
                y_value = ch4['mean_4'],
                plot_title = '<b>Violin Plots for ch2 Markers</b>',
                filename = 'Violin_ch2')
    
#%% Compute statistics for each chip
def each_chip_statistics(All, path_to_output):
    os.chdir(path_to_output)
    
    # marker csv
    marker = All[(All.ref == "AM1") | (All.ref == "AM3") | (All.ref == "PolyT") | (All.ref == "Chrome")]

    ch2_matrix = np.empty((496, 496))
    ch2_matrix[:] = np.nan
    ch4_matrix = np.empty((496, 496))
    ch4_matrix[:] = np.nan
    
    for x, y, m2, m4 in marker.loc[:, ['x', 'y', 'mean_2', 'mean_4']].values:
        ch2_matrix[int(y)][int(x)] = round(m2, 2)
        ch4_matrix[int(y)][int(x)] = round(m4, 2)
    
    np.savetxt("marker_ch1.csv", ch2_matrix, fmt = '%.4f', delimiter = ',')
    np.savetxt("marker_ch2.csv", ch4_matrix, fmt = '%.4f', delimiter = ',')
    
#%% Analysis in each chip
def each_chip_analysis(ch2, ch4, #ch2 = ch1, ch4 = ch2
                       file_path, each_chip_path, 
                       quantile_normalization = True):
    
    os.chdir(file_path)

    ann = pd.read_csv('Cent_M015_C88_annot.tab')
    ann = ann[['ref', 'x', 'y']]
    ann.y = 495 - ann.y #Reverse the yaxis
    
    # Background Annotation File
    bg_ann = polyt_chrome_ann(file_path)
    
    # AM1 & AM3 Annotation Files 
    AM_ann = AM1_AM3_ann()
    
    raw = np.stack((ch2['mean'].values, ch4['mean'].values), axis = 1)
    tmp = quantile_norm(raw)
    
    #Processed Data
    ch2_raw = pd.DataFrame()
    ch4_raw = pd.DataFrame()
    ch2_normalized = pd.DataFrame()
    ch4_normalized = pd.DataFrame()
    
    ch2_raw[['x', 'y']] = ch2[['x', 'y']]
    ch4_raw[['x', 'y']] = ch4[['x', 'y']]
    ch2_normalized[['x', 'y']] = ch2[['x', 'y']]
    ch4_normalized[['x', 'y']] = ch4[['x', 'y']]
    
    ch2_raw['mean'] = raw[:,0]
    ch4_raw['mean'] = raw[:,1]
    
    ch2_normalized['mean'] = tmp[:,0]
    ch4_normalized['mean'] = tmp[:,1]
    
    del ch2
    del ch4
    del raw
    del tmp
    
    #Merge Processed Data and Annotation Files
    tp1_raw = pd.merge(ch2_raw, ch4_raw, left_on = ['x', 'y'], right_on = ['x', 'y'], suffixes = ('_2', '_4'))
    res_raw = pd.merge(tp1_raw, ann, on = ['x', 'y'], how = 'outer')
    # res_raw.loc[(res_raw.ref == 'C') | (res_raw.ref == 'G'), 'label'] = 2
    # res_raw.loc[(res_raw.ref == 'A') | (res_raw.ref == 'T'), 'label'] = 4
    All_raw = pd.merge(res_raw, bg_ann, on = ['x', 'y'], how = 'left', suffixes = ('', '_bg'))
    All_raw = pd.merge(All_raw, AM_ann, on = ['x', 'y'], how = 'left', suffixes = ('', '_AM'))
    All_raw.loc[pd.isna(All_raw.ref), 'ref'] = All_raw.loc[pd.isna(All_raw.ref), 'ref_bg']
    All_raw.loc[pd.isna(All_raw.ref), 'ref'] = All_raw.loc[pd.isna(All_raw.ref), 'ref_AM']
    All_raw = All_raw.drop(['ref_bg', 'ref_AM'], axis = 1)
    # All_bg_raw = pd.merge(tp1_raw, bg_ann, left_on = ['x', 'y'], right_on = ['x', 'y'])
    # All_AM_raw = pd.merge(tp1_raw, AM_ann, left_on = ['x', 'y'], right_on = ['x', 'y'])
    del tp1_raw
    
    tp1_normalized = pd.merge(ch2_normalized, ch4_normalized, left_on = ['x', 'y'], right_on = ['x', 'y'], suffixes = ('_2', '_4'))
    res_normalized = pd.merge(tp1_normalized, ann, on = ['x', 'y'], how = 'outer')
    # res_normalized.loc[(res_normalized.ref == 'C') | (res_normalized.ref == 'G'), 'label'] = 2
    # res_normalized.loc[(res_normalized.ref == 'A') | (res_normalized.ref == 'T'), 'label'] = 4
    All_normalized = pd.merge(res_normalized, bg_ann, on = ['x', 'y'], how = 'left', suffixes = ('', '_bg'))
    All_normalized = pd.merge(All_normalized, AM_ann, on = ['x', 'y'], how = 'left', suffixes = ('', '_AM'))
    All_normalized.loc[pd.isna(All_normalized.ref), 'ref'] = All_normalized.loc[pd.isna(All_normalized.ref), 'ref_bg']
    All_normalized.loc[pd.isna(All_normalized.ref), 'ref'] = All_normalized.loc[pd.isna(All_normalized.ref), 'ref_AM']
    All_normalized = All_normalized.drop(['ref_bg', 'ref_AM'], axis = 1)
    # All_bg_normalized = pd.merge(tp1_normalized, bg_ann, left_on = ['x', 'y'], right_on = ['x', 'y'])
    # All_AM_normalized = pd.merge(tp1_normalized, AM_ann, left_on = ['x', 'y'], right_on = ['x', 'y'])
    del tp1_normalized

    each_chip_plots(All = All_raw, path_to_output = each_chip_path)
    
    each_chip_statistics(All = All_raw, path_to_output = each_chip_path)

#%% Compute groupby statistics
def all_chips_groupby_statistics(data, group_by, filename, fun = [], fun_name = []):
    # data = each_chip_marker_raw
    # group_by = ["task_id", 'marker_pos', 'ref']
    
    types = ['mean_2', 'mean_4']

    for col in types:
        # col = 'mean_4'
        ch_mean = data.groupby(group_by)[col].mean()
        ch_std = data.groupby(group_by)[col].std()
        ch_cv = data.groupby(group_by)[col].apply(lambda x: np.std(x)/np.mean(x))
        ch_summary = pd.concat([ch_mean, ch_std, ch_cv], 
                               keys = ['mean', 'std', 'CV'], axis = 1)
        
        ct = 0
        while(ct < len(fun)):
            ch_stat = data.groupby(group_by)[col].apply(fun[ct])
            ch_stat = ch_stat.to_frame().rename(columns = {col: fun_name[ct]})
            ch_summary = pd.concat([ch_summary, ch_stat], axis = 1)
            ct += 1
        
        if(col == types[0]):
            ch_summary.to_csv("{}_{}.csv".format(filename, 'ch1'))
        else:
            ch_summary.to_csv("{}_{}.csv".format(filename, 'ch2'))

#%% Draw summary plots for all chips
def all_chips_plots(data, mean_max_intensity, std_max_intensity):
    
    ch1_all_mean = data.groupby(["y", "x"])['mean_2'].mean().values.reshape(496, 496)
    ch1_all_std = data.groupby(["y", "x"])['mean_2'].std().values.reshape(496, 496)
    
    ch2_all_mean = data.groupby(["y", "x"])['mean_4'].mean().values.reshape(496, 496)
    ch2_all_std = data.groupby(["y", "x"])['mean_4'].std().values.reshape(496, 496)
    
    chip_heatmap(data = ch1_all_mean, 
                 zmax = mean_max_intensity,
                 plot_title = '<b>Heatmap for all chips mean (ch1)</b>',
                 filename = 'Heatmap_allchips_mean_ch1')
    
    chip_heatmap(data = ch1_all_std, 
                 zmax = std_max_intensity,
                 plot_title = '<b>Heatmap for all chips std (ch1)</b>',
                 filename = 'Heatmap_allchips_std_ch1')
    
    chip_heatmap(data = ch2_all_mean, 
                 zmax = mean_max_intensity,
                 plot_title = '<b>Heatmap for all chips mean (ch2)</b>',
                 filename = 'Heatmap_allchips_mean_ch2')
    
    chip_heatmap(data = ch2_all_std, 
                 zmax = std_max_intensity,
                 plot_title = '<b>Heatmap for all chips std (ch2)</b>',
                 filename = 'Heatmap_allchips_std_ch2')

    
#%% Compute statistics for all chips
def all_chips_statistics(data):  
    
    each_chip_marker_raw = data[~pd.isna(data.ref)]
    each_chip_marker_raw = marker_pos_ann(each_chip_marker_raw)
    each_chip_marker_raw.to_csv("each_chip_marker_raw.csv", index = False)
    
    all_chips_groupby_statistics(data = each_chip_marker_raw, 
                                 group_by = ["task_id", 'marker_pos', 'ref'],
                                 filename = "each_chip_marker_summary")
    
    all_chips_groupby_statistics(data = each_chip_marker_raw, 
                                 group_by = ["task_id", 'ref'],
                                 filename = "each_chip_summary")
    
    all_chips_groupby_statistics(data = each_chip_marker_raw, 
                                 group_by = ['marker_pos', 'ref'],
                                 filename = "each_marker_summary")

#%% All chips summary
def all_chips_summary(ch1_all, ch2_all):
    
    ch1_all.to_csv("output-ch1.tsv", sep = "\t", index = False)
    ch2_all.to_csv("output-ch2.tsv", sep = "\t", index = False)
    
    bg_ann = polyt_chrome_ann(path_to_folder)
    AM_ann = AM1_AM3_ann()

    each_chip_raw = pd.merge(ch1_all, ch2_all, 
                             left_on = ['task_id', 'x', 'y'], 
                             right_on = ['task_id', 'x', 'y'], 
                             suffixes = ('_2', '_4'))
    
    each_chip_raw = pd.merge(each_chip_raw, bg_ann, on = ['x', 'y'], how = 'left', suffixes = ('', '_bg'))
    each_chip_raw = pd.merge(each_chip_raw, AM_ann, on = ['x', 'y'], how = 'left', suffixes = ('', '_AM'))
    each_chip_raw.loc[pd.isna(each_chip_raw.ref), 'ref'] = each_chip_raw.loc[pd.isna(each_chip_raw.ref), 'ref_AM']
    each_chip_raw = each_chip_raw.drop(['ref_AM'], axis = 1)
    
    all_chips_plots(each_chip_raw, mean_max_intensity, std_max_intensity)
    
    all_chips_statistics(data = each_chip_raw)

#%% Make new working directories
def make_directories(wd, folder_list):
    for string in folder_list:
        dirpath = os.path.join(wd, string)
        try:
            os.mkdir(dirpath)
        except FileExistsError:
            continue
        else:
            print('Directory {} created'.format(dirpath))
#%%
#ch1 = channel 1, ch2 = channel 2 
#### i = 1
def fileIO():

    ## Collect input datapath, channel names, output data path #################################### 
    # os.chdir("C:/Users/jeff/Desktop/B1C-096-TW/Summit.Analysis_v0.1.1")  #test
    os.chdir('../')
    ch_names = []
    heatmap_paths = []
    output_paths = []
    
    for foldName, subfolders, filenames in os.walk(os.getcwd()):
        if foldName.endswith('grid'):
            output_paths.append(foldName)
        for filename in filenames:
            # filename = filenames[20]
            if filename.endswith('chip_log.json'):
                with open('{}\\chip_log.json'.format(foldName)) as channel:
                    channel_list = json.load(channel)
                    if(len(channel_list['channels']) != 3):                    
                        print("Channel list has {} channels".format(len(channel_list['channels'])))
                        sys.exit("Error: Channels are not correct!")                  
                    else:
                        ch_names.append(channel_list['channels'][1]['name'])
                        ch_names.append(channel_list['channels'][2]['name'])                            
            if filename.endswith('heatmap.csv'):
                heatmap_paths.append('{}\\heatmap.csv'.format(foldName))
    
    ## Run analysis for individual chip ###############################################
    ch1_all = pd.DataFrame()
    ch2_all = pd.DataFrame()
        
    for i in range(len(output_paths)):
        # i = 1
        os.chdir(output_paths[i])
        os.chdir('../')
        # ./B1C-096-TW/46_20190402141722/
        
        make_directories(os.getcwd(), ["analysis"])
        path_to_each_chip = os.path.join(os.getcwd(), "analysis")
        os.chdir(path_to_each_chip)
        # ./B1C-096-TW/46_20190402141722/analysis/
        
        # Read file
        for j in range(i*2, i*2 + 2):
            if(heatmap_paths[j].find(ch_names[i*2]) != -1):
                ch1 = pd.read_csv(heatmap_paths[j])[['task_id', 'x', 'y', 'mean']]
            else:
                ch2 = pd.read_csv(heatmap_paths[j])[['task_id', 'x', 'y', 'mean']]
        
        print(ch1.head(3))
        print(ch2.shape)
        
        ch1_all = ch1_all.append(ch1, ignore_index = True)
        ch2_all = ch2_all.append(ch2, ignore_index = True)
        
        each_chip_analysis(ch1, ch2,  
                           file_path = path_to_folder, 
                           each_chip_path = path_to_each_chip)
        
        os.chdir('../')
        # ./B1C-096-TW/46_20190402141722/
        
    os.chdir('../')
    # ./B1C-096-TW/
    
    RFID = os.getcwd().split("\\")[-1]
    make_directories(os.getcwd(), ["{}.analysis".format(RFID)])
    path_to_analysis_result = os.path.join(os.getcwd(), "{}.analysis".format(RFID))
    os.chdir(path_to_analysis_result)
    # ./B1C-096-TW/analysis/
    
    make_directories(os.getcwd(), ["{}.summary".format(RFID)])
    os.chdir(os.path.join(os.getcwd(), "{}.summary".format(RFID)))
    # ./B1C-096-TW/analysis/summary/
    
    all_chips_summary(ch1_all, ch2_all)
    
#%% path_to_folder = 'C:/Users/jeff/Desktop/B1C-096-TW/Summit.Analysis_v0.1.0'
path_to_folder = sys.argv[1]
mean_max_intensity = eval(sys.argv[2])
std_max_intensity = eval(sys.argv[3])

os.chdir(path_to_folder)
fileIO()
