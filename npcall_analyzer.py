# -*- coding: utf-8 -*-
import sys, os, traceback, json, csv
import subprocess, concurrent.futures as cf
import tkinter as tk, tkfilebrowser as tkbrowser
import numpy as np, matplotlib.pyplot as pt, pandas as pd
# from _plotly_future_ import v4_subplots
from plotly import graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
from pathlib import Path
from itertools import product
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.mixture import GaussianMixture
# os.chdir(r"C:\Users\jeff\Desktop\Centrillion\cen_work_material\Microarray\Summit.NPCallAnalyzer")
from imgproc_utils import quantile_norm, auto_contrast
from npcall_viz import visualize_npcall_distribution, visualize_npcall_degrade
from bg_ann import polyt_chrome_ann, AM1_AM3_AM5_ann
from bgann_analysis import each_chip_statistics, each_chip_plots

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

def draw_topography(
    result_path = 'topography.html',
    auto_thres = 10000,
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
            width  = 850,
            height = 800,
            xaxis  = dict(
                side = 'top',
                constrain = 'domain',
            ),
            yaxis  = dict(
                autorange = 'reversed',
                scaleanchor = 'x',
                constrain = 'domain',
            ),
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
    pio.write_html(
        fig,
        str(result_path),
        auto_open = auto_open,
    )

def __dfs(u, visited, neighbors, items):
    visited.add(u)
    items.append(u)
    for v in neighbors(u):
        if v not in visited:
            __dfs(v, visited, neighbors, items)

def partition_via_dfs(vertices, neighbors):
    visited = set()
    for v in vertices:
        if v not in visited:
            items = []
            __dfs(v, visited, neighbors, items)
            yield items



class SummitGrid:
    
    def __init__(self, binary_path):
        self.binary_path = binary_path
    
    def __call__(self, import_dir, export_dir):
        subprocess.run([
            str(self.binary_path),
            '-i', str(import_dir),
            '-o', str(export_dir),
            '-n', '3',
            # '-d', '6',
            '--marker_append',
            '--no_bgp',
        ],
        check = True)

class Application(tk.Frame):
    
    PADDING_SIZE = 3
    LISTBOX_SIZE = (100, 30)   

    def __init__(self, master = None, config_file = None):
        print('parse settings.json')
        self.config_file = config_file
        with open(config_file) as f:
            opts = json.load(f)
            
            # configure path to Summit.Grid
            path = Path(opts['summit_grid'])
            if path.exists():
                self.summit_grid = SummitGrid(path)
            else:
                raise OSError('Summit.Grid not found: {}'.format(str(path)))
            
            # configure path to annotation file
#            path = Path(opts['annot_path'])
#            if path.exists():
#                self.annot_path = path
#            else:
#                raise OSError('Annotation file not found: {}'.format(str(path)))
            
            # configure export directory
            export_dir = Path(opts['export_dir'])
            export_dir.mkdir(exist_ok = True)
            
            # configure auto open
            self.auto_open = opts['auto_open']
            
            # configure font size
            self.font_size = opts['font_size']
        
        print('build UI components')
        This = type(self)
        tk.Frame.__init__(self, master)    
        self.root = master
        super().pack(pady = This.PADDING_SIZE)
        
        ctrl = tk.Frame(master)
        ctrl.pack(fill = tk.X, padx = This.PADDING_SIZE, pady = This.PADDING_SIZE)
        
        self.buttons = [
            tk.Button(ctrl, text = 'Import', command = self.handle_browse_import_dirs),
            tk.Button(ctrl, text = 'Export', command = self.handle_browse_export_dir),
            tk.Button(ctrl, text = 'Cross analysis', command = self.handle_cross_analysis),
            tk.Button(ctrl, text = 'Indiv analysis', command = self.handle_indiv_analysis),
            tk.Button(ctrl, text = 'Summit Grid'   , command = self.handle_summit_grid),
        ]
        for button in self.buttons[:2]:
            button.pack(padx = This.PADDING_SIZE, side = tk.LEFT)
        self.export_dir = tk.Label(ctrl, text = str(export_dir))
        self.export_dir.pack(padx = This.PADDING_SIZE, side = tk.LEFT)
        for button in self.buttons[2:]:
            button.pack(padx = This.PADDING_SIZE, side = tk.RIGHT)
            
        frame = tk.Frame(master)
        frame.pack(fill = tk.X, padx = This.PADDING_SIZE, pady = This.PADDING_SIZE)
        
        self.flags = { key: tk.IntVar() for key in ['QN/ALL', 'QN/NP', 'LDA/NP'] }
        self.flags['LDA/NP'].set(1)
        for key, value in self.flags.items():            
            checkbutton = tk.Checkbutton(frame, text = key, variable = value)
            checkbutton.pack(padx = This.PADDING_SIZE, side = tk.RIGHT)
        
        file = tk.Frame(master)
        file.pack(fill = tk.X, pady = This.PADDING_SIZE, padx = This.PADDING_SIZE)
        
        self.listbox = tk.Listbox(
            file,
            width = This.LISTBOX_SIZE[0],
            height = This.LISTBOX_SIZE[1],
            selectmode = tk.EXTENDED
        )
        self.listbox.pack(pady = This.PADDING_SIZE, padx = This.PADDING_SIZE, side = tk.LEFT)
        
        print('Initialization complete')
    
    
    def selection(self):
        return map(Path, map(self.listbox.get, self.listbox.curselection()))    
    
    @staticmethod
    def get_directory_from_config(opts, key):
        try:
            path = Path(opts[key])
            if not path.is_dir():
                raise KeyError
            else:
                return path
        except KeyError:
            return Path.home()
    
    def handle_browse_import_dirs(self):
        key  = 'import_dir'
        with open(self.config_file) as f:
            opts = json.load(f)
            path = str(type(self).get_directory_from_config(opts, key))
        dirnames = tkbrowser.askopendirnames(title = 'Import', initialdir = path)
        for dirname in list(dirnames):
            self.listbox.insert('end', dirname)
        with open(self.config_file, 'w') as f:
            opts[key] = Path(dirname).parent.as_posix()
            json.dump(opts, f, indent = 4)

    def handle_browse_export_dir(self):
        key  = 'export_dir'
        with open(self.config_file) as f:
            opts = json.load(f)
            path = str(self.get_directory_from_config(opts, key))
        dirname = tkbrowser.askopendirname(title = 'Export', initialdir = path)
        self.export_dir['text'] = dirname
        with open(self.config_file, 'w') as f:
            opts[key] = Path(dirname).as_posix()
            json.dump(opts, f, indent = 4)

    def handle_summit_grid(self):
        self.apply_summit_grid(self.selection())
    
    def handle_indiv_analysis(self):
        self.apply_summit_grid(
            import_dir
            for import_dir in self.selection()
            if not (import_dir / 'grid' / 'COMPLETE').exists()
        )
        self.apply_indiv_analysis(self.selection())
    
    def handle_cross_analysis(self):
        self.apply_summit_grid(
            import_dir
            for import_dir in self.selection()
            if not (import_dir / 'grid' / 'COMPLETE').exists()
        )
        self.apply_indiv_analysis(
            import_dir
            for import_dir in self.selection()
            if not (import_dir / 'indiv' / 'COMPLETE').exists()
        )
        self.apply_cross_analysis(self.selection())

    #===================================================================================
    
    @staticmethod
    def get_sample_info(import_dir):
        with open(str(import_dir / 'chip_log.json')) as f:
            info = json.load(f)
            info_spec = info['chip']['spec']
            # chip.name
            chip_name = info_spec['name']
            # chip.shape
            chip_shape = info_spec['h_cl'], info_spec['w_cl']
            # channels
            channels = [None] * 6
            for item in info['channels']:
                channels[int(item['filter'])] = item['name']
            else:
                channels = [ c for c in channels if c is not None ]
            # return
            return chip_name, chip_shape, channels
    
    def apply_summit_grid(self, import_dirs):
        import_dirs = list(import_dirs)
        if len(import_dirs) == 0:
            return
        print('apply Summit.Grid')
        with cf.ThreadPoolExecutor(max_workers = 3) as pool:
            futures = {}
            for import_dir in import_dirs:
                print('-', 'start', str(import_dir))
                future = pool.submit(self.summit_grid, import_dir, import_dir)
                futures[future] = import_dir
            for future in cf.as_completed(futures):
                import_dir = futures[future]
                future.result()
                print('-', str(import_dir), 'finished')
        print('complete\n')
    
    def draw_np_failed_calls(self, ps, shape, annot_np, output_dir):
        cnts = ps.groupby(['x','y'])['num_fails'].sum() # ps
        matx = np.ones(shape, int) * -1
        for (x, y), c in cnts.iteritems():
            matx[y, x] = c
        fig = make_subplots(
            rows = 3,
            cols = 1,
            vertical_spacing = 0.05,
        )
        
        y_values = np.unique(annot_np['y'].values)
        neighbors = lambda u: (v for v in [u - 1, u + 1] if u in y_values)
        for row, rng in enumerate(partition_via_dfs(y_values, neighbors), 1): #banff
        # for row, rng in enumerate([ yrng[:5], yrng[5:10], yrng[10:] ], 1): #yz01
            ymin, ymax = min(rng), max(rng) + 1
            fig.add_trace(
                go.Heatmap(
                    z = matx[ymin:ymax, :],
                    x = np.arange(496),
                    y = np.arange(ymin, ymax),
                ),
                row = row,
                col = 1,
            )
            fig.layout.update(**{
                'xaxis{}'.format(row): dict(side = "bottom"),
                'yaxis{}'.format(row): dict(autorange = 'reversed')
            })
        fig.layout.update(
            autosize = False,
            width    = 500 * 10,
            height   = 700,
        )
        pio.write_html(
            fig,
            str(output_dir / 'failure_map.html'),
            auto_open = self.auto_open,
        )
    
    def apply_indiv_analysis(self, import_dirs):
        This = type(self)
        import_dirs = list(import_dirs)
        export_dir  = Path(self.export_dir['text'])
        if len(import_dirs) == 0:
            return
        
        annots = {}
        bg_annots = {}
        
        print('for each sample')
        with open(str(export_dir / 'npcall.csv'), 'w', newline = '') as f:
            writer = csv.writer(f)
            for import_dir in import_dirs:
                record = [str(import_dir)]
                
                print('-', 'start', str(import_dir))
                output_dir = import_dir / 'indiv'
                if not output_dir.exists():
                    output_dir.mkdir(exist_ok = True)
                
                print('  -', 'get sample information')
                chip_name, chip_shape, channels = This.get_sample_info(import_dir)
                
                if chip_name not in annots:
                    print('  -', 'load annotations')
                    with open(self.config_file) as f:
                        opts = json.load(f)
                        info = opts['annotation'][chip_name]
                        columns = ['probe_id', 'x', 'y'] + list(info['cols'].keys())
                        annot = pd.read_csv(info['path'], usecols = columns)
                        annot = annot.rename(columns = info['cols'])
                        annot.y = chip_shape[1] - annot.y - 1
                        annots[chip_name] = annot
                else:
                    annot = annots[chip_name]
                
                print('  -', 'merge channels')
                columns = ['x', 'y', 'mean']
                path = import_dir / 'grid' / 'channels' / '{}' / 'heatmap.csv'
                df = pd.merge(
                    pd.read_csv(str(path).format(channels[1]), usecols = columns),
                    pd.read_csv(str(path).format(channels[2]), usecols = columns),
                    left_on = columns[:2], right_on = columns[:2], suffixes = ('_g', '_r')
                )
                # df.to_csv(str(output_dir / 'df.csv'), index = 0)
                
                if self.flags['QN/ALL'].get() == 1:
                    print('  -', 'apply quantile normalization to all probes')
                    columns = ['mean_g', 'mean_r']
                    df.loc[:, columns] = quantile_norm(df[columns].values)
                
                print('  -', 'fetch NP probe annotations')
                annot_np = annot.loc[annot.probe_id.str.contains('CEN-NP')].copy(True)
                annot_np.loc[(annot_np.ref == 'C') | (annot_np.ref == 'G'), 'actual'] = 'CG'
                annot_np.loc[(annot_np.ref == 'A') | (annot_np.ref == 'T'), 'actual'] = 'AT'
                
                print('  -', 'fetch NP probe intensities')
                keys = ['x','y']
                ps = pd.merge(annot_np, df, on = keys)
                
                print('  -', 'fetch AM1, AM3 (AM5), Chrome, PolyT probe annotations')
                if chip_name not in bg_annots:
                    with open(self.config_file) as f:
                        opts = json.load(f)
                        file_path = opts['annotation']['banff']['path'].split("/")
                        file_path.pop()
                        file_path = '/'.join(file_path)
                
                    p_c_ann = polyt_chrome_ann()[chip_name]
                    am_ann = AM1_AM3_AM5_ann()[chip_name]
                    bg_annot = dict(PolyT_Chrome = p_c_ann, AM = am_ann)
                    
                    bg_annots[chip_name] = bg_annot
                
                else:
                    
                    bg_annot = bg_annots[chip_name]

                print('  -', 'fetch AM1, AM3 (AM5), Chrome, PolyT probe intensities')
                keys = ['x','y']
                temp = pd.merge(annot_np, df, on = keys, how = 'outer')
                all_df = pd.merge(temp, bg_annot["PolyT_Chrome"], on = keys, how = 'left', suffixes = ('', '_bg'))
                all_df = pd.merge(all_df, bg_annot["AM"], on = keys, how = 'left', suffixes = ('', '_AM'))
                all_df.loc[pd.isna(all_df.ref), 'ref'] = all_df.loc[pd.isna(all_df.ref), 'ref_bg']
                all_df.loc[pd.isna(all_df.ref), 'ref'] = all_df.loc[pd.isna(all_df.ref), 'ref_AM']
                all_df = all_df.drop(['ref_bg', 'ref_AM'], axis = 1)
                del temp

                print('  -', 'export analysis for AM1, AM3 (AM5), Chrome, PolyT probes')
                each_chip_plots(All = all_df, path_to_output = output_dir, chip_name = chip_name)
                each_chip_statistics(All = all_df, path_to_output = output_dir, chip_name = chip_name)                
                
                if self.flags['QN/NP'].get() == 1:
                    print('  -', 'apply quantile normalization to NP probes')
                    columns = ['mean_g', 'mean_r']
                    ps.loc[:, columns] = quantile_norm(ps[columns].values)
                
                # ps.loc[:, 'contrast'] = (ps.mean_g - ps.mean_r) / (ps.mean_g + ps.mean_r)

#                bc = True
#                inc_na = True
#                
#                if bc:
#                    eps = 1e-4
#                    columns = ['mean_g', 'mean_r']
#                    value0 = ps[columns[0]].values - np.percentile(ps[columns[0]].values, 20)
#                    value1 = ps[columns[1]].values - np.percentile(ps[columns[1]].values, 40)
#                    ps.loc[:, columns[0]] = np.maximum(eps, value0)
#                    ps.loc[:, columns[1]] = np.maximum(eps, value1)
#                    nnz = (value0 > eps) & (value1 > eps)
#                    columns = ['mean_g', 'mean_r']
#                    inputs  = np.log2(ps.loc[nnz, columns].values)
#                    targets = ps.loc[nnz,'actual'].values
#                else:
                
                columns  = ['mean_g', 'mean_r']
                inputs   = np.log2(ps.loc[:, columns].values)
                targets  = ps.loc[:, 'actual'].values
                # contrast = ps.loc[:, 'contrast'].values
                
                print('  -', 'apply linear discriminant analysis')
                model = LinearDiscriminantAnalysis()
                model.fit(inputs, ps['actual'].values)
                outputs = model.predict(inputs)
                ps.loc[:, 'predicted'] = outputs
                ps.loc[:, 'num_fails'] = np.array(outputs != ps['actual'].values, dtype = int)
                ps.loc[:, 'confidence'] = np.abs(inputs.dot(model.coef_.transpose())[:,0] + model.intercept_)
                # ps.to_csv(str(output_dir / 'ps.csv'), index = 0)
                accuracy = 1 - sum(ps['num_fails'].values) / len(ps)
                record.append('{:.2f}'.format(accuracy * 100))
                
                print('  -', 'export a heatmap of NP failed calls')
                self.draw_np_failed_calls(ps, chip_shape, annot_np, output_dir)
                
                print('  -', 'export group failed calls by replicates')
                res = ps.groupby(['seq'])['num_fails'].sum()
                res.to_csv(str(output_dir / 'rep_errors.csv'), header = False)
                
                print('  -', 'export topography')
                r_values = df['mean_r'].values.reshape(chip_shape)
                g_values = df['mean_g'].values.reshape(chip_shape)
                c1,c2 = model.coef_[0]
                calib = lambda x: 2.0 ** (- (np.log2(x) * c1 + model.intercept_[0]) / c2)
                if self.flags['LDA/NP'].get() == 1:
                    print('  -', 'apply linear normalization')
                    g_values = calib(g_values)
                draw_topography(
                    intensity = np.stack((g_values, r_values)),
                    result_path = output_dir / 'topography.html',
                    auto_thres = 5000,
                    auto_open = self.auto_open,
                )
                tp = df.copy(True)
                tp.loc[:, 'mean_g'] = g_values.ravel()
                tp.loc[:, 'mean_r'] = r_values.ravel()
                tp.to_csv(str(output_dir / 'topography.csv'), index = 0)
                
                print('  -', 're-fetch NP probe intensities')
                ps = pd.merge(annot_np, df, on = keys)
                if self.flags['QN/NP'].get() == 1:
                    print('  -', 're-apply quantile normalization to NP probes')
                    columns = ['mean_g', 'mean_r']
                    ps.loc[:, columns] = quantile_norm(ps[columns].values)
                
                print('  -', 'export scatter plot')
                css_at = {'color': 'green', 'alpha': 0.2 }
                css_cg = {'color': 'red'  , 'alpha': 0.2 }
                at = ps[ps.actual == 'AT']
                cg = ps[ps.actual == 'CG']
                text = 'ACC: {:.2f}%'.format(accuracy * 100)
                fig = pt.figure(figsize = (12, 12))
                ax = fig.add_subplot(1,1,1)                
                ax.scatter(np.log2(at.mean_g.values), np.log2(at.mean_r.values), 4, **css_at)
                ax.scatter(np.log2(cg.mean_g.values), np.log2(cg.mean_r.values), 4, **css_cg)
                # xmin, xmax = min(ps.mean_g.values), max(ps.mean_g.values)
                # ax.plot(np.log2([xmin, xmax]), np.log2([calib(xmin), calib(xmax)]), '-b')
                ax.set_xlabel('{} intensities'.format(channels[1]))
                ax.set_ylabel('{} intensities'.format(channels[2]))
                ax.legend(['Decision boundary', 'C/G type', 'A/T type'])
                ax.text(
                    np.array([0.03, 0.97]).dot(ax.get_xlim()),
                    np.array([0.13, 0.87]).dot(ax.get_ylim()),
                    text,
                    verticalalignment = 'top',
                    horizontalalignment = 'right'
                )
                fig.set_tight_layout(True)
                fig.savefig(str(output_dir / 'scatter.png'), dpi = 200)
                pt.close(fig)
                record.append(str(output_dir / 'scatter.LDA.png'))
                
                
                for key in ['QDA']:
                    print('  -', 'apply', key)
                    if key == 'GMM':
                        means_init = np.diag(inputs.mean(axis = 0)) + 5
                        # print(means_init)
                        title = 'Results of NP call by using unsupervised clustering (GMM)'
                        model = GaussianMixture(
                            n_components = 2,
                            means_init = means_init,
                            warm_start = True,
                            tol = 1e-6,
                        )
                        model.fit(inputs)
                        indexes = np.argsort([ np.arctan2(*mean[::-1]) for mean in model.means_ ])
                        decoder = np.array(['CG', 'AT'])[indexes]
                        outputs = np.array([decoder[i] for i in model.predict(inputs)])
                        means = model.means_
                        covariances = model.covariances_
                    else:
                        title = 'Results of NP call by using supervised clustering (QDA)'
                        model = QuadraticDiscriminantAnalysis(store_covariance = True)
                        model.fit(inputs, targets)
                        
#                        if inc_na:
#                            inputs  = np.log2(ps.loc[:,columns].values)
#                            targets = ps.loc[:,'actual'].values
                        
                        outputs = model.predict(inputs)
                        means = model.means_
                        covariances = np.array(model.covariance_)
                    path = output_dir / 'scatter.{}.png'.format(key)
                    visualize_npcall_distribution(
                        inputs,
                        outputs,
                        targets,
                        list(zip(('CG','AT'), ('green','red'))),
                        means,
                        covariances,
                        export_path = path,
                        title = title,
                        fontsize = self.font_size
                    )
                    path = output_dir / 'scatter.{}.degrade.png'.format(key)
                    visualize_npcall_degrade(
                        inputs,
                        outputs,
                        targets,
                        list(zip(('CG','AT'), ('green','red'))),
                        means,
                        covariances,
                        export_path = path,
                        title = title,
                        fontsize = self.font_size
                    )
                
                # draw scatter plot
                writer.writerow(record)
                
                with open(str(output_dir / 'COMPLETE'), 'w'):
                    pass
                
                
                
        print('complete\n')
        
    
    def apply_cross_analysis(self, import_dirs):
        This = type(self)
        import_dirs = list(import_dirs)
        export_dir  = Path(self.export_dir['text'])
        if len(import_dirs) == 0:
            return
        
        print('get sample information')
        chip_name, chip_shape, channels = This.get_sample_info(import_dirs[0])
        
        print('load annotations')
        with open(self.config_file) as f:
            opts = json.load(f)
            info = opts['annotation'][chip_name]
            columns = ['probe_id', 'x', 'y'] + list(info['cols'].keys())
            annot = pd.read_csv(info['path'], usecols = columns)
            annot = annot.rename(columns = info['cols'])
            annot.y = chip_shape[1] - annot.y - 1
        
        print('export topography')
        keys = ['y', 'x']
        sels = ['mean_g', 'mean_r']
        size = chip_shape + (-1,)
        df = pd.concat([
            pd.read_csv(str(import_dir / 'indiv' / 'topography.csv'))
            for import_dir in import_dirs
        ])
        gp = df.groupby(keys)
        median = gp.median()[sels].values.reshape(size).transpose(2,0,1)
        mean   = gp.mean()  [sels].values.reshape(size).transpose(2,0,1)
        stdev  = gp.std()   [sels].values.reshape(size).transpose(2,0,1)
        cv     = stdev / mean * 100
        draw_topography(
            median = median,
            mean = mean,
            stdev = stdev,
            cv = cv,
            result_path = export_dir / 'topography.html',
            auto_thres = 5000,
            auto_open = self.auto_open,
        )
        
        print('  -', 'fetch NP probe annotations')
        annot_np = annot.loc[annot.probe_id.str.contains('CEN-NP')].copy(True)
        annot_np.loc[(annot_np.ref == 'A') | (annot_np.ref == 'T'), 'actual'] = 'AT'
        annot_np.loc[(annot_np.ref == 'C') | (annot_np.ref == 'G'), 'actual'] = 'CG'
        
        print('  -', 'merge npcall results with NP probe annotations')
        df = pd.concat([
            pd.read_csv(str(import_dir / 'indiv' / 'ps.csv'))
            for import_dir in import_dirs
        ])
        num_fails = df.groupby(keys)['num_fails'].sum().reset_index(['x','y'])        
        self.draw_np_failed_calls(num_fails, chip_shape, annot_np, export_dir)
        
        print('  -', 'analyze replicate error probes')
        rep = pd.concat([
            pd.read_csv(str(import_dir / 'indiv' / 'rep_errors.csv'), header = None)
            for import_dir in import_dirs
        ])
        res = rep.groupby(0).sum().reset_index(0)
        res.to_csv(str(export_dir / 'rep_errors.csv'), header = False, index = 0)        
        cnts = res[1].values
        bins = np.arange(max(cnts) + 2) - 0.5
        h, _ = np.histogram(cnts, bins)
        x    = np.arange(len(h))
        fig = pt.figure(figsize = (12, 8))
        ax = fig.add_subplot(1,1,1)
        ax.bar(x[1:], h[1:])
        ax.set_xlabel('# failed calls')
        ax.set_ylabel('# distinct assays')
        ax.set_xticks(x[1:])
        for c, v in zip(x[1:], h[1:]):
            ax.text(c, v, str(v), ha = 'center', va = 'bottom')
        fig.set_tight_layout(True)
        fig.savefig(str(export_dir / 'rep_errors.png'), dpi = 200)
        pt.close(fig)
        
        print('complete\n')
        
    
if __name__ == '__main__':
    try:
        win = tk.Tk()
        win.title('SUMMIT Image Data Analyzer')
        app = Application(win, './resource/settings.json')
        app.mainloop()
    except Exception:
        traceback.print_exception(*sys.exc_info())
        os.system('pause')
    finally:
        # os.system('pause')
        pass
