import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as pt
import plotly.graph_objects as go


def auto_contrast(images, thres = 40000, bins = 256):  
    image = images[0] if isinstance(images, list) else images
    if image.dtype == np.uint8:
        cmax = bins = 256
        cmin = 0
    elif image.dtype == np.uint16:
        cmax = bins = 65536
        cmin = 0
    elif image.dtype == np.float32 or image.dtype == np.float64:
        cmax = max(image.max() for image in images)
        cmin = 0
    else:
        raise Exception('undefined pixel type: {}'.format(image.dtype)) 
    h, x = np.histogram(image.ravel(), bins = bins, range = (cmin, cmax))
    if isinstance(images, list):
        for image in images[1:]:
            h += np.histogram(image.ravel(), bins = bins, range = (cmin, cmax))[0]
    c = len(images) if isinstance(images, list) else 1
    b = np.where(h > c * np.product(image.shape) / thres)[0]
    return x[b[0]], x[b[-1] + 1]

#%%
    
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

path = Path(r'C:\Git\data\20190402_1NIF880-23-08\195_20190402135139\indiv\df.csv')
df = pd.read_csv(str(path))
    
columns = ['probe_id', 'x', 'y', 'probe_seq_5p', 'ref']
annot = pd.read_csv('C:/Git/data/Cent_M015_C88_annot.tab', usecols = columns)
annot.y = 495 - annot.y

annot_np = annot.loc[annot.probe_id.str.contains('CEN-NP')].copy(True)
annot_np.loc[(annot_np.ref == 'A') | (annot_np.ref == 'T'), 'actual'] = 'AT'
annot_np.loc[(annot_np.ref == 'C') | (annot_np.ref == 'G'), 'actual'] = 'CG'

keys = ['x','y']
ps = pd.merge(annot_np, df, left_on = keys, right_on = keys)

columns = ['mean_g', 'mean_r']
model = LinearDiscriminantAnalysis()
inputs = np.log2(ps[columns].values)
model.fit(inputs, ps['actual'].values)
outputs = model.predict(inputs)
ps.loc[:, 'num_fails'] = np.array(outputs != ps['actual'].values, dtype = int)
accuracy = 1 - sum(ps['num_fails'].values) / len(ps)
print(accuracy)


c1,c2 = model.coef_[0]
calib = lambda x: 2.0 ** (- (np.log2(x) * c1 + model.intercept_[0]) / c2)
#
#pt.scatter(u, v, 2, alpha = 0.2)
#pt.xlim(9, 12)
#pt.ylim(9, 12)
#x = np.arange(9, 13)
#pt.plot(x, x, '-r')

#%%

g_values = calib(df['mean_g'].values).reshape((496,496))
r_values = df['mean_r'].values.reshape((496,496))
gmin, gmax = auto_contrast(g_values, 5000)
rmin, rmax = auto_contrast(r_values, 5000)
cmin = min(rmin, gmin)
cmax = max(rmax, gmax)

g_values = np.round(np.maximum(0, np.minimum(1, (g_values - gmin) / (gmax - gmin))) * 255)
r_values = np.round(np.maximum(0, np.minimum(1, (r_values - rmin) / (rmax - rmin))) * 255)

print(g_values.max(), g_values.min())
print(r_values.max(), r_values.min())

pt.figure()
pt.scatter(g_values, r_values, 3, alpha = 0.2)
pt.plot(np.arange(256), np.arange(256))
pt.xlim(0, 255)
pt.ylim(0, 255)
pt.show()

data = (g_values + r_values * 256) / 65535
        
colorscale = []
for r in range(256):
    colorscale.append([(  0 + r * 256) / 65535, 'rgb(0  ,{},0)'.format(r)])
    colorscale.append([(255 + r * 256) / 65535, 'rgb(255,{},0)'.format(r)])
    
#    for g in range(256):
#        colorscale.append([(g + r * 256) / 65535, 'rgb({},{},0)'.format(g,r)])

fig = go.Figure(
    data = go.Heatmap(
        z = data,
        zmin = 0.0,
        zmax = 1.0,
        colorscale = colorscale,
        showscale = False,
    ),
    layout = go.Layout(
        width  = 800,
        height = 800,
        xaxis = dict(side = "top"),
        yaxis = dict(autorange = "reversed")
    )
)
fig.write_html(
    str('topography.html'),
    auto_open = True,
)

#%%

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