import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np

fig = make_subplots(rows=3, cols=1 ,vertical_spacing = 0.01, shared_xaxes = True)

fig.append_trace(go.Heatmap(
    z = pd.DataFrame(np.random.random((100,100,))),
), row=1, col=1)

fig.append_trace(go.Heatmap(
    z = pd.DataFrame(np.random.random((100,100,))),
), row=2, col=1)

fig.append_trace(go.Heatmap(
    z = pd.DataFrame(np.random.random((100,100,)))
), row=3, col=1)

fig.layout(weight = )

plotly.offline.plot(fig, 
            filename = 'test.html', 
            auto_open = True)
    