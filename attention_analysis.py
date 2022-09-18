import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib 

font = {'family' : 'normal',
        'size'   : 13}

matplotlib.rc('font', **font)

df = pd.DataFrame([
    ["S", ],
    ["VS-Hds", ],
    ["VS-Dims", ]
    ["VS-Layers", ]
    ["VVS-Hds", ],
    ["VVS-Dims", ]
    ["VVS-Layers", ]
], columns=["Model", "Dot", "Softmax", "Project"])

df.plot(x='Model',
        kind='bar',
        stacked=True,
        title='Attention Breakdown | B=256 | Layer 1 | GINE+MHSA')
plt.xticks(rotation = 45)
plt.show()