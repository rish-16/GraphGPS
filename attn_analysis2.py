import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib 
from pprint import pprint

font = {'family' : 'normal',
        'size'   : 10}

matplotlib.rc('font', **font)

"""
S-default
VS-Heads
VS-Dims
VS-Layers
"""

data2plot = {
    "VVS-Heads": [
        {'dot': 0.00011658668518066406, 'softmax': 8.559226989746094e-05, 'proj': 8.296966552734375e-05},
        {'dot': 0.00010180473327636719, 'softmax': 6.413459777832031e-05, 'proj': 8.058547973632812e-05},
        {'dot': 0.00011539459228515625, 'softmax': 8.249282836914062e-05, 'proj': 8.726119995117188e-05},
        {'dot': 0.00010561943054199219, 'softmax': 9.441375732421875e-05, 'proj': 8.440017700195312e-05},
        {'dot': 0.00012063980102539062, 'softmax': 8.082389831542969e-05, 'proj': 8.20159912109375e-05}
    ],
    "VVS-Dims": [
        {'dot': 7.581710815429688e-05, 'softmax': 8.225440979003906e-05, 'proj': 7.748603820800781e-05},
        {'dot': 0.00010418891906738281, 'softmax': 8.058547973632812e-05, 'proj': 8.368492126464844e-05},
        {'dot': 9.274482727050781e-05, 'softmax': 8.487701416015625e-05, 'proj': 8.392333984375e-05},
        {'dot': 9.417533874511719e-05, 'softmax': 7.605552673339844e-05, 'proj': 8.177757263183594e-05},
        {'dot': 8.678436279296875e-05, 'softmax': 6.127357482910156e-05, 'proj': 7.796287536621094e-05}
    ],
    "VVS-Layers": [
        {'dot': 0.00011777877807617188, 'softmax': 7.843971252441406e-05, 'proj': 8.797645568847656e-05},
        {'dot': 0.0001220703125, 'softmax': 7.152557373046875e-05, 'proj': 8.749961853027344e-05},
        {'dot': 0.00012874603271484375, 'softmax': 8.893013000488281e-05, 'proj': 8.821487426757812e-05},
        {'dot': 0.00011372566223144531, 'softmax': 6.246566772460938e-05, 'proj': 0.00010132789611816406},
        {'dot': 0.00011706352233886719, 'softmax': 8.940696716308594e-05, 'proj': 9.322166442871094e-05}
    ]
}

final_data = [[], [], []]

for kid, (key, val) in enumerate(data2plot.items()):
    final_data[kid].append(key)

for kid, (key, val) in enumerate(data2plot.items()):
    temp = []
    dot = 0
    softmax = 0
    proj = 0
    
    for rec in val:
        dot += rec['dot']
        softmax += rec['softmax']
        proj += rec['proj']

    final_data[kid].append(dot / 5)
    final_data[kid].append(softmax / 5)
    final_data[kid].append(proj / 5)

pprint (final_data)

"""
[['VVS-Heads',
  0.00011200904846191407,
  8.149147033691407e-05,
  8.344650268554688e-05],
 ['VVS-Dims',
  9.074211120605469e-05,
  7.700920104980469e-05,
  8.096694946289063e-05],
 ['VVS-Layers',
  0.00011987686157226563,
  7.815361022949219e-05,
  9.164810180664062e-05]]
"""

# df = pd.DataFrame(data2plot, columns=["Model", "Dot", "Softmax", "Project"])

# df.plot(x='Model',
#         kind='bar',
#         stacked=True,
#         title='Attention | B=256 | Layer 1 | GINE+MHSA | RWSE')
# plt.xticks(rotation = 45)
# plt.show()