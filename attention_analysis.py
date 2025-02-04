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

data2plot = [
    [
        [0.0008363723754882812, 7.033348083496094e-05, 6.961822509765625e-05, 6.365776062011719e-05, 7.200241088867188e-05], [0.0010521411895751953, 6.771087646484375e-05, 6.29425048828125e-05, 9.846687316894531e-05, 0.0002689361572265625], [0.00010585784912109375, 8.487701416015625e-05, 7.748603820800781e-05, 8.797645568847656e-05, 8.511543273925781e-05]
    ],
    [[6.604194641113281e-05, 6.890296936035156e-05, 6.580352783203125e-05, 6.175041198730469e-05, 6.437301635742188e-05], [7.462501525878906e-05, 0.00011134147644042969, 6.794929504394531e-05, 6.151199340820312e-05, 6.961822509765625e-05], [9.34600830078125e-05, 0.00010561943054199219, 8.988380432128906e-05, 7.557868957519531e-05, 8.273124694824219e-05]],
    [[5.841255187988281e-05, 5.4836273193359375e-05, 6.67572021484375e-05, 6.270408630371094e-05, 5.364418029785156e-05], [7.295608520507812e-05, 8.153915405273438e-05, 6.413459777832031e-05, 6.031990051269531e-05, 6.4849853515625e-05], [6.866455078125e-05, 6.413459777832031e-05, 5.9604644775390625e-05, 6.222724914550781e-05, 5.745887756347656e-05]],
    [[8.869171142578125e-05, 6.270408630371094e-05, 6.127357482910156e-05, 6.508827209472656e-05, 6.29425048828125e-05], [8.988380432128906e-05,  6.914138793945312e-05, 8.511543273925781e-05, 6.651878356933594e-05, 7.557868957519531e-05], [8.225440979003906e-05, 7.200241088867188e-05, 8.606910705566406e-05, 7.033348083496094e-05, 9.417533874511719e-05]]
]

for i in range(len(data2plot)):
    temps = []
    for j in range(len(data2plot[i])):
        tmp = sum(data2plot[i][j]) / len(data2plot[i][j])
        temps.append(tmp)

    data2plot[i] = temps

data2plot[0].insert(0, "S-Default")
data2plot[1].insert(0, "VS-Heads")
data2plot[2].insert(0, "VS-Dims")
data2plot[3].insert(0, "VS-Layers")

vvs_data = [
    ['VVS-Heads', 0.00011200904846191407, 8.149147033691407e-05, 8.344650268554688e-05],
    ['VVS-Dims', 9.074211120605469e-05, 7.700920104980469e-05, 8.096694946289063e-05],
    ['VVS-Layers', 0.00011987686157226563, 7.815361022949219e-05, 9.164810180664062e-05]
]

data2plot.extend(vvs_data)

df = pd.DataFrame(data2plot, columns=["Model", "Dot", "Softmax", "Project"])

df.plot(x='Model',
        kind='bar',
        stacked=False,
        title='Attention | B=256 | Layer 1 | GINE+MHSA | RWSE')
plt.xticks(rotation = 45)
plt.show()

"""
SMALL
***LOCAL MESSAGE PASSING TIME: 0.006603240966796875
******dot product time: 0.0008363723754882812
******softmax time: 0.0010521411895751953
******projection time: 0.00010585784912109375
***TIME FOR GLOBAL MP: 0.007480621337890625


***LOCAL MESSAGE PASSING TIME: 0.0008091926574707031
******dot product time: 3.504753112792969e-05
******softmax time: 3.0279159545898438e-05
******projection time: 7.653236389160156e-05
***TIME FOR GLOBAL MP: 0.0019669532775878906


***LOCAL MESSAGE PASSING TIME: 0.0004246234893798828
******dot product time: 3.1948089599609375e-05
******softmax time: 2.9087066650390625e-05
******projection time: 7.176399230957031e-05
***TIME FOR GLOBAL MP: 0.0009186267852783203


***LOCAL MESSAGE PASSING TIME: 0.0009558200836181641
******dot product time: 3.24249267578125e-05
******softmax time: 2.7894973754882812e-05
******projection time: 8.487701416015625e-05
***TIME FOR GLOBAL MP: 0.0009274482727050781


***LOCAL MESSAGE PASSING TIME: 0.00041294097900390625
******dot product time: 3.0040740966796875e-05
******softmax time: 2.6464462280273438e-05
******projection time: 7.009506225585938e-05
***TIME FOR GLOBAL MP: 0.0009350776672363281
"""

"""
VS-HEADS
***LOCAL MESSAGE PASSING TIME: 0.0013799667358398438
******dot product time: 6.604194641113281e-05
******softmax time: 7.462501525878906e-05
******projection time: 9.34600830078125e-05
***TIME FOR GLOBAL MP: 0.0013463497161865234


***LOCAL MESSAGE PASSING TIME: 0.0008680820465087891
******dot product time: 3.170967102050781e-05
******softmax time: 2.86102294921875e-05
******projection time: 7.390975952148438e-05
***TIME FOR GLOBAL MP: 0.0009663105010986328


***LOCAL MESSAGE PASSING TIME: 0.00042366981506347656
******dot product time: 2.9802322387695312e-05
******softmax time: 2.8848648071289062e-05
******projection time: 7.343292236328125e-05
***TIME FOR GLOBAL MP: 0.0008158683776855469


***LOCAL MESSAGE PASSING TIME: 0.0008330345153808594
******dot product time: 3.0279159545898438e-05
******softmax time: 2.7418136596679688e-05
******projection time: 7.200241088867188e-05
***TIME FOR GLOBAL MP: 0.0009014606475830078


***LOCAL MESSAGE PASSING TIME: 0.0004105567932128906
******dot product time: 2.956390380859375e-05
******softmax time: 2.6226043701171875e-05
******projection time: 7.200241088867188e-05
***TIME FOR GLOBAL MP: 0.0009255409240722656
"""

"""
VS-DIMS
***LOCAL MESSAGE PASSING TIME: 0.001239776611328125
******dot product time: 5.841255187988281e-05
******softmax time: 7.295608520507812e-05
******projection time: 6.866455078125e-05
***TIME FOR GLOBAL MP: 0.004102468490600586
"""

"""
VS-LAYERS
***LOCAL MESSAGE PASSING TIME: 0.0009181499481201172
******dot product time: 8.869171142578125e-05
******softmax time: 8.988380432128906e-05
******projection time: 8.225440979003906e-05
***TIME FOR GLOBAL MP: 0.004060506820678711
"""