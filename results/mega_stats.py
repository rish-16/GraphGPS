import matplotlib.pyplot as plt
import numpy as np
import os, json

DEFAULT_TRAIN_PATH = './gps-default/agg/train/stats.json'
HEADS_TRAIN_PATH = './gps-vsmall-heads/agg/train/stats.json'
HIDDIM_TRAIN_PATH = './gps-vsmall-hiddendim/agg/train/stats.json'
LAYERS_TRAIN_PATH = './gps-vsmall-layers/agg/train/stats.json'
BS_TRAIN_PATH = './gps-vsmall-bs/agg/train/stats.json'

VVSMALL_HEADS_TRAIN_PATH = './gps-vvsmall-heads/agg/train/stats.json'
VVSMALL_HIDDIM_TRAIN_PATH = './gps-vvsmall-hiddendim/agg/train/stats.json'
VVSMALL_LAYERS_TRAIN_PATH = './gps-vvsmall-layers/agg/train/stats.json'
VVSMALL_BS_TRAIN_PATH = './gps-vvsmall-bs/agg/train/stats.json'

DEFAULT_TEST_PATH = './gps-default/agg/test/best.json'
HEADS_TEST_PATH = './gps-vsmall-heads/agg/test/best.json'
HIDDIM_TEST_PATH = './gps-vsmall-hiddendim/agg/test/best.json'
LAYERS_TEST_PATH = './gps-vsmall-layers/agg/test/best.json'
BS_TEST_PATH = './gps-vsmall-bs/agg/test/best.json'

VVSMALL_HEADS_TEST_PATH = './gps-vvsmall-heads/agg/test/best.json'
VVSMALL_HIDDIM_TEST_PATH = './gps-vvsmall-hiddendim/agg/test/best.json'
VVSMALL_LAYERS_TEST_PATH = './gps-vvsmall-layers/agg/test/best.json'
VVSMALL_BS_TEST_PATH = './gps-vvsmall-bs/agg/test/best.json'

train_collection = [
    DEFAULT_TRAIN_PATH, 
    HEADS_TRAIN_PATH, 
    HIDDIM_TRAIN_PATH, 
    LAYERS_TRAIN_PATH, 
    BS_TRAIN_PATH,
    VVSMALL_HEADS_TRAIN_PATH, 
    VVSMALL_HIDDIM_TRAIN_PATH, 
    VVSMALL_LAYERS_TRAIN_PATH, 
]

test_collection = [
    DEFAULT_TEST_PATH, 
    HEADS_TEST_PATH, 
    HIDDIM_TEST_PATH, 
    LAYERS_TEST_PATH, 
    BS_TEST_PATH, 
    VVSMALL_HEADS_TEST_PATH, 
    VVSMALL_HIDDIM_TEST_PATH, 
    VVSMALL_LAYERS_TEST_PATH
]

models = ["S-default", "VS-heads", "VS-dims", "VS-layers", "VS-bsize", "VVS-heads", "VVS-dims", "VVS-layers"]

# TRAINING TIME vs PARAM
xaxis = []
yaxis = []

for record in train_collection:
    print (record)
    with open(record, "r") as f:
        data = json.load(f)

    data = data['stats']
    times = []
    for i in range(len(data)):
        time = data[i]['time_epoch']
        times.append(time) 

    avg_train_time = sum(times) / len(times)
    print (avg_train_time, "\n")
    xaxis.append(avg_train_time)

for record in test_collection:
    print (record)
    with open(record, "r") as f:
        data = json.load(f)
        
    assert data['mae'] == data['loss']
    mae = data['mae']
    print (mae)
    yaxis.append(mae)

color_mapper = {
    "S-default": "#feca57",
    "VS-heads": "#b2de27",
    "VS-dims": "#c2f970",
    "VS-layers": "#b7f4d8",
    "VS-bsize": "#03c9a9",
    "VVS-heads": "#fe7968",
    "VVS-dims": "#c44d56",
    "VVS-layers": "#ff9478",
    # "VVS-bsize": "#5f27cd"
}

xaxis = np.asarray(xaxis)
yaxis = np.asarray(yaxis)
models = np.asarray(models)

fig, ax = plt.subplots(figsize=(11, 7), dpi=80)
for g in np.unique(models):
    ix = np.where(models == g)
    ax.scatter(xaxis[ix], yaxis[ix], c = color_mapper[g], label = g, s=200)

plt.xlabel("# Params (1E6)", fontsize=18)
plt.ylabel("Test MAE", fontsize=18)
ax.legend(fontsize=20, loc='upper right')
plt.show()
