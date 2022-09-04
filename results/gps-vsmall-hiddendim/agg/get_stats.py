import os, json
import numpy as np
import matplotlib.pyplot as plt

PATH = "./train/stats.json"
BEST_PATH = "./test/best.json"

with open(PATH, "r")  as f:
    data = json.load(f)

with open(BEST_PATH, "r") as f:
    best_data = json.load(f)

data = data['stats']

times = [data[i]['time_epoch'] for i in range(len(data))]
best_mae = best_data['mae']
assert best_data['mae'] == best_data['loss']

time_avg = sum(times) / len(times)
print (f"Average time for gps-small-heads: {time_avg} seconds")

print (best_mae)