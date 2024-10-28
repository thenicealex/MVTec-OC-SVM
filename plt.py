import matplotlib.pyplot as plt
import numpy as np

# AUC, f1, precision, recall
# Bottle
# hog = [0.834, 0.778, 0.875, 0.700]
# lbp = [0.564, 0.341, 0.333, 0.350]
# glcm = [0.642, 0.464, 0.361, 0.650]
# ch = [0.521, 0.235, 0.286, 0.2]

# Wood
hog = [0.850, 0.679, 0.514, 1.000]
lbp = [0.564, 0.341, 0.333, 0.350]
glcm = [0.642, 0.464, 0.361, 0.650]
ch = [0.521, 0.235, 0.286, 0.2]

plt.figure(figsize=(12, 8))

metrics = ['AUC', 'F1', 'Precision', 'Recall']
colors = ['#e3716e', '#54b3aa', '#7ac7e2', '#f7df87']

bar_width = 0.2

x = np.arange(len(metrics))

bars_hog = plt.bar(x - bar_width*1.5, hog, bar_width, label='HOG', color=colors[0])
bars_lbp = plt.bar(x - bar_width*0.5, lbp, bar_width, label='LBP', color=colors[1])
bars_glcm = plt.bar(x + bar_width*0.5, glcm, bar_width, label='GLCM', color=colors[2])
bars_ch = plt.bar(x + bar_width*1.5, ch, bar_width, label='Color histogram', color=colors[3])

def add_labels(bars, values, position):
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, f'{value:.3f}', ha='center', va='bottom', fontsize=8, color='black')

add_labels(bars_hog, hog, 'center')
add_labels(bars_lbp, lbp, 'center')
add_labels(bars_glcm, glcm, 'center')
add_labels(bars_ch, ch, 'center')

# for i, metric in enumerate(metrics):
#     plt.text(x[i] - bar_width, -0.02, metric, ha='center', fontsize=10, color='black')
    
plt.legend(title="Features", loc='upper right',  fontsize='small')

plt.title('Wood Performance Metrics for Different Features')
plt.xlabel('Metrics')
plt.ylabel('Scores')
plt.xticks(x, metrics)
plt.grid(True, linestyle='--', alpha=0.6)

plt.savefig("result_wood_single.png", dpi=300)