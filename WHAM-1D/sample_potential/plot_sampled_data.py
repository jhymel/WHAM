import os, sys
import matplotlib.pyplot as plt
import numpy as np

xbins = 150
min_x = -0.03
max_x = 0.26
png_name = 'sampled_data.png'
dpi = 300

home = os.getcwd()

path_to_files = os.path.join(home, 'umbrella_samples')
files = [f for f in os.listdir(path_to_files) if f.startswith('x_')]
files.sort()

with open('wham_metadata.txt') as f:
    meta_data = f.readlines()

fig, ax = plt.subplots()
for line in meta_data:
    path = line.split()[0]
    data = np.genfromtxt(path)
    hist, bin_edges = np.histogram(data, bins=xbins, density=True, range=(min_x,max_x))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax.plot(bin_centers, hist)
    target_x = float(line.split()[1])
    ax.scatter(target_x, 0, color='black')

ax.set(xlabel='X ($\mathrm{\AA}$)', ylabel='P(x)')
plt.savefig(png_name, dpi=dpi)

sys.exit()

