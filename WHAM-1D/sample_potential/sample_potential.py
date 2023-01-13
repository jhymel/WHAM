import os, sys
import numpy as np
import random

home = os.getcwd()

# Load in model 1D potential created in build_potential directory
xs = np.load('../build_potential/x_axis.npy')
U = np.load('../build_potential/potential.npy')

kbT = 2.479 # kJ/mol at 300K
kx = 150000 # Umbrella force constant: kJ/mol/Angstrom^2
start, stop = -0.03, 0.26 # Initial and final umbrella locations # Angstrom
spacing = 0.01 # Umbrella spacing: Angstrom
samples = 100000 # per umbrella
output_directory_name = 'umbrella_samples'

def umbrella_potential(kx, x0, xs):
    umb_pot = np.zeros(len(xs))
    for x_index, xi in enumerate(xs):
        umb_pot[x_index] = 0.5*kx*np.square(xi-x0)
    return umb_pot

# Loop over umbrella midpoints
g = open('wham_metadata.txt', 'w+')
for x in np.arange(start, stop, spacing):
    x = np.round(x,4)
    print (x)

    # Build biased probability distribution from underlying potential + umbrella bias
    umb_pot = umbrella_potential(kx, x, xs)
    total_pot = umb_pot + U
    prob = np.exp(total_pot/(-1*kbT))
    prob /= np.sum(prob)

    # Sample points from an nD probability distribution
    # Found code at: https://stackoverflow.com/questions/56017163
    flat = prob.flatten()
    sample_index = np.random.choice(a=flat.size, size=samples, p=flat)
    adjusted_index = np.unravel_index(sample_index, prob.shape)
    adjusted_index = np.array(list(zip(*adjusted_index)))

    # Write out each set of sampled points to wham_metadata.txt
    # Format: path/to/samples force_constant umbrella_center
    output_directory = os.path.join(home, output_directory_name)
    g.write('%s/x_%.4f %s %s\n' % (output_directory, x, x, kx))
    if not os.path.exists(output_directory): os.mkdir(output_directory)
    with open('%s/x_%.4f' % (output_directory, x),'w+') as f:
        for guess in adjusted_index:
            x_val = xs[guess[0]]
            f.write('%.5f\n' % x_val)
g.close()
sys.exit()

