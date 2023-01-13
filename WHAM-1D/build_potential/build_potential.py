import os, sys
import numpy as np
import matplotlib.pyplot as plt

max_z = 50 # kJ/mol
xbins = 5000
#min_x = -0.03 # Angstrom
#max_x = 0.26 # Angstrom
min_x = -0.10 # Angstrom
max_x = 0.30 # Angstrom

# The model 1D potential is defined using the following equation: 
# U(x) = \sum_{i=1}^5 U_i exp(-a_i (x-x_i)^2)
# With values for each i defined in U_terms in order: [Ui, Ai, Xi]

U_terms = [
[-74.831, 497.4447, 0.0000],
[-48.639, 149.2336, 0.1927],
[-48.639, 149.2336, 0.2601],
[-48.639, 149.2336, 0.3275],
[-74.831,  49.7444, 0.0963]
]

# Initialize empty potential
U = np.zeros(xbins)
xs = np.linspace(min_x, max_x, xbins)

def potential(x,terms):
    p = 0
    for t in terms:
        p += t[0]*np.exp(-t[1]*np.square(x-t[2]))
    return p

# Fill empty potential and set lowest value to zero
for index_x, x in enumerate(xs):
    U[index_x] = potential(x, U_terms)
U -= np.min(U)

# Save potential data to files
np.save('potential.npy', U)
np.save('x_axis.npy', xs)

# Set values above maximum z to NaN
for index_x, x in enumerate(xs):
    if U[index_x] > max_z:
        U[index_x] = np.NAN

# Plot curve
fig, ax = plt.subplots()
ax.plot(xs, U)
ax.set(xlabel='X ($\AA$)', ylabel='Free Energy (kJ/mol)', xlim=(-0.025,0.25))
fig.tight_layout()
plt.savefig('1d_potential.png', dpi=300)

sys.exit()

