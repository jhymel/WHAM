import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps

class simulation_class:
    """
    Class for storing information relevant to a single umbrella simulation.
    """
    def __init__(self, colvar_file, beta, kx, target_x, xbins=50, f=1.0):
        self.colvar_data = np.genfromtxt(colvar_file)
        self.n_samples = self.colvar_data.size
        self.beta = beta
        self.kx = kx
        self.target_x = target_x
        self.f = f
        self.xbins = xbins

    def compute_biased_prob(self, xmin=-0.03, xmax=0.26):
        """
        Computed umbrella-biased probability distribution without any reweighting.
        Biased probability is computed for a single umbrella simulation and attached to a simulation object as an instance variable.
        """
        histogram, xedges = np.histogram(self.colvar_data, bins=self.xbins, density=True, range=(xmin,xmax))
        xcenters = (xedges[:-1] + xedges[1:]) / 2
        self.xedges = xedges
        self.xcenters = xcenters
        self.prob_biased = histogram


class wham_1d_reweighting:
    """
    Class for computing reweighting of 1D WHAM simulations
    """
    def __init__(self, targets, kx, colvar_paths, beta, xbins):
        """
        Load all collective variable data into a list of simulation objects and compute biased probabilites
        """
        self.beta = beta
        self.kx = kx
        self.xbins = xbins
        simulation_data_list = []
        for target, colvar_path in zip(targets, colvar_paths):
            simulation_data = simulation_class(colvar_file=colvar_path, beta=self.beta, kx=self.kx, target_x=target, xbins=self.xbins)
            simulation_data_list.append(simulation_data)
        self.simulation_data_list = simulation_data_list

        print ('Computing biased probabilities for each umbrella...')
        for simulation in self.simulation_data_list:
            simulation.compute_biased_prob()
        
        self.xedges = self.simulation_data_list[0].xedges
        self.xcenters = self.simulation_data_list[0].xcenters
        
    def compute_top_eqn(self):
        """
        Computes the top sum in eqn 6 in the PDF at the head of this repo.
        Stores the output in the WHAM object.
        """
        for index, simulation in enumerate(self.simulation_data_list):
            if index == 0:
                top_eqn = np.zeros_like(simulation.prob_biased)
                top_eqn += simulation.n_samples*simulation.prob_biased
            else:
                top_eqn += simulation.n_samples*simulation.prob_biased
        self.top_eqn = top_eqn

    def umbrella_potential(self, x, target_x, kx):
        """
        Computes umbrella potential bias (eqn 4 in the PDF at the head of this repo).
        """
        return 0.5*kx*np.square(x - target_x)

    def compute_bot_eqn(self):
        """
        Computes the bottom sum in eqn 6 in the PDF at the head of this repo.
        """
        bot_eqn = 0
        for index, simulation in enumerate(self.simulation_data_list):
            bot_eqn += simulation.n_samples*np.exp(simulation.beta*simulation.f)*np.exp(-1*simulation.beta*self.umbrella_potential(simulation.xcenters, simulation.target_x, simulation.kx))
        return bot_eqn

    def compute_trial_prob(self, top_eqn, bot_eqn):
        """
        Combines the outputs of "compute_top_eqn" and "compute_bot_eqn" to compute the full right hand side of eqn 6 in the PDF at the head of this repo.
        Returns a trial 1D probability distribution for WHAM equation SCF.
        """
        bot_eqn_invert = np.reciprocal(bot_eqn)
        prob = np.multiply(top_eqn,bot_eqn_invert)
        return prob

    def compute_trial_fs(self, prob):
        """
        Computes new f's by solving eqn 7 in the PDF at the head of this repo.
        Returns a list of new f constants for WHAM SCF.
        """
        for index, simulation in enumerate(self.simulation_data_list):
            Z = np.multiply(np.exp(-1*simulation.beta*self.umbrella_potential(simulation.xcenters, simulation.target_x, simulation.kx)), prob)
            int_Z = simps(Z, simulation.xcenters)
            f = -1*(1/simulation.beta)*np.log(int_Z)
            simulation.f = f
        return np.array([simulation.f for simulation in self.simulation_data_list])

    def WHAM_calc(self, max_iters=1000, threshold=1e-4):
        """
        Self-consistently solves the 1D WHAM equations (eqns 6 and 7) in order to stitch together sampling along the umbrella coordinate.
        Returns the 1D free energy surface on a grid.
        """
        self.compute_top_eqn()
        
        for step in range(max_iters):
            bot_eqn = self.compute_bot_eqn()
            trial_prob = self.compute_trial_prob(self.top_eqn, bot_eqn)
            initial_fs = np.array([simulation.f for simulation in self.simulation_data_list])
            trial_fs = self.compute_trial_fs(trial_prob)
            diff = initial_fs - trial_fs
            average_diff = np.mean(np.abs(diff))
            if step % 1000 == 0:
                print ('step %s, diff %s' % (step, average_diff))
            if average_diff < threshold:
                print ('convergence threshold met in %s steps, computing FES...' % step)
                self.final_prob = trial_prob
                break
            if step == max_iters-1:
                print ('max steps hit, final MAD was %s, now computing FES...' % average_diff)
                self.final_prob = trial_prob

        FES = -1*(1/self.beta)*np.log(self.final_prob)
        FES[np.isneginf(FES)] = 0
        FES -= np.min(FES)
        self.FES = FES

    def plot_FES(self, filename='1d_FES_WHAM.png', xlabel='X ($\AA$)', ylabel='Free Energy (kJ/mol)', title='Model 1D Potential', xmin=None, xmax=None, dpi=300):
        """
        Plots the 1D free energy surface computed via WHAM.
        """
        xmin = xmin or np.min(self.xedges) # Helpful but weird syntax from https://stackoverflow.com/questions/7371244
        xmax = xmax or np.max(self.xedges)

        fig, ax = plt.subplots()
        ax.plot(self.xcenters, self.FES, label='Computed Surface')
        xs = np.load('../build_potential/x_axis.npy')
        U = np.load('../build_potential/potential.npy')
        ax.plot(xs, U, label='Original Surface')
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title, xlim=(xmin,xmax))
        ax.legend()
        plt.savefig(filename, dpi=dpi)
        plt.close()
        
def main():
    """
    Computes the FES with respect to one coordinates using the weighted histogram analysis method (WHAM).
    Need to input value of beta (in appropriate energy units) and k (in kJ/mol/angstrom^2). Also need umbrella sampling coordinates and data sampled using umbrella sampling.
    """
    beta = 1/2.479 # KbT = 2.479 kJ/mol
    kx = 150000.0 # kJ/mol/angstrom^2
    xbins = 250

    home = os.getcwd()
    with open('../sample_potential/wham_metadata.txt','r') as meta:
        metadata = meta.readlines()
    colvar_paths = []
    targets = [] # umbrella centers
    for line in metadata:
        path = line.split()[0]
        target_x = float(line.split()[1])
        kx = float(line.split()[2])
        colvar_paths.append(path)
        targets.append(target_x)

    WHAM = wham_1d_reweighting(targets, kx, colvar_paths, beta=beta, xbins=xbins)
    WHAM.WHAM_calc(max_iters=10000, threshold=1e-4)
    WHAM.plot_FES() 

if __name__ == "__main__":
    main()



