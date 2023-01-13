import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
import time

class simulation_class:
    """
    Class for storing information relevant to a single umbrella simulation.
    """
    def __init__(self, COLVAR_file, beta, kx, ky, target_x, target_y, f=1.0):
        self.CV_data = np.genfromtxt(COLVAR_file)
        self.n_samples = self.CV_data.shape[0]
        self.beta = beta
        self.kx, self.ky = kx, ky
        self.target_x, self.target_y = target_x, target_y
        self.f = f
        print ('Reading in umbrella... n_samples: %s, target_x: %s, target_y: %s' % (self.n_samples, self.target_x, self.target_y))

    def compute_P_bias(self, x_col=1, y_col=2, xbins=150, ybins=150, xmin=-0.03, xmax=0.26, ymin=-0.1, ymax=0.1):
        """
        Computed umbrella-biased probability distribution without any reweighting.
        Biased probability is computed for a single umbrella simulation and attached to a simulation object as an instance variable.
        """
        histogram, xedges, yedges = np.histogram2d(x=self.CV_data[:, x_col], y=self.CV_data[:, y_col], bins=(xbins,ybins), density=True, range=[(xmin,xmax),(ymin,ymax)])
        xcenters = (xedges[:-1] + xedges[1:]) / 2
        ycenters = (yedges[:-1] + yedges[1:]) / 2
        self.prob_biased = histogram.T
        self.xedges, self.yedges = xedges, yedges
        self.xcenters, self.ycenters = xcenters, ycenters

class wham_2d_reweighting:
    """
    Class for computing reweighting of 2D WHAM simulations
    """
    def __init__(self, target_coords, kx, ky, COLVAR_paths, beta):
        """
        Load all the CV data into a list of simulation objects
        """
        self.beta = beta
        self.kx = kx
        self.ky = ky
        simulation_data_list = []
        for targets, COLVAR_path in zip(target_coords, COLVAR_paths):
            simulation_data = simulation_class(COLVAR_file=COLVAR_path, beta=self.beta, kx=self.kx, ky=self.ky, target_x=targets[0], target_y=targets[1])
            simulation_data_list.append(simulation_data)
        self.simulation_data_list = simulation_data_list

        for simulation in self.simulation_data_list:
            #print ('Computing biased probability for sim: %s' % simulation)
            simulation.compute_P_bias()
        
        self.xedges = self.simulation_data_list[0].xedges
        self.yedges = self.simulation_data_list[0].yedges
        self.xcenters = self.simulation_data_list[0].xcenters
        self.ycenters = self.simulation_data_list[0].ycenters
        
    def compute_top_eqn(self):
        """
        Computes the top sum in eqn 9 in the PDF at the head of this repo.
        Stores the output in the WHAM object.
        """
        for index, simulation in enumerate(self.simulation_data_list):
            if index == 0:
                top_eqn = np.zeros_like(simulation.prob_biased)
                top_eqn += simulation.n_samples*simulation.prob_biased
            else:
                top_eqn += simulation.n_samples*simulation.prob_biased
        self.top_eqn = top_eqn

    def umbrella_potential(self, xs, ys, target_x, target_y, kx, ky):
        """
        Computes umbrella potential bias
        """
        x = 0.5*kx*np.square(xs - target_x)
        y = 0.5*ky*np.square(ys - target_y)
        x_matrix = np.tile(x, (len(y), 1))
        y_matrix = np.tile(y, (len(x), 1))
        return x_matrix.T + y_matrix

    def compute_bot_eqn(self):
        """
        Computes the bottom sum in eqn 9 in the PDF at the head of this repo.
        """
        bot_eqn = 0
        for index, simulation in enumerate(self.simulation_data_list):
            bot_eqn += simulation.n_samples*np.exp(simulation.beta*simulation.f)*np.exp(-1*simulation.beta*self.umbrella_potential(simulation.xcenters, simulation.ycenters, simulation.target_x, simulation.target_y, simulation.kx, simulation.ky))
        return bot_eqn

    def compute_trial_prob(self, top_eqn, bot_eqn):
        """
        Combines the outputs of "compute_top_eqn" and "compute_bot_eqn" to compute the full right hand side of eqn 9 in the PDF at the head of this repo.
        Returns a trial 2D probability distribution for WHAM equation SCF.
        """
        bot_eqn_invert = np.reciprocal(bot_eqn)
        prob = np.multiply(top_eqn,bot_eqn_invert)
        return prob

    def compute_trial_fs(self, prob):
        """
        Computes new f's by solving eqn 10 in the PDF at the head of this repo. 
        Returns a list of new f constants for WHAM SCF.
        """
        for index, simulation in enumerate(self.simulation_data_list):
            Z = np.multiply(np.exp(-1*simulation.beta*self.umbrella_potential(simulation.xcenters, simulation.ycenters, simulation.target_x, simulation.target_y, simulation.kx, simulation.ky)), prob)
            int_Z = simps(simps(Z, simulation.xcenters), simulation.ycenters)
            f = -1*(1/simulation.beta)*np.log(int_Z)
            simulation.f = f
        return [simulation.f for simulation in self.simulation_data_list]

    def plot_biased_surface(self, filename='2d_biased_probability.png', cmap='jet', xlabel='X ($\AA$)', ylabel='Y ($\AA$)', title='Biased Probability', colorbar_label='Probability', dpi=300):
        """
        Plots the biased 2D probability distribtuion created from the weighted sum of samples from all umbrellas.
        """
        fig, ax = plt.subplots()
        c = ax.pcolormesh(self.xedges, self.yedges, self.top_eqn, cmap=cmap)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        fig.colorbar(c, ax=ax, label=colorbar_label)
        fig.tight_layout()
        plt.savefig(filename, dpi=dpi)
        plt.close()

    def WHAM_calc(self, max_iters=1000, threshold=1e-4, convergence=False, write_freq=100):
        """
        Self-consistently solves the WHAM equations (eqns 9 and 10 in the PDF at the head of this repo) in order to stitch together sampling from multiple umbrella simulations.
        Returns the 2D free energy surface on a grid.
        """
        self.compute_top_eqn()
        self.plot_biased_surface()
        
        for step in range(max_iters):
            bot_eqn = self.compute_bot_eqn()
            trial_prob = self.compute_trial_prob(self.top_eqn, bot_eqn)
            initial_fs = np.array([simulation.f for simulation in self.simulation_data_list])
            trial_fs = self.compute_trial_fs(trial_prob)
            diff = initial_fs - trial_fs
            average_diff = np.mean(np.abs(diff))
            print ('Step %s, diff %s' % (step, average_diff))
            if (step != 0) and (step % write_freq == 0) and convergence:
                FES = -1*(1/self.beta)*np.log(trial_prob)
                FES[np.isneginf(FES)] = 0
                FES -= np.min(FES)
                self.FES = FES
                if not os.path.exists('convergence'): os.mkdir('convergence')
                self.plot_FES(filename = 'convergence/step_%s.png' % str(step).zfill(4))
            if average_diff < threshold:
                print ('Convergence threshold met in %s steps, computing FES...' % step)
                self.final_prob = trial_prob
                break
            if step == max_iters-1:
                print ('Max steps hit, computing FES...')
                self.final_prob = trial_prob

        FES = -1*(1/self.beta)*np.log(self.final_prob)
        FES[np.isneginf(FES)] = 0
        FES -= np.min(FES)
        self.FES = FES

    def plot_FES(self, filename='2d_FES_WHAM.png', cmap='jet', vmin=0.0, vmax=50.0, xlabel='X ($\AA$)', ylabel='Y ($\AA$)', title='Model 2D Potential', xmin=-0.03, xmax=0.26, ymin=-0.1, ymax=0.1, colorbar_label='Free Energy (kJ/mol)', dpi=300):
        """
        Plots the 2D free energy surface computed via WHAM-2D.
        """
        fig, ax = plt.subplots()
        c = ax.pcolormesh(self.xedges, self.yedges, self.FES, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title, xlim=(xmin,xmax), ylim=(ymin,ymax))
        fig.colorbar(c, ax=ax, label=colorbar_label)
        fig.tight_layout()
        plt.savefig(filename, dpi=dpi)
        plt.close()
        
def main():
    """
    Computes the FES with respect to two coordinates using the weighted histrogram anaylsis method (WHAM).
    Need to input value of beta (in appropriate energy units), trajectory information sampled using umbrellas, as well as biasing/umbrella centers and force constants.

    This data is usually read in through a wham_metadata.txt file which stores force constants, umbrella centers, and a path to trajectory information for each window.
    """
    beta = 1/2.479 # KbT = 2.479 kJ/mol
    metadata_path = '../sample_potential'

    home = os.getcwd()
    with open(os.path.join(metadata_path, 'wham_metadata.txt'),'r') as meta:
        metadata = meta.readlines()
    target_coords = []
    COLVAR_paths = []
    for line in metadata:
        path = os.path.join(metadata_path, line.split()[0])
        target_x = float(line.split()[1])
        target_y = float(line.split()[2])
        kx = float(line.split()[3])
        ky = float(line.split()[4])
        target_coords.append([target_x, target_y])
        COLVAR_paths.append(path)

    WHAM = wham_2d_reweighting(target_coords, kx, ky, COLVAR_paths, beta=beta)
    WHAM.WHAM_calc(max_iters=1000, convergence=True)
    WHAM.plot_FES() 

if __name__ == "__main__":
    main()



