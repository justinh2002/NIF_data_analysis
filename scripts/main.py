#!/usr/bin/env python

from cfpack import *
import cfpack as cfp
import flashlib as fl
import numpy as np
import glob
import os
import dill
import matplotlib.pyplot as plt
import matplotlib
import h5py
matplotlib.use('pdf')
from matplotlib import rcParams
from scipy.special import sph_harm
from scipy.integrate import simps
import pyshtools as pysh
from joblib import Parallel, delayed  
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from numba import njit, prange  # Import Numba for JIT compilation


# latex
rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = r'\usepackage{bm}'
# basics
rcParams['lines.linewidth'] = 1.2
rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 17
rcParams['axes.linewidth'] = 0.8
# x-ticks
rcParams['xtick.top'] = True
rcParams['xtick.direction'] = 'in'
rcParams['xtick.minor.visible'] = True
rcParams['xtick.major.size'] = 6
rcParams['xtick.minor.size'] = 3
rcParams['xtick.major.width'] = 1.5
rcParams['xtick.minor.width'] = 1.5
rcParams['xtick.major.pad'] = 5
rcParams['xtick.minor.pad'] = 5
# y-ticks
rcParams['ytick.right'] = True
rcParams['ytick.direction'] = 'in'
rcParams['ytick.minor.visible'] = True
rcParams['ytick.major.size'] = 6
rcParams['ytick.minor.size'] = 3
rcParams['ytick.major.width'] = 1.5
rcParams['ytick.minor.width'] = 1.5
rcParams['ytick.major.pad'] = 5
rcParams['ytick.minor.pad'] = 5
# legend
rcParams['legend.fontsize'] = 20 #rcParams['font.size']
rcParams['legend.labelspacing'] = 0.2
rcParams['legend.loc'] = 'upper left'
rcParams['legend.frameon'] = False
# figure
rcParams['figure.figsize'] = (8.0, 5.0)
rcParams['figure.dpi'] = 150
rcParams['savefig.dpi'] = 200
rcParams['savefig.bbox'] = 'tight'

class NIF:

    def __init__(self):
        self.simpath = "../simdata/high_res/"
        self.bname = "NIF"
        self.plt_files = sorted(glob.glob(self.simpath + self.bname + "_hdf5_plt_cnt_????"))
        self.part_files = sorted(glob.glob(self.simpath + self.bname + "_hdf5_part_????"))
        self.initial_part_file = self.simpath + self.bname + "_hdf5_part_0000"

    def plt_fname(self, dump, withpath=True):
        fname = self.bname + "_hdf5_plt_cnt_{:04d}".format(dump)
        if withpath:
            fname = self.simpath + fname
        return fname

    def extract_sphere_data(self, sphere_center=[0, 0, 0], sphere_radius=0.05, files=None, verbose=1):
        if files is None:
            files = self.plt_files  # default is all plt files
        for fname in files:
            outfile = fname + "_sphere_dat.pkl"
            if not os.path.isfile(outfile):
                print("extracting data from '" + fname + "'...", color="green")
                gg = fl.FlashGG(fname, verbose=verbose - 1)
                datasets = ["dens", "velx", "vely", "velz", "magx", "magy", "magz", "pres"]
                cell_datas, cell_coords = gg.GetCellsInSphere(dsets=datasets, sphere_center=sphere_center,
                                                              sphere_radius=sphere_radius)
                class ret:
                    time = gg.scalars["time"]
                    dens = cell_datas[0]
                    velx = cell_datas[1]
                    vely = cell_datas[2]
                    velz = cell_datas[3]
                    magx = cell_datas[4]
                    magy = cell_datas[5]
                    magz = cell_datas[6]
                    pres = cell_datas[7]
                    coords = cell_coords
                with open(outfile, "wb") as fobj:
                    dill.dump(ret, fobj)
                    print("'" + outfile + "' written.", color="magenta")

    def read_sphere_data(self, dump):
        fname = self.plt_fname(dump) + "_sphere_dat.pkl"
        print("reading from '" + fname + "'...")
        ret = dill.load(open(fname, "rb"))
        return ret
        
    def process_sphere_data(self, files=None):
        # Process either a specified list of files or all plt files by default
        if files is None:
            files = self.plt_files  # Default to processing all available plt files

        for fname in files:
            pkl_filename = fname + "_sphere_dat.pkl"
            output_filename = fname + "_processed_data.npy"

            # Load the .pkl file
            if not os.path.isfile(pkl_filename):
                print(f"File {pkl_filename} does not exist, skipping.")
                continue

            try:
                with open(pkl_filename, "rb") as pkl_file:
                    data_dict = dill.load(pkl_file)
            except Exception as e:
                print(f"Failed to load {pkl_filename}: {e}")
                continue

            # Extract time information
            sim_time = data_dict.time

            # Access the data arrays directly from the loaded object
            dens = data_dict.dens  # Density
            velx = data_dict.velx  # Velocity in x
            vely = data_dict.vely  # Velocity in y
            velz = data_dict.velz  # Velocity in z
            magx = data_dict.magx  # Magnetic field in x
            magy = data_dict.magy  # Magnetic field in y
            magz = data_dict.magz  # Magnetic field in z
            coords = data_dict.coords  # Coordinates
            pressure = data_dict.pres  # Assuming pressure is included in your data

            # Calculate number of cells
            num_cells = len(dens)

            if num_cells == 0:
                print("No cells available in the data!")
                return None

            # If the grid is uniform, calculate the average spacing between cells
            # Compute the cell volume for a uniform grid
            dx = np.mean(np.diff(np.unique(coords[:, 0])))
            dy = np.mean(np.diff(np.unique(coords[:, 1])))
            dz = np.mean(np.diff(np.unique(coords[:, 2])))

            # Calculate cell volume assuming uniform spacing
            cell_volume = dx * dy * dz

            # Total volume occupied by the spherical region
            total_volume = num_cells * cell_volume

            # Mass averaging for velocities
            total_mass = np.sum(dens * cell_volume)
            volume_weighted_density = dens * cell_volume
            volume_averaged_density = np.sum(volume_weighted_density) / total_volume
            velx_mass_avg = np.sum(velx * dens * cell_volume) / total_mass
            vely_mass_avg = np.sum(vely * dens * cell_volume) / total_mass
            velz_mass_avg = np.sum(velz * dens * cell_volume) / total_mass

            # Now subtract the average from the velocity components to get the fluctuating part
            velx_prime = velx - velx_mass_avg  # u' (fluctuating x-component)
            vely_prime = vely - vely_mass_avg  # v' (fluctuating y-component)
            velz_prime = velz - velz_mass_avg  # w' (fluctuating z-component)
            
            

            # Calculate the total kinetic energy of the fluctuating velocity
            kinetic_energy_prime = 0.5 * dens * (velx_prime**2 + vely_prime**2 + velz_prime**2) * cell_volume
            total_kinetic_energy_prime = np.sum(kinetic_energy_prime)

            # Calculate magnetic energy (assuming μ0 = 1)
            magnetic_energy = (1 / (8 * np.pi)) * (magx**2 + magy**2 + magz**2) * cell_volume
            total_magnetic_energy = np.sum(magnetic_energy)
            maximum_magnetic_energy = np.max(magnetic_energy)

            # Volume averaging for magnetic field components
            magx_volume_avg = np.sum(magx * cell_volume) / total_volume
            magy_volume_avg = np.sum(magy * cell_volume) / total_volume
            magz_volume_avg = np.sum(magz * cell_volume) / total_volume

            # Calculate standard deviation of density
            density_std = np.std(dens)
            #pressure = 

            # Calculate Mach number: sqrt(velocity^2) / speed of sound
            gamma = 5 / 3  # Assuming an adiabatic index for ideal gas
            speed_of_sound = np.sqrt(gamma * pressure / dens)  # Speed of sound
            velocity_magnitude = np.sqrt(velx**2 + vely**2 + velz**2)  # Magnitude of velocity
            mach_number = velocity_magnitude / speed_of_sound

            # Calculate the average Mach number
            avg_mach_number = np.mean(mach_number)

            # Create a new dictionary with processed values and simulation time
            processed_data = {
                "time": sim_time,
                "velx_mass_avg": velx_mass_avg,
                "vely_mass_avg": vely_mass_avg,
                "velz_mass_avg": velz_mass_avg,
                "velx_prime": velx_prime,
                "vely_prime": vely_prime,
                "velz_prime": velz_prime,
                "magx": magx,
                "magy": magy,
                "magz": magz,
                "volume_averaged_density": volume_averaged_density,
                "density_std": density_std,  # Add standard deviation of density
                "total_kinetic_energy_prime": total_kinetic_energy_prime,
                "magx_volume_avg": magx_volume_avg,
                "magy_volume_avg": magy_volume_avg,
                "magz_volume_avg": magz_volume_avg,
                "total_magnetic_energy": total_magnetic_energy,
                "max_magnetic_energy": maximum_magnetic_energy,
                "avg_mach_number": avg_mach_number,  # Add Mach number
                "coords": coords
            }

            # Save the processed data into a new .npy file
            np.save(output_filename, processed_data)

            print(f"Processed data with time={sim_time} saved to {output_filename}")

            
    def save_quantities_against_time(self, filename='time_quantities_data.npz', files=None):
        if files is None:
            files = self.plt_files  # Default to processing all available plt files

        # Lists to store the data
        times = []
        densities = []
        density_stds = []
        kinetic_energies = []
        magnetic_energies = []
        velx_dispersion = []
        vely_dispersion = []
        velz_dispersion = []
        magx_dispersion = []
        magy_dispersion = []
        magz_dispersion = []
        max_magnetic_energy = []
        avg_mach_numbers = []

        # Extract data from each processed file
        for fname in files:
            processed_filename = fname + "_processed_data.npy"
            if not os.path.isfile(processed_filename):
                print(f"File {processed_filename} does not exist, skipping.")
                continue

            # Load processed data
            try:
                data = np.load(processed_filename, allow_pickle=True).item()
            except Exception as e:
                print(f"Failed to load {processed_filename}: {e}")
                continue

            # Append time and other quantities to lists
            times.append(data["time"])
            densities.append(data["volume_averaged_density"])
            density_stds.append(data["density_std"])  # Standard deviation of density
            kinetic_energies.append(data["total_kinetic_energy_prime"])
            magnetic_energies.append(data["total_magnetic_energy"])
            max_magnetic_energy.append(data["max_magnetic_energy"])
            avg_mach_numbers.append(data["avg_mach_number"])  # Average Mach number
            
            # Calculate the velocity dispersion (standard deviation) for each time step
            velx_dispersion.append(np.sqrt(np.mean(data["velx_prime"]**2)))
            vely_dispersion.append(np.sqrt(np.mean(data["vely_prime"]**2)))
            velz_dispersion.append(np.sqrt(np.mean(data["velz_prime"]**2)))

            # Calculate the magnetic field dispersion (standard deviation) for each time step
            magx_dispersion.append(np.std(data["magx"]))
            magy_dispersion.append(np.std(data["magy"]))
            magz_dispersion.append(np.std(data["magz"]))

        # Save the data to an .npz file
        np.savez(filename, times=times, densities=densities, density_stds=density_stds,
                kinetic_energies=kinetic_energies, magnetic_energies=magnetic_energies,
                velx_dispersion=velx_dispersion, vely_dispersion=vely_dispersion,
                velz_dispersion=velz_dispersion, magx_dispersion=magx_dispersion,
                magy_dispersion=magy_dispersion, magz_dispersion=magz_dispersion,
                max_magnetic_energy=max_magnetic_energy, avg_mach_numbers=avg_mach_numbers)

        print(f"Data saved to {filename}")

    def plot_quantities_against_time_from_file(self, filename='time_quantities_data.npz'):
        # Load the data from the .npz file
        try:
            data = np.load(filename)
        except Exception as e:
            print(f"Failed to load {filename}: {e}")
            return

        # Extract the data
        times = data['times']
        densities = data['densities']
        std_density = data['density_stds']
        
        kinetic_energies = data['kinetic_energies']
        magnetic_energies = data['magnetic_energies']
        velx_dispersion = data['velx_dispersion']
        vely_dispersion = data['vely_dispersion']
        velz_dispersion = data['velz_dispersion']
        max_magnetic_energy = data['max_magnetic_energy']
        magx_dispersion = data['magx_dispersion']
        magy_dispersion = data['magy_dispersion']
        magz_dispersion = data['magz_dispersion']
        mach_nu = data['avg_mach_numbers']
        
        sq_density = std_density ** 2 + densities ** 2
        C = sq_density/densities 
        
        mean_density = densities# gas clumping factor
        vel =  np.sqrt(velx_dispersion**2 + vely_dispersion**2 + velz_dispersion**2)
        alfven = np.sqrt(magx_dispersion**2+magy_dispersion**2+magz_dispersion)/np.sqrt(4*np.pi*mean_density)
        

        cs =  vel/mach_nu
        print(2*cs[375]**2/alfven[375])
        print(cs[375]/(100*1000))
        fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
        t = np.linspace(1e-7, 3e-7, 100)
        ekin = 1e3 * t ** (-1.2)

        # # First subplot: Kinetic Energy
                # plt.subplot(3, 2, 4)
        axes[0].semilogy(times * 10**9, mean_density, label="Velx",lw = 3)
        axes[0].set_ylim(1e-2,1e-0)
        ax2 = axes[0].twinx()  # Create a second y-axis sharing the same x-axis
        ax2.plot(times * 10**9, std_density, color='orange', label="Std Density", lw=3)
        ax2.set_ylabel(r"$\sigma_\rho \: [\mathrm{g} \: \mathrm{cm^{-1}}]$",color = 'orange',fontsize = 20)
        #axes[0].sxlabel("$t$ (ns)", fontsize=20)
        axes[0].set_ylabel(r"$ \langle  \rho \rangle \: [\mathrm{g} \: \mathrm{cm^{-1}}]$", fontsize=20,color = 'blue')
        plt.xlim(0,300)
        axes[1].semilogy(times * 10**9, kinetic_energies, label="Kinetic Energy", color='r', lw=3)
        axes[1].set_ylabel("$E_{\mathrm{kin}} \: [\mathrm{erg} \: \mathrm{g^{-1}}]$", fontsize=20)
        axes[1].set_xlim(0, 300)
        axes[1].set_ylim(1e-12, 1e12)
                # Adding an inset for the power law decay
        inset_ax = axes[1].inset_axes([0.6, 0.2, 0.35, 0.5])
        t = np.linspace(1.25e-7, 1.5e-7, 100)
        scaling_factor = 4e2
        scaling_factor_2 = 1e1
        ekin = scaling_factor * t ** (-1.2)

        ekin_2 = scaling_factor_2 * t ** (-10/7)

        inset_ax.loglog(t * 10**9, ekin, color='b', linestyle='--', label=r"$t^{-1.2}$")
        inset_ax.loglog(t * 10**9, ekin_2, color='g', linestyle='--', label=r"$t^{-10/7}$")
        inset_ax.loglog(times * 10 **9, kinetic_energies,color = 'r' ,lw = 3)
        inset_ax.set_xlim(100,300)
        inset_ax.set_ylim(1e9,1e12)
        inset_ax.set_xlabel("$t$ (ns)", fontsize=13)
        # Set tick font sizes
        inset_ax.tick_params(axis='both', which='major', labelsize=8)
        inset_ax.tick_params(axis='both', which='minor', labelsize=8)
        inset_ax.set_ylabel("$E_{\mathrm{kin}}$", fontsize=13)
        inset_ax.legend(fontsize = 8)
        
        # #axes[0].grid(True)

        # # Second subplot: Magnetic Energy
        axes[2].semilogy(times * 10**9, magnetic_energies, label="Magnetic Energy", color='g', lw=3)
        axes[2].set_xlabel("$t$ (ns)", fontsize=20)
        axes[2].set_ylabel("$E_{\mathrm{mag}} \: [\mathrm{G}]$", fontsize=20)
        axes[2].set_xlim(0, 300)
        # #axes[1].grid(True)
        # axes[2].semilogy(times * 10**9, np.array(magnetic_energies)/np.array(kinetic_energies), label="Magnetic Energy", color='navy', lw=3)
        # axes[2].set_xlabel("$t$ (ns)", fontsize=20)
        # axes[2].set_ylabel("$E_{\mathrm{mag}}/E_{\mathrm{kin}}$", fontsize=20)
        # axes[2].set_xlim(0, 300)
        # axes[2].set_ylim(1e-20, 1e-17)

        # Apply tight layout and show/save the figure
        plt.tight_layout()


        #plt.ylim(1e-2,1e-0)
        
       # plt.title("Velx dispersion vs Time")

        # plt.subplot(3, 2, 5)
        # plt.semilogy(times, vely_dispersion, label="Vely", color='c')
        # plt.xlabel("Time")
        # plt.ylabel("Vely dispersion")
        # plt.title("Vely dispersion vs Time")

        # plt.subplot(3, 2, 6)
        # plt.semilogy(times, velz_dispersion, label="Velz", color='m')
        # plt.xlabel("Time")
        # plt.ylabel("Velz dispersion")
        # plt.title("Velz dispersion vs Time")

        plt.tight_layout()
        plt.savefig('emag_ekin_decay.pdf')
        print("Plot saved as 'time_quantities.pdf'")
        
    def plot_more_quantities_against_time_from_file(self, filename='time_quantities_data.npz'):
        # Load the data from the .npz file
        try:
            data = np.load(filename)
        except Exception as e:
            print(f"Failed to load {filename}: {e}")
            return

        # Extract the data
        times = data['times']
        densities = data['densities']
        std_density = data['density_stds']
        
        kinetic_energies = data['kinetic_energies']
        magnetic_energies = data['magnetic_energies']
        velx_dispersion = data['velx_dispersion']
        vely_dispersion = data['vely_dispersion']
        velz_dispersion = data['velz_dispersion']
        max_magnetic_energy = data['max_magnetic_energy']
        magx_dispersion = data['magx_dispersion']
        magy_dispersion = data['magy_dispersion']
        magz_dispersion = data['magz_dispersion']
        sq_density = std_density ** 2 + densities ** 2
        C = sq_density/densities 
        
        mean_density = densities# gas clumping factor
        
        
        
        
        fig, axes = plt.subplots(3, 1, figsize=(8,12), sharex=True)
        t = np.linspace(1e-7, 3e-7, 100)
        ekin = 1e3 * t ** (-1.2)

        # # First subplot: Kinetic Energy
                # plt.subplot(3, 2, 4)
        axes[0].semilogy(times * 10**9, magnetic_energies/mean_density**2, label="Velx",lw = 3)
        axes[0].set_xlim(75,300)
        #ax2 = axes[0].twinx()  # Create a second y-axis sharing the same x-axis
        # ax2.plot(times * 10**9, std_density, color='orange', label="Std Density", lw=3)
        # ax2.set_ylabel(r"$\sigma_\rho \: [\mathrm{g} \: \mathrm{cm^{-1}}]$",color = 'orange',fontsize = 20)
        #axes[1].set_xlabel("$t$ (ns)", fontsize=20)
        axes[0].set_ylabel(r"$ E_{\mathrm{mag}}/\langle \rho \rangle^2 \: [ \mathrm{G} \: \mathrm{g}^{-2} \: \mathrm{cm^{6}}]$", fontsize=20)
        #plt.xlim(0,300)
        axes[1].semilogy(times * 10**9,( 1/mean_density**2) * magnetic_energies/kinetic_energies, label="Kinetic Energy", color='r', lw=3)
        axes[1].set_ylabel(r"$E_{\mathrm{mag}}/(\langle \rho \rangle^2 E_{\mathrm{kin}}) \: [\mathrm{g}^{-2} \: \mathrm{cm^{6}}]$", fontsize=20)
        axes[1].set_xlim(75, 300)
        axes[1].set_ylim(1e-19, 1e-14)
        # #axes[0].grid(True)

        # # Second subplot: Magnetic Energy
        axes[2].semilogy(times * 10**9, ( np.mean(mean_density)**2/mean_density**2) * (magnetic_energies)/(kinetic_energies), label="Magnetic Energy", color='g', lw=3)
        axes[2].set_xlabel("$t$ (ns)", fontsize=20)
        axes[2].set_ylabel(r"$ \bar{\rho}_t^2/ \langle \rho \rangle \times  E_{\mathrm{mag}}/E_{\mathrm{kin}}$", fontsize=20)
        axes[2].set_ylim(1e-21, 1e-16)
        
        axes[2].set_xlim(75, 300)
        # #axes[1].grid(True)
        # axes[2].semilogy(times * 10**9, np.array(magnetic_energies)/np.array(kinetic_energies), label="Magnetic Energy", color='navy', lw=3)
        # axes[2].set_xlabel("$t$ (ns)", fontsize=20)
        # axes[2].set_ylabel("$E_{\mathrm{mag}}/E_{\mathrm{kin}}$", fontsize=20)
        # axes[2].set_xlim(0, 300)
        # axes[2].set_ylim(1e-20, 1e-17)

        # Apply tight layout and show/save the figure
        plt.tight_layout()


        #plt.ylim(1e-2,1e-0)
        
       # plt.title("Velx dispersion vs Time")

        # plt.subplot(3, 2, 5)
        # plt.semilogy(times, vely_dispersion, label="Vely", color='c')
        # plt.xlabel("Time")
        # plt.ylabel("Vely dispersion")
        # plt.title("Vely dispersion vs Time")

        # plt.subplot(3, 2, 6)
        # plt.semilogy(times, velz_dispersion, label="Velz", color='m')
        # plt.xlabel("Time")
        # plt.ylabel("Velz dispersion")
        # plt.title("Velz dispersion vs Time")

        plt.tight_layout()
        plt.savefig('emagekin_rho.pdf')
        print("Plot saved as 'time_quantities.pdf'")
        
    def plot_even_more_quantities_against_time_from_file(self, filename='time_quantities_data.npz'):
        # Load the data from the .npz file
        try:
            data = np.load(filename)
        except Exception as e:
            print(f"Failed to load {filename}: {e}")
            return

        # Extract the data
        times = data['times']
        densities = data['densities']
        std_density = data['density_stds']
        
        kinetic_energies = data['kinetic_energies']
        magnetic_energies = data['magnetic_energies']
        velx_dispersion = data['velx_dispersion']
        vely_dispersion = data['vely_dispersion']
        velz_dispersion = data['velz_dispersion']
        magx_dispersion = data['magx_dispersion']
        magy_dispersion = data['magy_dispersion']
        magz_dispersion = data['magz_dispersion']
        max_magnetic_energy = data['max_magnetic_energy']
        
        sq_density = std_density ** 2 + densities ** 2
        C = sq_density/densities 
        
        mean_density = densities# gas clumping factor
        
        
        
        
        fig, axes = plt.subplots(2, 1, figsize=(8,8), sharex=True)
        t = np.linspace(1e-7, 3e-7, 100)
        ekin = 1e3 * t ** (-1.2)

        # # First subplot: Kinetic Energy
                # plt.subplot(3, 2, 4)
        axes[0].semilogy(times * 10**9, velx_dispersion , label=r"$\sigma_{v_x}$",lw = 3,color = 'blueviolet')
        axes[0].semilogy(times * 10**9, vely_dispersion , label=r"$\sigma_{v_y}$",lw = 3,color = 'aqua',alpha = 0.8)
        axes[0].semilogy(times * 10**9, velz_dispersion , label=r"$\sigma_{v_z}$",lw = 3,color = 'maroon')
        axes[0].set_xlim(75,300)
        axes[0].set_ylim(1e5,3e6)
        axes[0].legend(loc = 'best')
        axes[0].set_ylabel(r"$\sigma_{v_{i}} \: [\textrm{cm} \: \textrm{s}^{-1}]$", fontsize=20)
        #ax2 = axes[0].twinx()  # Create a second y-axis sharing the same x-axis
        # ax2.plot(times * 10**9, std_density, color='orange', label="Std Density", lw=3)
        # ax2.set_ylabel(r"$\sigma_\rho \: [\mathrm{g} \: \mathrm{cm^{-1}}]$",color = 'orange',fontsize = 20)
        #axes[1].set_xlabel("$t$ (ns)", fontsize=20)
        #axes[0].set_ylabel(r"$ E_{\mathrm{mag}}/\langle \rho \rangle^2 \: [ \mathrm{G} \: \mathrm{g}^{-2} \: \mathrm{cm^{6}}]$", fontsize=20)
        #plt.xlim(0,300)
        axes[1].semilogy(times * 10**9, magx_dispersion , label=r"$\sigma_{B_{x}}$",lw = 3,color = 'blueviolet')
        axes[1].semilogy(times * 10**9, magy_dispersion , label=r"$\sigma_{B_{y}}$",lw = 3,color = 'aqua',alpha = 0.8)
        axes[1].semilogy(times * 10**9, magz_dispersion , label=r"$\sigma_{B_{z}}$",lw = 3,color = 'maroon')
        axes[1].set_ylabel(r"$\sigma_{B_{i}} \: [\mathrm{G}]$", fontsize=20)
        axes[1].set_xlim(75, 300)
        axes[1].set_ylim(1e-5,1e-3)
        #axes[1].set_ylim(1e-19, 1e-14)
        # #axes[0].grid(True)

        # # # Second subplot: Magnetic Energy
        # axes[2].semilogy(times * 10**9, np.array(magnetic_energies)/np.array(kinetic_energies), label="Magnetic Energy", color='g', lw=3)
        axes[1].set_xlabel("$t$ (ns)", fontsize=20)
        # axes[2].set_ylabel("$E_{\mathrm{mag}}/E_{\mathrm{kin}}$", fontsize=20)
        # axes[2].set_ylim(1e-20, 1e-17)
        plt.legend(loc = 'lower right')
        # axes[2].set_xlim(75, 300)
        # #axes[1].grid(True)
        # axes[2].semilogy(times * 10**9, np.array(magnetic_energies)/np.array(kinetic_energies), label="Magnetic Energy", color='navy', lw=3)
        # axes[2].set_xlabel("$t$ (ns)", fontsize=20)
        # axes[2].set_ylabel("$E_{\mathrm{mag}}/E_{\mathrm{kin}}$", fontsize=20)
        # axes[2].set_xlim(0, 300)
        # axes[2].set_ylim(1e-20, 1e-17)

        # Apply tight layout and show/save the figure
        plt.tight_layout()


        #plt.ylim(1e-2,1e-0)
        
       # plt.title("Velx dispersion vs Time")

        # plt.subplot(3, 2, 5)
        # plt.semilogy(times, vely_dispersion, label="Vely", color='c')
        # plt.xlabel("Time")
        # plt.ylabel("Vely dispersion")
        # plt.title("Vely dispersion vs Time")

        # plt.subplot(3, 2, 6)
        # plt.semilogy(times, velz_dispersion, label="Velz", color='m')
        # plt.xlabel("Time")
        # plt.ylabel("Velz dispersion")
        # plt.title("Velz dispersion vs Time")

        plt.tight_layout()
        plt.savefig('sigma_vb.pdf')
        print("Plot saved as 'time_quantities.pdf'")


        
    def read_and_print_sphere_data(self, files=None):
        if files is None:
            files = self.plt_files  # Default to reading all available plt files

        for fname in files:
            pkl_filename = fname + "_sphere_dat.pkl"

            # Load the .pkl file
            if not os.path.isfile(pkl_filename):
                print(f"File {pkl_filename} does not exist, skipping.")
                continue

            try:
                with open(pkl_filename, "rb") as pkl_file:
                    data_dict = dill.load(pkl_file)
            except Exception as e:
                print(f"Failed to load {pkl_filename}: {e}")
                continue

            # Print out the data
            print(f"Data from file: {pkl_filename}")
            print(f"Time: {data_dict.time}")
            print(f"Density shape: {data_dict.dens.shape}")
            print(f"Velocities shapes: velx: {data_dict.velx.shape}, vely: {data_dict.vely.shape}, velz: {data_dict.velz.shape}")
            print(f"Magnetic fields shapes: magx: {data_dict.magx.shape}, magy: {data_dict.magy.shape}, magz: {data_dict.magz.shape}")
            print(f"Coordinates shape: {data_dict.coords.shape}")

            # Optionally, print out some specific data values (for example, first 5 entries)
            print("\nSample Data:")
            print(f"Density (first 5): {data_dict.dens[:5]}")
            print(f"Velx (first 5): {data_dict.velx[:5]}")
            print(f"Magx (first 5): {data_dict.magx[:5]}")
            print("-" * 50)
                

    def plot_magnetic_field_pdfs(self, start_index=None, end_index=None, files=None):
        if files is None:
            files = self.plt_files  # Default to processing all available plt files

        # If no specific indices are provided, process only one file
        if start_index is None:
            start_index = 0  # Default to the first file
        if end_index is None:
            end_index = start_index  # Process just one file if no range is specified

        # Lists to store all magnetic field data
        magx_values = []
        magy_values = []
        magz_values = []

        # Loop over the specified file range
        for i in range(start_index, end_index + 1):
            fname = files[i]
            pkl_filename = fname + "_sphere_dat.pkl"
            output_filename = fname + "_processed_data.npy"

            # Check if .pkl file exists
            if not os.path.isfile(pkl_filename):
                print(f"File {pkl_filename} does not exist, skipping.")
                continue

            # Load the .pkl file
            try:
                with open(pkl_filename, "rb") as pkl_file:
                    data_dict = dill.load(pkl_file)
            except Exception as e:
                print(f"Failed to load {pkl_filename}: {e}")
                continue

            # Access the data arrays directly from the loaded object
            magx = data_dict.magx  # Magnetic field in X direction
            magy = data_dict.magy  # Magnetic field in Y direction
            magz = data_dict.magz  # Magnetic field in Z direction

            # Append the magnetic field data to the corresponding lists
            magx_values.extend(magx)
            magy_values.extend(magy)
            magz_values.extend(magz)

        # Convert to numpy arrays for further processing if needed
        magx_values = np.array(magx_values)
        magy_values = np.array(magy_values)
        magz_values = np.array(magz_values)

        # Plot PDFs of magnetic fields
        plt.figure(figsize=(12, 8))

        plt.subplot(3, 1, 1)
        plt.hist(magx_values, bins=500,density= True, alpha=0.7, label='MagX')
        plt.xlabel('MagX')
        plt.ylabel('Probability Density')
        plt.title('PDF of Magnetic Field (X Component)')
        plt.grid(True)

        plt.subplot(3, 1, 2)
        plt.hist(magy_values, bins=500, density=True, alpha=0.7, label='MagY', color='g')
        plt.xlabel('MagY')
        plt.ylabel('Probability Density')
        plt.title('PDF of Magnetic Field (Y Component)')
        plt.grid(True)

        plt.subplot(3, 1, 3)
        plt.hist(magz_values, bins=500, density=True, alpha=0.7, label='MagZ', color='r')
        plt.xlabel('MagZ')
        plt.ylabel('Probability Density')
        plt.title('PDF of Magnetic Field (Z Component)')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f'magnetic_field_pdfs_{start_index}_to_{end_index}.pdf')
        plt.close()
        
    def spherical_divergence(self, r, theta, phi, A_r, A_theta, A_phi):
        """Compute the divergence of a vector field in spherical coordinates."""
        # Use numpy's gradient function to approximate derivatives
        dr = np.gradient(r)
        dtheta = np.gradient(theta)
        dphi = np.gradient(phi)

        # Compute divergence using spherical coordinate formula
        div_r = (1 / r**2) * np.gradient(r**2 * A_r, dr)
        div_theta = (1 / (r * np.sin(theta))) * np.gradient(A_theta * np.sin(theta), dtheta)
        div_phi = (1 / (r * np.sin(theta))) * np.gradient(A_phi, dphi)

        return div_r + div_theta + div_phi

    def spherical_gradient(self, r, theta, phi, f):
        """Compute the gradient of a scalar field in spherical coordinates."""
        # Use numpy's gradient function to approximate derivatives
        dr = np.gradient(r)
        dtheta = np.gradient(theta)
        dphi = np.gradient(phi)

        # Compute the gradient
        grad_r = np.gradient(f, dr)
        grad_theta = np.gradient(f, dtheta) / r
        grad_phi = np.gradient(f, dphi) / (r * np.sin(theta))

        return grad_r, grad_theta, grad_phi

    def spherical_curl(self, r, theta, phi, A_r, A_theta, A_phi):
        """Compute the curl of a vector field in spherical coordinates."""
        # Use numpy's gradient function to approximate derivatives
        dr = np.gradient(r)
        dtheta = np.gradient(theta)
        dphi = np.gradient(phi)

        # Curl components in spherical coordinates
        curl_r = (1 / (r * np.sin(theta))) * (
            np.gradient(A_phi * np.sin(theta), dtheta) - np.gradient(A_theta, dphi)
        )
        curl_theta = (1 / r) * (
            (1 / np.sin(theta)) * np.gradient(A_r, dphi) - np.gradient(r * A_phi, dr)
        )
        curl_phi = (1 / r) * (np.gradient(r * A_theta, dr) - np.gradient(A_r, dtheta))

        return curl_r, curl_theta, curl_phi
    def compute_energy_spectra(self, dump, l_max=30, num_bins_r=50, num_bins_theta=90, num_bins_phi=180):
        """
        Compute kinetic and magnetic energy spectra using vector spherical harmonics for a given dump.

        Parameters:
        - dump: The dump number to process.
        - l_max: Maximum spherical harmonic degree.
        - num_bins_r, num_bins_theta, num_bins_phi: Number of bins in r, theta, phi directions.
        """
        import numpy as np
        from scipy.special import sph_harm
        from scipy.integrate import simps
        import matplotlib.pyplot as plt

        # Read the sphere data for the given dump
        data = self.read_sphere_data(dump)

        # Extract velocities, magnetic fields, and positions
        velx = data.velx
        vely = data.vely
        velz = data.velz
        magx = data.magx
        magy = data.magy
        magz = data.magz
        coords = data.coords.T  # positions (n_points, 3)

        # Convert positions to spherical coordinates
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r)  # polar angle [0, pi]
        phi = np.arctan2(y, x)    # azimuthal angle [-pi, pi]
        phi[phi < 0] += 2 * np.pi  # Convert phi to [0, 2pi]

        # Handle invalid values in r, theta, phi
        valid_mask = np.isfinite(r) & (r > 0) & np.isfinite(theta) & np.isfinite(phi)
        r = r[valid_mask]
        theta = theta[valid_mask]
        phi = phi[valid_mask]
        velx = velx[valid_mask]
        vely = vely[valid_mask]
        velz = velz[valid_mask]
        magx = magx[valid_mask]
        magy = magy[valid_mask]
        magz = magz[valid_mask]

        # Convert velocities and magnetic fields to spherical components
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        # Velocities
        u_r = velx * sin_theta * cos_phi + vely * sin_theta * sin_phi + velz * cos_theta
        u_theta = velx * cos_theta * cos_phi + vely * cos_theta * sin_phi - velz * sin_theta
        u_phi = -velx * sin_phi + vely * cos_phi

        # Magnetic fields
        B_r = magx * sin_theta * cos_phi + magy * sin_theta * sin_phi + magz * cos_theta
        B_theta = magx * cos_theta * cos_phi + magy * cos_theta * sin_phi - magz * sin_theta
        B_phi = -magx * sin_phi + magy * cos_phi

        # Define number of bins
        num_bins_r = num_bins_r
        num_bins_theta = num_bins_theta
        num_bins_phi = num_bins_phi

        # Compute bin edges
        r_bins = np.linspace(r.min(), r.max(), num_bins_r + 1)
        theta_bins = np.linspace(0, np.pi, num_bins_theta + 1)
        phi_bins = np.linspace(0, 2*np.pi, num_bins_phi + 1)

        # Bin indices
        r_indices = np.digitize(r, r_bins) - 1
        theta_indices = np.digitize(theta, theta_bins) - 1
        phi_indices = np.digitize(phi, phi_bins) - 1

        # Clip indices to ensure they are within valid range
        r_indices = np.clip(r_indices, 0, num_bins_r - 1)
        theta_indices = np.clip(theta_indices, 0, num_bins_theta - 1)
        phi_indices = np.clip(phi_indices, 0, num_bins_phi - 1)

        # Shape of the grid
        shape = (num_bins_r, num_bins_theta, num_bins_phi)

        # Accumulate sums and counts
        flat_indices = np.ravel_multi_index((r_indices, theta_indices, phi_indices), dims=shape)

        def accumulate_sums(values):
            sums_flat = np.bincount(flat_indices, weights=values, minlength=np.prod(shape))
            return sums_flat.reshape(shape)

        counts_flat = np.bincount(flat_indices, minlength=np.prod(shape))
        counts = counts_flat.reshape(shape)

        u_r_sum = accumulate_sums(u_r)
        u_theta_sum = accumulate_sums(u_theta)
        u_phi_sum = accumulate_sums(u_phi)
        B_r_sum = accumulate_sums(B_r)
        B_theta_sum = accumulate_sums(B_theta)
        B_phi_sum = accumulate_sums(B_phi)

        # Compute averages
        def compute_average(sums):
            return np.divide(sums, counts, out=np.zeros_like(sums), where=counts != 0)

        u_r_grid = compute_average(u_r_sum)
        u_theta_grid = compute_average(u_theta_sum)
        u_phi_grid = compute_average(u_phi_sum)
        B_r_grid = compute_average(B_r_sum)
        B_theta_grid = compute_average(B_theta_sum)
        B_phi_grid = compute_average(B_phi_sum)

        # Now proceed with the rest of your code
        # Create grids for r, theta, phi bin centers
        r_bin_centers = (r_bins[:-1] + r_bins[1:]) / 2
        theta_bin_centers = (theta_bins[:-1] + theta_bins[1:]) / 2
        phi_bin_centers = (phi_bins[:-1] + phi_bins[1:]) / 2

        # Create meshgrids for Theta and Phi
        Theta_grid, Phi_grid = np.meshgrid(theta_bin_centers, phi_bin_centers, indexing='ij')
        # Theta_grid and Phi_grid have shape (num_bins_theta, num_bins_phi)

        # Precompute the spherical harmonics
        def precompute_spherical_harmonics(l_max, theta_grid, phi_grid):
            Y_lms = {}
            for l in range(1, l_max+1):
                for m in range(-l, l+1):
                    Y_lm = sph_harm(m, l, phi_grid, theta_grid)  # Ensure phi and theta are arrays of the same shape
                    Y_lms[(l, m)] = Y_lm
            return Y_lms

        print("Precomputing spherical harmonics...")
        Y_lms = precompute_spherical_harmonics(l_max, Theta_grid, Phi_grid)

        # Function to perform the vector spherical harmonic decomposition
        def vector_spherical_harmonic_decomposition(u_r_grid, u_theta_grid, u_phi_grid, l_max):
            num_r = u_r_grid.shape[0]

            # Initialize coefficients
            coeffs_u_r = np.zeros((l_max+1, 2*l_max+1, num_r), dtype=complex)
            coeffs_u_S = np.zeros((l_max+1, 2*l_max+1, num_r), dtype=complex)
            coeffs_u_T = np.zeros((l_max+1, 2*l_max+1, num_r), dtype=complex)

            # Loop over radial positions
            for idx_r in range(num_r):
                # Extract data at fixed radius
                ur = u_r_grid[idx_r, :, :]  # Shape (num_bins_theta, num_bins_phi)
                utheta = u_theta_grid[idx_r, :, :]
                uphi = u_phi_grid[idx_r, :, :]

                # Loop over degrees and orders
                for l in range(1, l_max+1):
                    for m in range(-l, l+1):
                        idx_m = m + l

                        # Get the spherical harmonics at current l and m
                        Y_lm = Y_lms[(l, m)]  # Shape (num_bins_theta, num_bins_phi)

                        # Compute gradient of Y_lm
                        dtheta = theta_bin_centers[1] - theta_bin_centers[0]
                        dphi = phi_bin_centers[1] - phi_bin_centers[0]

                        dY_dtheta = np.gradient(Y_lm, axis=0) / dtheta  # Shape (num_bins_theta, num_bins_phi)
                        dY_dphi = np.gradient(Y_lm, axis=1) / dphi  # Shape (num_bins_theta, num_bins_phi)
                        dY_dphi = dY_dphi / np.sin(Theta_grid)  # Adjust for sin(theta)

                        # Compute the inner products over theta and phi
                        sin_theta = np.sin(Theta_grid)
                        integrand_r = ur * np.conj(Y_lm) * sin_theta
                        coeff_r = simps(simps(integrand_r, phi_bin_centers), theta_bin_centers)
                        coeffs_u_r[l, idx_m, idx_r] = coeff_r

                        # Spheroidal component
                        integrand_S_theta = utheta * np.conj(dY_dtheta) * sin_theta
                        integrand_S_phi = uphi * np.conj(dY_dphi) * sin_theta
                        coeff_S = simps(simps(integrand_S_theta + integrand_S_phi, phi_bin_centers), theta_bin_centers)
                        coeffs_u_S[l, idx_m, idx_r] = coeff_S

                        # Toroidal component
                        integrand_T_theta = utheta * np.conj(-dY_dphi) * sin_theta
                        integrand_T_phi = uphi * np.conj(dY_dtheta) * sin_theta
                        coeff_T = simps(simps(integrand_T_theta + integrand_T_phi, phi_bin_centers), theta_bin_centers)
                        coeffs_u_T[l, idx_m, idx_r] = coeff_T

            return coeffs_u_r, coeffs_u_S, coeffs_u_T

        # Decompose the velocity field
        print("Decomposing velocity field...")
        coeffs_u_r, coeffs_u_S, coeffs_u_T = vector_spherical_harmonic_decomposition(u_r_grid, u_theta_grid, u_phi_grid, l_max)

        # Similarly decompose the magnetic field
        print("Decomposing magnetic field...")
        coeffs_B_r, coeffs_B_S, coeffs_B_T = vector_spherical_harmonic_decomposition(B_r_grid, B_theta_grid, B_phi_grid, l_max)

        # Compute energy per mode
        def compute_energy_per_mode(coeffs_r, coeffs_S, coeffs_T, r_values):
            E_l_m = np.zeros((coeffs_r.shape[0], coeffs_r.shape[1]))
            for l in range(1, coeffs_r.shape[0]):
                for idx_m in range(2*l + 1):
                    # Square of the coefficient magnitude integrated over radius
                    coeff_squared = (np.abs(coeffs_r[l, idx_m, :])**2 +
                                    np.abs(coeffs_S[l, idx_m, :])**2 +
                                    np.abs(coeffs_T[l, idx_m, :])**2)
                    # Integrate over radius
                    E_l_m[l, idx_m] = simps(coeff_squared * r_values**2, r_values)
            return E_l_m

        print("Computing energy per mode for velocity field...")
        E_l_m_u = compute_energy_per_mode(coeffs_u_r, coeffs_u_S, coeffs_u_T, r_bin_centers)
        print("Computing energy per mode for magnetic field...")
        E_l_m_B = compute_energy_per_mode(coeffs_B_r, coeffs_B_S, coeffs_B_T, r_bin_centers)

        # Sum over m to get total energy per degree l
        E_l_u = np.sum(E_l_m_u, axis=1)
        E_l_B = np.sum(E_l_m_B, axis=1)

        # Map l to wavenumber k
        R = r.max()  # Or use self.R if defined
        k_l = np.array([l / R for l in range(l_max+1)])

        # Save the energy spectra
        output_filename = self.plt_fname(dump) + "_energy_spectra.npz"
        np.savez(output_filename, k_l=k_l[1:], E_l_u=E_l_u[1:], E_l_B=E_l_B[1:])
        print(f"Energy spectra saved to '{output_filename}'")

        # Plot the energy spectra
        plt.figure(figsize=(10, 6))
        plt.loglog(k_l[1:], E_l_u[1:], label='Kinetic Energy Spectrum')
        plt.loglog(k_l[1:], E_l_B[1:], label='Magnetic Energy Spectrum')
        plt.xlabel('Wavenumber $k$')
        plt.ylabel('Energy $E(k)$')
        plt.title('Energy Spectra')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.savefig(self.plt_fname(dump) + "_energy_spectra.pdf")
        plt.close()
        print(f"Energy spectra plot saved as '{self.plt_fname(dump) + '_energy_spectra.pdf'}'")




    def process_magnetic_field_divergence_curl(self, files=None):
        """Compute divergence and curl of the magnetic field over a range of files."""
        if files is None:
            files = self.plt_files  # Default to processing all available plt files

        for fname in files:
            pkl_filename = fname + "_sphere_dat.pkl"

            if not os.path.isfile(pkl_filename):
                print(f"File {pkl_filename} does not exist, skipping.")
                continue

            # Load the .pkl file
            try:
                with open(pkl_filename, "rb") as pkl_file:
                    data_dict = dill.load(pkl_file)
            except Exception as e:
                print(f"Failed to load {pkl_filename}: {e}")
                continue

            # Access the data arrays directly from the loaded object
            magx = data_dict.magx  # Magnetic field in X direction
            magy = data_dict.magy  # Magnetic field in Y direction
            magz = data_dict.magz  # Magnetic field in Z direction
            coords = data_dict.coords

            # Assume `coords` contains (r, theta, phi)
            r, theta, phi = coords[:, 0], coords[:, 1], coords[:, 2]

            # Convert Cartesian magnetic fields (magx, magy, magz) to spherical components
            A_r, A_theta, A_phi = self.cartesian_to_spherical_vector(coords, magx, magy, magz)

            # Compute divergence and curl in spherical coordinates
            divergence = self.spherical_divergence(r, theta, phi, A_r, A_theta, A_phi)
            curl_r, curl_theta, curl_phi = self.spherical_curl(r, theta, phi, A_r, A_theta, A_phi)

            # Store or process divergence and curl for further analysis

    def cartesian_to_spherical_vector(self, coords, vx, vy, vz):
        """Convert a vector field from Cartesian to spherical coordinates."""
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r)  # polar angle
        phi = np.arctan2(y, x)    # azimuthal angle

        # Convert Cartesian components to spherical components
        A_r = vx * np.sin(theta) * np.cos(phi) + vy * np.sin(theta) * np.sin(phi) + vz * np.cos(theta)
        A_theta = vx * np.cos(theta) * np.cos(phi) + vy * np.cos(theta) * np.sin(phi) - vz * np.sin(theta)
        A_phi = -vx * np.sin(phi) + vy * np.cos(phi)

        return A_r, A_theta, A_phi
    
    def _read_particles(self, filen):
        """Helper method to read particle data from a file."""
        g = fl.Particles(filen)
        return g.read()

    def extract_time(self, fname):
        """Extract the simulation time from a given HDF5 file."""
        gg = fl.FlashGG(fname, verbose=0)
        return gg.scalars["time"]
        
    def track_bounding_sphere_in_time(self, center, radius, output_filename="tracked_sphere_data.pkl"):
        """
        Track data within a bounding sphere over time and store the relevant data.

        Parameters:
        - center: The initial center of the bounding sphere as [x, y, z].
        - radius: The radius of the bounding sphere.
        - output_filename: The filename to save the tracked data.
        - initial_tags_filename: Optional, path to an HDF5 file containing precomputed initial tags.
        """
        tracked_data = {}
        initial_tags = None  # To store the tags of particles initially within the bounding sphere
         # Fixed particle file for initial tag calculation

        # Ensure both part_files and plt_files are aligned
        if len(self.part_files) != len(self.plt_files):
            print("Error: Number of particle files does not match number of plot files.")
            return

        # If initial_tags is None (i.e., not provided by a file), compute the initial tags from the default particle file
        if initial_tags is None:
            print(f"Computing initial tags from '{self.initial_part_file}'...")
            my_dict = self._read_particles(self.initial_part_file)
            x, y, z = my_dict['posx'], my_dict['posy'], my_dict['posz']
            tags = my_dict['tag']

            # Calculate the distance from the center and find particles within the sphere
            distances = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
            mask = distances <= radius
            initial_tags = tags[mask]
            print(f"Initial set of particles within sphere: {len(initial_tags)} particles selected.")

            # Save initial tags to an HDF5 file
            with h5py.File("particle_mark_tags.h5", 'w') as f:
                f.create_dataset("tags", data=initial_tags)
            print(f"Initial tags saved to 'particle_mark_tags.h5'.")

        # Loop through all particle files and corresponding plot files
        for part_file, plt_file in zip(self.part_files, self.plt_files):
            print(f"Processing file {part_file} for bounding sphere tracking...")

            # Extract the time from the corresponding plot file
            sim_time = self.extract_time(plt_file)

            # Read particle data from the current dump
            my_dict = self._read_particles(part_file)

            x, y, z = my_dict['posx'], my_dict['posy'], my_dict['posz']
            tags = my_dict['tag']
            magx, magy, magz = my_dict['magx'], my_dict['magy'], my_dict['magz']
            velx, vely, velz = my_dict['velx'], my_dict['vely'], my_dict['velz']
            dens = my_dict['dens']
            pres = my_dict['pres']

            # Track only the particles with the initial tags
            mask = np.isin(tags, initial_tags)

            # Extract data for the tracked particles
            tracked_x, tracked_y, tracked_z = x[mask], y[mask], z[mask]
            tracked_tags = tags[mask]
            tracked_magx, tracked_magy, tracked_magz = magx[mask], magy[mask], magz[mask]
            tracked_velx, tracked_vely, tracked_velz = velx[mask], vely[mask], velz[mask]
            tracked_dens = dens[mask]
            tracked_pres = pres[mask]

            # Store the data for this timestep
            tracked_data[sim_time] = {
                "x": tracked_x,
                "y": tracked_y,
                "z": tracked_z,
                "tags": tracked_tags,
                "magx": tracked_magx,
                "magy": tracked_magy,
                "magz": tracked_magz,
                "velx": tracked_velx,
                "vely": tracked_vely,
                "velz": tracked_velz,
                "dens": tracked_dens,
                "pres": tracked_pres
            }

        # Save the tracked data to a file for further analysis
        with open(output_filename, "wb") as f:
            dill.dump(tracked_data, f)

        print(f"Tracked data saved to '{output_filename}'")

    def read_tracked_data(self, filename):
        """Load the tracked data from a file."""
        if not os.path.isfile(filename):
            print(f"File {filename} does not exist.")
            return None

        with open(filename, "rb") as f:
            tracked_data = dill.load(f)
        
        return tracked_data

    def save_tracked_quantities(self, tracked_data, output_filename="tracked_quantities_data.npz"):
        """Calculate and save various quantities over time for the tracked particles."""
        times = sorted(tracked_data.keys())  # Extract and sort times
        mag_squared_means = []
        magnetic_energies = []
        kinetic_energies = []
        densities = []
        std_velx = []
        std_vely = []
        std_velz = []
        std_magx = []
        std_magy = []
        std_magz = []

        # Loop over sorted times and calculate the desired quantities
        for time in times:
            data = tracked_data[time]

            # Magnetic field calculations
            mag_mean = [data["magx"].mean(), data["magy"].mean(), data["magz"].mean()]
            mag_squared_mean = np.dot(mag_mean, mag_mean)
            mag_squared_means.append(mag_squared_mean)
            
            magy_max = np.max(data["magy"])
            magx_max = np.max(data["magx"])
            magz_max = np.max(data["magz"])
            

            # Calculate turbulent magnetic energy based on standard deviation of magnetic field components
            std_magx = np.std(data["magx"])
            std_magy = np.std(data["magy"])
            std_magz = np.std(data["magz"])
            turbulent_magnetic_energy = 0.5 * (std_magx**2 + std_magy**2 + std_magz**2)  # RMS magnetic energy
            magnetic_energies.append(turbulent_magnetic_energy)

            # Turbulent kinetic energy based on standard deviation of velocity components
            std_velx = np.std(data["velx"])
            std_vely = np.std(data["vely"])
            std_velz = np.std(data["velz"])
            turbulent_kinetic_energy = 0.5 * (std_velx**2 + std_vely**2 + std_velz**2)  # RMS kinetic energy
            kinetic_energies.append(turbulent_kinetic_energy)

            # Volume-averaged density
            volume_averaged_density = data["dens"].mean()  # Average to get the mean density
            densities.append(volume_averaged_density)
            
            std_magx.append(std_magx)
            std_magy.append(std_magy)
            std_magz.append(std_magz)
            
            std_velx.append(std_velx)
            std_vely.append(std_vely)
            std_velz.append(std_velz)

        # Save the calculated quantities to a file
        np.savez(output_filename, times=times, mag_squared_means=mag_squared_means,
                magnetic_energies=magnetic_energies, kinetic_energies=kinetic_energies,
                densities=densities,std_velx = std_velx,std_vely = std_vely,std_velz = std_velz,
                std_magx = std_magx, std_magy = std_magy,std_magz = std_magz)

        print(f"Tracked quantities saved to '{output_filename}'")
        
    def plot_saved_quantities(self, input_filename="tracked_quantities_data.npz"):
        """Load saved quantities from a file and plot them over time."""
        # Load the data from the .npz file
        try:
            data = np.load(input_filename)
        except Exception as e:
            print(f"Failed to load {input_filename}: {e}")
            return

        # Extract the data
        times = data['times']
        mag_squared_means = data['mag_squared_means']
        magnetic_energies = data['magnetic_energies']
        kinetic_energies = data['kinetic_energies']
        densities = data['densities']
        std_magx = data['std_magx']
        std_magy = data['std_magy']
        std_magz = data['std_magz']
        std_velx = data['std_velx']
        std_vely = data['std_vely']
        std_velz = data['std_velz']
        #pres 
        
       # densities = 
        print(densities[0])


        # Plotting the quantities
        plt.figure(figsize=(15, 10))

        # # Magnetic Field Squared Mean Plot (1st position in a 3x3 grid)
        # plt.subplot(3, 3, 1)
        # plt.plot(times, mag_squared_means, '-o', label="Mean Magnetic Field Squared", color='b')
        # plt.xlabel("Simulation Time")
        # plt.ylabel("Magnetic Field Squared Mean")

        # plt.legend()
        # plt.grid(True)

        # Turbulent Magnetic Energy Plot (2nd position in a 3x3 grid)
        plt.subplot(2, 2, 1)
        plt.semilogy(times, magnetic_energies, '-o', label="Turbulent Magnetic Energy", color='c')
        plt.xlabel("Simulation Time")
        plt.ylabel("Turbulent Magnetic Energy")
        plt.xlim(1e-9,200e-9)
        plt.ylim(1e-9,1e-6)

        plt.legend()
        plt.grid(True)

        # Turbulent Kinetic Energy Based on Std Dev of Velocity Components Plot (4th position)
        plt.subplot(2, 2, 2)
        plt.semilogy(times, kinetic_energies, '-o', label="Turbulent Kinetic Energy (Std Dev of Velocity)", color='g')
        plt.xlabel("Simulation Time")
        plt.ylabel("Turbulent Kinetic Energy")
        plt.xlim(1e-9,200e-9)
        plt.ylim(1e6,1e13)

        plt.legend()
        plt.grid(True)

        # # Standard Deviations of Magnetic Field Components Plot (5th position)
        plt.subplot(2, 2, 3)
        plt.plot(times, std_magx, '-o', label="Std Dev of magx", color='b')
        plt.plot(times, std_magy, '-o', label="Std Dev of magy", color='r')
        plt.plot(times, std_magz, '-o', label="Std Dev of magz", color='m')
        plt.xlabel("Simulation Time")
        plt.ylabel("Standard Deviation")
        plt.title("Standard Deviations of Magnetic Field Components vs Simulation Time")
        plt.legend()
        plt.grid(True)

        # Standard Deviations of Velocity Components Plot (6th position)
        plt.subplot(2, 2, 4)
        plt.plot(times, std_velx, '-o', label="Std Dev of velx", color='b')
        plt.plot(times, std_vely, '-o', label="Std Dev of vely", color='r')
        plt.plot(times, std_velz, '-o', label="Std Dev of velz", color='g')
        plt.xlabel("Simulation Time")
        plt.ylabel("Standard Deviation")
        plt.title("Standard Deviations of Velocity Components vs Simulation Time")
        plt.legend()
        plt.grid(True)

        # Volume-Averaged Density Plot (8th position)
        # plt.subplot(3, 3, 8)
        # plt.plot(times, densities, '-o', label="Volume-Averaged Density", color='m')
        # plt.xlabel("Simulation Time")
        # plt.ylabel("Volume-Averaged Density")

        # plt.legend()
        # plt.grid(True)

        # Adjust layout and save the figure
        plt.tight_layout()
        plt.savefig("tracked_quantities_vs_time.pdf")
        plt.close()
        print("Plot saved as 'tracked_quantities_vs_time.pdf'")
        
    # def perform_helmholtz_decomposition(self, dump, num_neighbors=10):
    #     """
    #     Perform Helmholtz decomposition on the velocity field for a given dump without interpolation.

    #     Parameters:
    #     - dump: The dump number to process.
    #     - num_neighbors: Number of nearest neighbors to use for gradient approximations.
    #     """
    #     import scipy.spatial

    #     # Read the sphere data for the given dump
    #     data = self.read_sphere_data(dump)

    #     # Extract velocities and positions
    #     velx = data.velx
    #     vely = data.vely
    #     velz = data.velz
    #     coords = data.coords  # positions (x, y, z)

    #     # Number of data points
    #     num_points = coords.shape[1]

    #     # Build KDTree for efficient neighbor search
    #     tree = scipy.spatial.cKDTree(coords.T)

    #     # Precompute neighbor indices for all points
    #     neighbor_indices = np.zeros((num_points, num_neighbors), dtype=np.int32)
    #     for i in range(num_points):
    #         # Find indices of the nearest neighbors (including the point itself)
    #         distances, idx = tree.query(coords[:, i], k=num_neighbors)
    #         neighbor_indices[i, :] = idx

    #     # Compute divergence and curl directly on unstructured data
    #     divergence = np.zeros(num_points)
    #     curl_x = np.zeros(num_points)
    #     curl_y = np.zeros(num_points)
    #     curl_z = np.zeros(num_points)

    #     print("Computing divergence and curl on unstructured data...")

    #     for i in range(num_points):
    #         idx = neighbor_indices[i, :]
    #         # Exclude the point itself from neighbors if necessary
    #         idx = idx[idx != i]
    #         if len(idx) < 4:
    #             continue

    #         # Local coordinates relative to the point of interest
    #         dx = coords[0, idx] - coords[0, i]
    #         dy = coords[1, idx] - coords[1, i]
    #         dz = coords[2, idx] - coords[2, i]

    #         # Build the design matrix A
    #         A = np.column_stack((dx, dy, dz))

    #         # Differences in velocities
    #         delta_velx = velx[idx] - velx[i]
    #         delta_vely = vely[idx] - vely[i]
    #         delta_velz = velz[idx] - velz[i]

    #         # Solve least squares problems to find gradients
    #         try:
    #             # For velx
    #             grad_velx, _, _, _ = np.linalg.lstsq(A, delta_velx, rcond=None)
    #             # For vely
    #             grad_vely, _, _, _ = np.linalg.lstsq(A, delta_vely, rcond=None)
    #             # For velz
    #             grad_velz, _, _, _ = np.linalg.lstsq(A, delta_velz, rcond=None)
    #         except np.linalg.LinAlgError:
    #             continue

    #         # Compute divergence at this point
    #         divergence[i] = grad_velx[0] + grad_vely[1] + grad_velz[2]

    #         # Compute curl components at this point
    #         curl_x[i] = grad_velz[1] - grad_vely[2]
    #         curl_y[i] = grad_velx[2] - grad_velz[0]
    #         curl_z[i] = grad_vely[0] - grad_velx[1]

    #     # Perform Helmholtz decomposition using Rbf
    #     print("Performing Helmholtz decomposition using Rbf...")

    #     from scipy.interpolate import Rbf

    #     # Coordinates
    #     x = coords[0, :]
    #     y = coords[1, :]
    #     z = coords[2, :]

    #     # Solve for scalar potential phi (from divergence)
    #     print("Interpolating scalar potential phi using Rbf...")
    #     phi_rbf = Rbf(x, y, z, divergence, function='multiquadric', epsilon=1)
    #     phi = phi_rbf(x, y, z)

    #     # Compute gradient of phi to get irrotational component
    #     grad_phi = np.zeros((num_points, 3))

    #     # Compute numerical gradients of phi using finite differences
    #     print("Computing gradient of phi...")
    #     for i in range(num_points):
    #         idx = neighbor_indices[i, :]
    #         idx = idx[idx != i]
    #         if len(idx) < 4:
    #             continue

    #         dx = coords[:, idx] - coords[:, i][:, np.newaxis]
    #         A = dx.T

    #         delta_phi = phi[idx] - phi[i]

    #         try:
    #             grad_phi_i, _, _, _ = np.linalg.lstsq(A, delta_phi, rcond=None)
    #             grad_phi[i, :] = grad_phi_i
    #         except np.linalg.LinAlgError:
    #             continue

    #     vel_irrot = -grad_phi

    #     # Compute solenoidal component by subtracting irrotational component from original velocity
    #     vel_solenoidal = np.vstack((velx, vely, velz)).T - vel_irrot

    #     # Verify that the sum of irrotational and solenoidal components approximates the original velocity field
    #     vel_reconstructed = vel_irrot + vel_solenoidal

    #     # Compute the difference between the original and reconstructed velocity fields
    #     diff = np.sqrt((velx - vel_reconstructed[:, 0])**2 +
    #                 (vely - vel_reconstructed[:, 1])**2 +
    #                 (velz - vel_reconstructed[:, 2])**2)

    #     max_diff = np.max(diff)
    #     print(f"Maximum difference between original and reconstructed velocity fields: {max_diff}")

    #     # Save the results to a file
    #     output_filename = self.plt_fname(dump) + "_helmholtz_decomposition.npz"
    #     np.savez(output_filename,
    #             coords=coords,
    #             velx_original=velx, vely_original=vely, velz_original=velz,
    #             velx_irrot=vel_irrot[:, 0], vely_irrot=vel_irrot[:, 1], velz_irrot=vel_irrot[:, 2],
    #             velx_solenoidal=vel_solenoidal[:, 0], vely_solenoidal=vel_solenoidal[:, 1], velz_solenoidal=vel_solenoidal[:, 2],
    #             divergence=divergence,
    #             curl_x=curl_x, curl_y=curl_y, curl_z=curl_z)
    #     print(f"Helmholtz decomposition results saved to '{output_filename}'")



    def compute_stretching_compression(self, dump, num_neighbors=10):
        """
        Compute stretching and compression terms in the induction equation for a given dump without interpolation.

        Parameters:
        - dump: The dump number to process.
        - num_neighbors: Number of nearest neighbors to use for gradient approximations.
        """

        import scipy.spatial

        # Read the sphere data for the given dump
        data = self.read_sphere_data(dump)
        
        # Extract velocities, magnetic fields, and positions
        velx = data.velx
        vely = data.vely
        velz = data.velz
        magx = data.magx
        magy = data.magy
        magz = data.magz
        coords = data.coords  # positions (x, y, z)
        
        # Number of data points
        num_points = coords.shape[1]
        
        # Build KDTree for efficient neighbor search
        tree = scipy.spatial.cKDTree(coords.T)
        
        # Precompute neighbor indices for all points
        neighbor_indices = np.zeros((num_points, num_neighbors), dtype=np.int32)
        for i in range(num_points):
            # Find indices of the nearest neighbors (including the point itself)
            distances, idx = tree.query(coords[:, i], k=num_neighbors)
            neighbor_indices[i, :] = idx

        # Initialize arrays for gradients
        grad_vx = np.zeros((num_points, 3))
        grad_vy = np.zeros((num_points, 3))
        grad_vz = np.zeros((num_points, 3))
        divergence_v = np.zeros(num_points)

        print("Computing spatial derivatives using finite difference approximations on unstructured grids...")

        for i in range(num_points):
            idx = neighbor_indices[i, :]
            idx = idx[idx != i]
            if len(idx) < 4:
                continue

            # Local coordinates relative to the point of interest
            dx = coords[0, idx] - coords[0, i]
            dy = coords[1, idx] - coords[1, i]
            dz = coords[2, idx] - coords[2, i]

            # Build the design matrix A
            A = np.column_stack((dx, dy, dz))

            # Differences in velocities
            delta_velx = velx[idx] - velx[i]
            delta_vely = vely[idx] - vely[i]
            delta_velz = velz[idx] - velz[i]

            # Solve least squares problems to find gradients
            try:
                # For velx
                grad_velx, _, _, _ = np.linalg.lstsq(A, delta_velx, rcond=None)
                grad_vx[i, :] = grad_velx

                # For vely
                grad_vely, _, _, _ = np.linalg.lstsq(A, delta_vely, rcond=None)
                grad_vy[i, :] = grad_vely

                # For velz
                grad_velz, _, _, _ = np.linalg.lstsq(A, delta_velz, rcond=None)
                grad_vz[i, :] = grad_velz

                # Compute divergence at this point
                divergence_v[i] = grad_velx[0] + grad_vely[1] + grad_velz[2]

            except np.linalg.LinAlgError:
                continue

        # Assemble the gradient tensor of velocity field
        grad_v = np.stack((grad_vx, grad_vy, grad_vz), axis=-2)  # shape (num_points, 3, 3)

        # Assemble the magnetic field components
        grid_B = np.vstack((magx, magy, magz)).T  # shape (num_points, 3)

        # Compute stretching term: (B • ∇) v
        stretching_term = np.einsum('ij,ijk->ik', grid_B, grad_v)

        # Compute compression term: -B * (∇ • v)
        compression_term = -grid_B * divergence_v[:, np.newaxis]

        # Compute magnitudes
        stretching_magnitude = np.linalg.norm(stretching_term, axis=1)
        compression_magnitude = np.linalg.norm(compression_term, axis=1)
        rhs_magnitude = np.linalg.norm(stretching_term + compression_term, axis=1)

        # Compute averages
        avg_stretching = np.mean(stretching_magnitude)
        avg_compression = np.mean(compression_magnitude)
        avg_rhs = np.mean(rhs_magnitude)

        print(f"Average stretching term magnitude: {avg_stretching}")
        print(f"Average compression term magnitude: {avg_compression}")
        print(f"Average RHS term magnitude: {avg_rhs}")

        # Save the results to a file
        output_filename = self.plt_fname(dump) + "_stretching_compression.npz"
        np.savez(output_filename,
                stretching_term=stretching_term,
                compression_term=compression_term,
                rhs_induction=stretching_term + compression_term,
                avg_stretching=avg_stretching,
                avg_compression=avg_compression,
                avg_rhs=avg_rhs)
        print(f"Stretching and compression terms saved to '{output_filename}'")
        
    def compute_vorticity_divergence(self, dump, num_neighbors=10, return_averages=False):
        """
        Compute vorticity and divergence of the velocity field for a given dump using finite difference approximations on unstructured grids.

        Parameters:
        - dump: The dump number to process.
        - num_neighbors: Number of nearest neighbors to use for finite difference approximation.
        - return_averages: If True, returns the average vorticity and divergence magnitudes.
        """
        import scipy.spatial

        # Read the sphere data for the given dump
        data = self.read_sphere_data(dump)

        # Extract velocities and positions
        velx = data.velx
        vely = data.vely
        velz = data.velz
        coords = data.coords  # positions (x, y, z)

        # Number of data points
        num_points = coords.shape[1]

        # Initialize arrays for gradients
        d_velx_dx = np.zeros(num_points)
        d_velx_dy = np.zeros(num_points)
        d_velx_dz = np.zeros(num_points)

        d_vely_dx = np.zeros(num_points)
        d_vely_dy = np.zeros(num_points)
        d_vely_dz = np.zeros(num_points)

        d_velz_dx = np.zeros(num_points)
        d_velz_dy = np.zeros(num_points)
        d_velz_dz = np.zeros(num_points)

        print("Computing spatial derivatives using finite difference approximations on unstructured grids...")

        # Build KDTree for efficient neighbor search
        tree = scipy.spatial.cKDTree(coords.T)

        # For each point, find neighbors and compute gradients
        for i in range(num_points):
            # Find indices of the nearest neighbors (including the point itself)
            distances, idx = tree.query(coords[:, i], k=num_neighbors)

            # Exclude the point itself if included
            if idx[0] == i:
                idx = idx[1:]
                distances = distances[1:]
            else:
                idx = idx[:-1]
                distances = distances[:-1]

            # Ensure we have enough neighbors
            if len(idx) < 3:
                # Not enough points to compute derivatives
                continue

            # Local coordinates relative to the point of interest
            dx = coords[0, idx] - coords[0, i]
            dy = coords[1, idx] - coords[1, i]
            dz = coords[2, idx] - coords[2, i]

            # Build the design matrix A
            A = np.column_stack((dx, dy, dz))  # Shape (N, 3)

            # Differences in velocities
            delta_velx = velx[idx] - velx[i]  # Shape (N,)
            delta_vely = vely[idx] - vely[i]
            delta_velz = velz[idx] - velz[i]

            # Solve least squares problems to find gradients
            try:
                # For velx
                grad_velx, _, _, _ = np.linalg.lstsq(A, delta_velx, rcond=None)
                d_velx_dx[i], d_velx_dy[i], d_velx_dz[i] = grad_velx

                # For vely
                grad_vely, _, _, _ = np.linalg.lstsq(A, delta_vely, rcond=None)
                d_vely_dx[i], d_vely_dy[i], d_vely_dz[i] = grad_vely

                # For velz
                grad_velz, _, _, _ = np.linalg.lstsq(A, delta_velz, rcond=None)
                d_velz_dx[i], d_velz_dy[i], d_velz_dz[i] = grad_velz

            except np.linalg.LinAlgError:
                # Singular matrix, skip this point
                continue

        # Compute divergence of velocity field
        divergence = d_velx_dx + d_vely_dy + d_velz_dz

        # Compute vorticity (curl of velocity field)
        vorticity_x = d_velz_dy - d_vely_dz
        vorticity_y = d_velx_dz - d_velz_dx
        vorticity_z = d_vely_dx - d_velx_dy

        # Compute magnitude of vorticity
        vorticity_magnitude = np.sqrt(vorticity_x**2 + vorticity_y**2 + vorticity_z**2)

        # Compute the magnitude of divergence
        divergence_magnitude = np.abs(divergence)

        # Compute average vorticity and divergence over the points
        avg_vorticity = np.mean(vorticity_magnitude)
        avg_divergence = np.mean(divergence_magnitude)

        print(f"Average vorticity magnitude: {avg_vorticity}")
        print(f"Average divergence magnitude: {avg_divergence}")

        # Save the results to a file
        output_filename = self.plt_fname(dump) + "_vorticity_divergence.npz"
        np.savez(output_filename,
                vorticity_x=vorticity_x,
                vorticity_y=vorticity_y,
                vorticity_z=vorticity_z,
                vorticity_magnitude=vorticity_magnitude,
                divergence=divergence,
                divergence_magnitude=divergence_magnitude,
                avg_vorticity=avg_vorticity,
                avg_divergence=avg_divergence)
        print(f"Vorticity and divergence saved to '{output_filename}'")

        if return_averages:
            return avg_vorticity, avg_divergence




    def plot_vorticity_divergence(self, dump, plane='xy', level=None):
        """
        Plot the vorticity magnitude and divergence in a specified plane.

        Parameters:
        - dump: The dump number to process.
        - plane: The plane to plot ('xy', 'xz', 'yz').
        - level: The index level along the axis perpendicular to the plane.
        """
        # Load the data
        data_filename = self.plt_fname(dump) + "_vorticity_divergence.npz"
        if not os.path.isfile(data_filename):
            print(f"Data file '{data_filename}' not found. Please run 'compute_vorticity_divergence' first.")
            return

        data = np.load(data_filename)
        vorticity_magnitude = data['vorticity_magnitude']
        divergence = data['divergence']

        # Determine the slice index
        if level is None:
            if plane == 'xy':
                level = vorticity_magnitude.shape[2] // 2  # Middle slice along z
            elif plane == 'xz':
                level = vorticity_magnitude.shape[1] // 2  # Middle slice along y
            elif plane == 'yz':
                level = vorticity_magnitude.shape[0] // 2  # Middle slice along x

        # Plotting
        plt.figure(figsize=(12, 6))

        if plane == 'xy':
            vort_slice = vorticity_magnitude[:, :, level]
            div_slice = divergence[:, :, level]
            extent = [0, vort_slice.shape[1], 0, vort_slice.shape[0]]
        elif plane == 'xz':
            vort_slice = vorticity_magnitude[:, level, :]
            div_slice = divergence[:, level, :]
            extent = [0, vort_slice.shape[1], 0, vort_slice.shape[0]]
        elif plane == 'yz':
            vort_slice = vorticity_magnitude[level, :, :]
            div_slice = divergence[level, :, :]
            extent = [0, vort_slice.shape[1], 0, vort_slice.shape[0]]
        else:
            print(f"Invalid plane '{plane}'. Choose from 'xy', 'xz', 'yz'.")
            return

        # Vorticity magnitude plot
        plt.subplot(1, 2, 1)
        plt.imshow(vort_slice, extent=extent, origin='lower', cmap='viridis')
        plt.colorbar(label='Vorticity Magnitude')
        plt.title('Vorticity Magnitude')
        plt.xlabel(plane[0] + ' axis')
        plt.ylabel(plane[1] + ' axis')

        # Divergence plot
        plt.subplot(1, 2, 2)
        plt.imshow(div_slice, extent=extent, origin='lower', cmap='seismic', vmin=-np.max(np.abs(div_slice)), vmax=np.max(np.abs(div_slice)))
        plt.colorbar(label='Divergence')
        plt.title('Divergence')
        plt.xlabel(plane[0] + ' axis')
        plt.ylabel(plane[1] + ' axis')

        plt.tight_layout()
        output_filename = self.plt_fname(dump) + f"_vorticity_divergence_{plane}_plane.png"
        plt.savefig(output_filename)
        plt.close()
        print(f"Vorticity and divergence plots saved to '{output_filename}'")

    def save_vorticity_divergence_over_time(self, dumps=None, output_filename='vorticity_divergence_over_time.npz'):
        """
        Compute and save average vorticity and divergence over time.

        Parameters:
        - dumps: List of dump numbers to process. If None, processes all available dumps.
        - output_filename: Filename to save the data.
        """
        if dumps is None:
            dumps = [int(os.path.basename(fname).split('_')[-1]) for fname in self.plt_files]
        times = []
        avg_vorticities = []
        avg_divergences = []

        for dump in dumps:
            data = self.read_sphere_data(dump)
            time = data.time
            # Check if data file exists to avoid recomputation
            data_filename = self.plt_fname(dump) + "_vorticity_divergence.npz"
            if not os.path.isfile(data_filename):
                avg_vorticity, avg_divergence = self.compute_vorticity_divergence(dump, return_averages=True)
            else:
                # Load the data
                loaded_data = np.load(data_filename)
                avg_vorticity = loaded_data['avg_vorticity']
                avg_divergence = loaded_data['avg_divergence']
            times.append(time)
            avg_vorticities.append(avg_vorticity)
            avg_divergences.append(avg_divergence)

        # Save data
        np.savez(output_filename, times=times, avg_vorticities=avg_vorticities, avg_divergences=avg_divergences)
        print(f"Vorticity and divergence over time saved to '{output_filename}'")

    def plot_vorticity_divergence_over_time(self, input_filename='vorticity_divergence_over_time.npz'):
        """
        Plot average vorticity and divergence against time.

        Parameters:
        - input_filename: Filename of the data file containing vorticity and divergence over time.
        """
        # Load the data
        if not os.path.isfile(input_filename):
            print(f"Data file '{input_filename}' not found. Please run 'save_vorticity_divergence_over_time' first.")
            return
        data = np.load(input_filename)
        times = data['times']
        avg_vorticities = data['avg_vorticities']
        avg_divergences = data['avg_divergences']

        # Convert times to nanoseconds if needed
        times_ns = np.array(times) * 1e9  # Assuming times are in seconds

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(times_ns, avg_vorticities, '-o', label='Average Vorticity Magnitude')
        plt.plot(times_ns, avg_divergences, '-s', label='Average Divergence Magnitude')
        plt.xlabel('Time (ns)')
        plt.ylabel('Average Magnitude')
        plt.title('Average Vorticity and Divergence over Time')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('vorticity_divergence_over_time.png')
        plt.close()
        print("Plot of vorticity and divergence over time saved as 'vorticity_divergence_over_time.png'")
    def helmholtz_decomposition(self, dump, grid_resolution=50):
        """
        Perform the Helmholtz decomposition on the velocity field from the data dump.
        """
        import numpy as np
        from scipy.interpolate import griddata
        from scipy.sparse.linalg import spsolve, cg
        from scipy.sparse import lil_matrix, csr_matrix
        from scipy.ndimage import gaussian_filter

        # Load the data
        data = self.read_sphere_data(dump)
        
        # Extract the velocity components and coordinates
        velx = data.velx
        vely = data.vely
        velz = data.velz
        coords = data.coords

        # Determine the bounding box of the sphere
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        z_min, z_max = coords[:, 2].min(), coords[:, 2].max()

        # Create a regular grid within the bounding box
        x = np.linspace(x_min, x_max, grid_resolution)
        y = np.linspace(y_min, y_max, grid_resolution)
        z = np.linspace(z_min, z_max, grid_resolution)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # Flatten the grid coordinates
        grid_coords = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

        # Interpolate velocities onto the grid
        velx_grid = griddata(coords, velx, grid_coords, method='linear').reshape(X.shape)
        vely_grid = griddata(coords, vely, grid_coords, method='linear').reshape(Y.shape)
        velz_grid = griddata(coords, velz, grid_coords, method='linear').reshape(Z.shape)

        # Replace NaNs resulting from interpolation with zeros
        velx_grid = np.nan_to_num(velx_grid)
        vely_grid = np.nan_to_num(vely_grid)
        velz_grid = np.nan_to_num(velz_grid)

        # Smooth the velocity fields to reduce numerical noise
        velx_grid = gaussian_filter(velx_grid, sigma=1)
        vely_grid = gaussian_filter(vely_grid, sigma=1)
        velz_grid = gaussian_filter(velz_grid, sigma=1)

        # Compute the divergence of the velocity field
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        dz = z[1] - z[0]

        div_v = np.zeros_like(velx_grid)

        # Central differences for interior points
        div_v[1:-1, 1:-1, 1:-1] = (
            (velx_grid[2:, 1:-1, 1:-1] - velx_grid[:-2, 1:-1, 1:-1]) / (2 * dx) +
            (vely_grid[1:-1, 2:, 1:-1] - vely_grid[1:-1, :-2, 1:-1]) / (2 * dy) +
            (velz_grid[1:-1, 1:-1, 2:] - velz_grid[1:-1, 1:-1, :-2]) / (2 * dz)
        )

        # Handle boundaries (assuming zero divergence outside the sphere)
        div_v = np.nan_to_num(div_v)

        # Set up the Laplacian operator
        nx, ny, nz = velx_grid.shape
        N = nx * ny * nz

        # Create the sparse matrix for the Laplacian
        Laplacian = lil_matrix((N, N))

        # Map 3D indices to 1D indices
        def idx(i, j, k):
            return i * (ny * nz) + j * nz + k

        # Build the Laplacian matrix
        print("Setting up the Laplacian matrix...")
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    n = idx(i, j, k)
                    Laplacian[n, n] = -2 / dx**2 - 2 / dy**2 - 2 / dz**2

                    if i > 0:
                        Laplacian[n, idx(i-1, j, k)] = 1 / dx**2
                    if i < nx - 1:
                        Laplacian[n, idx(i+1, j, k)] = 1 / dx**2
                    if j > 0:
                        Laplacian[n, idx(i, j-1, k)] = 1 / dy**2
                    if j < ny - 1:
                        Laplacian[n, idx(i, j+1, k)] = 1 / dy**2
                    if k > 0:
                        Laplacian[n, idx(i, j, k-1)] = 1 / dz**2
                    if k < nz - 1:
                        Laplacian[n, idx(i, j, k+1)] = 1 / dz**2

        # Apply boundary conditions (Dirichlet: φ = 0 on the boundary)
        print("Applying boundary conditions...")
        boundary_indices = []

        # Identify boundary points
        for i in [0, nx - 1]:
            for j in range(ny):
                for k in range(nz):
                    n = idx(i, j, k)
                    Laplacian[n, :] = 0
                    Laplacian[n, n] = 1
                    div_v[i, j, k] = 0  # φ = 0 at boundary
                    boundary_indices.append(n)

        for i in range(nx):
            for j in [0, ny - 1]:
                for k in range(nz):
                    n = idx(i, j, k)
                    Laplacian[n, :] = 0
                    Laplacian[n, n] = 1
                    div_v[i, j, k] = 0  # φ = 0 at boundary
                    boundary_indices.append(n)

        for i in range(nx):
            for j in range(ny):
                for k in [0, nz - 1]:
                    n = idx(i, j, k)
                    Laplacian[n, :] = 0
                    Laplacian[n, n] = 1
                    div_v[i, j, k] = 0  # φ = 0 at boundary
                    boundary_indices.append(n)

        # Convert to CSR format for efficient arithmetic and solving
        Laplacian = Laplacian.tocsr()

        # Flatten the divergence array for solving
        div_v_flat = -div_v.ravel()

        # Solve the Poisson equation
        print("Solving the Poisson equation...")
        phi_flat, info = cg(Laplacian, div_v_flat, maxiter=1000)
        if info != 0:
            print("Conjugate gradient solver did not converge.")
            return None

        phi = phi_flat.reshape(velx_grid.shape)

        # Compute the gradient of φ to get the curl-free component
        grad_phi_x = np.zeros_like(velx_grid)
        grad_phi_y = np.zeros_like(vely_grid)
        grad_phi_z = np.zeros_like(velz_grid)

        # Central differences for interior points
        grad_phi_x[1:-1, 1:-1, 1:-1] = (
            (phi[2:, 1:-1, 1:-1] - phi[:-2, 1:-1, 1:-1]) / (2 * dx)
        )
        grad_phi_y[1:-1, 1:-1, 1:-1] = (
            (phi[1:-1, 2:, 1:-1] - phi[1:-1, :-2, 1:-1]) / (2 * dy)
        )
        grad_phi_z[1:-1, 1:-1, 1:-1] = (
            (phi[1:-1, 1:-1, 2:] - phi[1:-1, 1:-1, :-2]) / (2 * dz)
        )

        # Handle boundaries (forward/backward differences)
        # x boundaries
        grad_phi_x[0, :, :] = (phi[1, :, :] - phi[0, :, :]) / dx
        grad_phi_x[-1, :, :] = (phi[-1, :, :] - phi[-2, :, :]) / dx
        # y boundaries
        grad_phi_y[:, 0, :] = (phi[:, 1, :] - phi[:, 0, :]) / dy
        grad_phi_y[:, -1, :] = (phi[:, -1, :] - phi[:, -2, :]) / dy
        # z boundaries
        grad_phi_z[:, :, 0] = (phi[:, :, 1] - phi[:, :, 0]) / dz
        grad_phi_z[:, :, -1] = (phi[:, :, -1] - phi[:, :, -2]) / dz

        # The irrotational (curl-free) component
        vel_curl_free_x = grad_phi_x
        vel_curl_free_y = grad_phi_y
        vel_curl_free_z = grad_phi_z

        # The solenoidal (divergence-free) component
        vel_div_free_x = velx_grid - vel_curl_free_x
        vel_div_free_y = vely_grid - vel_curl_free_y
        vel_div_free_z = velz_grid - vel_curl_free_z

        # Return the components
        return {
            'curl_free': {
                'velx': vel_curl_free_x,
                'vely': vel_curl_free_y,
                'velz': vel_curl_free_z,
            },
            'div_free': {
                'velx': vel_div_free_x,
                'vely': vel_div_free_y,
                'velz': vel_div_free_z,
            },
            'grid': {
                'X': X,
                'Y': Y,
                'Z': Z
            }
        }




    def perform_helmholtz_decomposition_spherical(self, dump):
        """
        Perform Helmholtz decomposition on the velocity field for a given dump using structured spherical coordinates.
        """
        import numpy as np
        import pyshtools as pysh

        # Read the data
        data = self.read_sphere_data(dump)

        # Extract positions and velocities in Cartesian coordinates
        coords = data.coords.T  # positions (n_points, 3)
        velx = data.velx
        vely = data.vely
        velz = data.velz

        # Convert positions to spherical coordinates
        print("Converting positions from Cartesian to spherical coordinates...")
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]

        # Compute spherical coordinates
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r)          # Polar angle (0 to pi)
        phi = np.arctan2(y, x)            # Azimuthal angle (-pi to pi)
        phi[phi < 0] += 2 * np.pi         # Convert phi to range [0, 2*pi]

        # Convert velocities to spherical components
        print("Converting velocities from Cartesian to spherical coordinates...")
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        # Radial unit vector components
        e_r_x = sin_theta * cos_phi
        e_r_y = sin_theta * sin_phi
        e_r_z = cos_theta

        # Theta unit vector components (polar angle)
        e_theta_x = cos_theta * cos_phi
        e_theta_y = cos_theta * sin_phi
        e_theta_z = -sin_theta

        # Phi unit vector components (azimuthal angle)
        e_phi_x = -sin_phi
        e_phi_y = cos_phi
        e_phi_z = np.zeros_like(r)

        # Compute velocities in spherical coordinates
        vr = velx * e_r_x + vely * e_r_y + velz * e_r_z
        vtheta = velx * e_theta_x + vely * e_theta_y + velz * e_theta_z
        vphi = velx * e_phi_x + vely * e_phi_y + velz * e_phi_z

        # Now, we need to interpolate the scattered data onto a structured grid
        print("Interpolating data onto structured spherical grid...")
        # Define the grid parameters based on pyshtools' requirements
        lmax = 31  # Maximum spherical harmonic degree
        nlat = 2 * lmax + 2
        nlon = 2 * nlat

        # Generate latitudes and longitudes
        latitudes = np.linspace(90, -90, nlat)  # Degrees
        longitudes = np.linspace(0, 360, nlon, endpoint=False)  # Degrees

        theta_grid = np.radians(90 - latitudes)  # Convert to colatitude in radians
        phi_grid = np.radians(longitudes)        # Convert to radians

        # Create 2D grids of theta and phi
        Theta_grid, Phi_grid = np.meshgrid(theta_grid, phi_grid, indexing='ij')

        # Use unique radii from the data
        r_unique = np.unique(r)
        nr = len(r_unique)
        r_grid = r_unique

        # Initialize arrays for the velocity components and divergence
        vr_grid = np.zeros((nr, nlat, nlon))
        vtheta_grid = np.zeros_like(vr_grid)
        vphi_grid = np.zeros_like(vr_grid)
        divergence = np.zeros_like(vr_grid)

        from scipy.interpolate import griddata

        print("Interpolating velocities onto the grid...")
        for idx_r, r_value in enumerate(r_grid):
            mask = np.isclose(r, r_value)
            if np.sum(mask) == 0:
                continue
            points = np.vstack((theta[mask], phi[mask])).T
            values_vr = vr[mask]
            values_vtheta = vtheta[mask]
            values_vphi = vphi[mask]

            grid_points = np.vstack((Theta_grid.ravel(), Phi_grid.ravel())).T

            vr_layer = griddata(points, values_vr, grid_points, method='linear')
            vtheta_layer = griddata(points, values_vtheta, grid_points, method='linear')
            vphi_layer = griddata(points, values_vphi, grid_points, method='linear')

            # Handle NaNs
            vr_layer = np.nan_to_num(vr_layer, nan=0.0)
            vtheta_layer = np.nan_to_num(vtheta_layer, nan=0.0)
            vphi_layer = np.nan_to_num(vphi_layer, nan=0.0)

            # Reshape to grid shape
            vr_grid[idx_r, :, :] = vr_layer.reshape((nlat, nlon))
            vtheta_grid[idx_r, :, :] = vtheta_layer.reshape((nlat, nlon))
            vphi_grid[idx_r, :, :] = vphi_layer.reshape((nlat, nlon))

        # Proceed with the decomposition
        print("Computing divergence and performing Helmholtz decomposition...")
        phi_coeffs_radial = []

        # Precompute sin(theta) grid
        sin_theta_grid = np.sin(Theta_grid)
        sin_theta_grid[sin_theta_grid == 0] = 1e-10  # Avoid division by zero

        for idx_r in range(nr):
            vr_layer = vr_grid[idx_r, :, :]
            vtheta_layer = vtheta_grid[idx_r, :, :]
            vphi_layer = vphi_grid[idx_r, :, :]

            # Compute divergence on the sphere
            d_vr_dr = np.gradient(vr_grid, r_grid, axis=0)[idx_r, :, :]  # Radial derivative
            d_vtheta_dtheta = np.gradient(vtheta_layer, theta_grid, axis=0)
            d_vphi_dphi = np.gradient(vphi_layer, phi_grid, axis=1)

            divergence_layer = (
                (1 / r_grid[idx_r]**2) * d_vr_dr * r_grid[idx_r]**2 +
                (1 / (r_grid[idx_r] * sin_theta_grid)) * d_vtheta_dtheta * sin_theta_grid +
                (1 / (r_grid[idx_r] * sin_theta_grid)) * d_vphi_dphi
            )

            divergence[idx_r, :, :] = divergence_layer

            # Expand divergence into spherical harmonics
            grid = pysh.SHGrid.from_array(divergence_layer)
            coeffs = grid.expand()
            divergence_coeffs = coeffs.coeffs
            lmax = coeffs.lmax

            # Solve Poisson equation in harmonic space
            phi_coeffs = np.zeros_like(divergence_coeffs)

            for l in range(lmax + 1):
                for m in range(l + 1):
                    if l == 0:
                        phi_coeffs[0, l, m] = 0
                        phi_coeffs[1, l, m] = 0
                    else:
                        phi_coeffs[0, l, m] = divergence_coeffs[0, l, m] / (l * (l + 1))
                        phi_coeffs[1, l, m] = divergence_coeffs[1, l, m] / (l * (l + 1))

            phi_coeffs_radial.append(phi_coeffs)

        # Reconstruct scalar potential phi
        print("Reconstructing scalar potential phi...")
        phi = np.zeros_like(vr_grid)

        for idx_r in range(nr):
            coeffs = pysh.SHCoeffs.from_array(phi_coeffs_radial[idx_r])
            phi_grid_layer = coeffs.expand(grid='DH').to_array()
            phi_grid_layer = phi_grid_layer[:nlat, :nlon]  # Ensure shapes match
            phi[idx_r, :, :] = phi_grid_layer

        # Compute gradient of phi to get irrotational component
        print("Computing gradient of scalar potential phi...")
        d_phi_dr = np.gradient(phi, r_grid, axis=0)
        d_phi_dtheta = np.gradient(phi, theta_grid, axis=1)
        d_phi_dphi = np.gradient(phi, phi_grid, axis=2)

        # Irrotational components
        r_grid_3d = r_grid[:, None, None]
        vel_irrot_r = -d_phi_dr
        vel_irrot_theta = -(1 / r_grid_3d) * d_phi_dtheta
        vel_irrot_phi = -(1 / (r_grid_3d * sin_theta_grid[None, :, :])) * d_phi_dphi

        # Solenoidal components
        print("Computing solenoidal component of the velocity field...")
        vel_solenoidal_r = vr_grid - vel_irrot_r
        vel_solenoidal_theta = vtheta_grid - vel_irrot_theta
        vel_solenoidal_phi = vphi_grid - vel_irrot_phi

        # Verify the decomposition
        print("Verifying the Helmholtz decomposition...")
        vel_reconstructed_r = vel_irrot_r + vel_solenoidal_r
        vel_reconstructed_theta = vel_irrot_theta + vel_solenoidal_theta
        vel_reconstructed_phi = vel_irrot_phi + vel_solenoidal_phi

        max_diff_r = np.max(np.abs(vr_grid - vel_reconstructed_r))
        max_diff_theta = np.max(np.abs(vtheta_grid - vel_reconstructed_theta))
        max_diff_phi = np.max(np.abs(vphi_grid - vel_reconstructed_phi))
        print(f"Maximum difference in v_r: {max_diff_r}")
        print(f"Maximum difference in v_theta: {max_diff_theta}")
        print(f"Maximum difference in v_phi: {max_diff_phi}")

        # Save the results to a file
        output_filename = self.plt_fname(dump) + "_helmholtz_decomposition_spherical.npz"
        np.savez(output_filename,
                r_grid=r_grid,
                Theta_grid=Theta_grid,
                Phi_grid=Phi_grid,
                vr_original=vr_grid,
                vtheta_original=vtheta_grid,
                vphi_original=vphi_grid,
                vr_irrot=vel_irrot_r,
                vtheta_irrot=vel_irrot_theta,
                vphi_irrot=vel_irrot_phi,
                vr_solenoidal=vel_solenoidal_r,
                vtheta_solenoidal=vel_solenoidal_theta,
                vphi_solenoidal=vel_solenoidal_phi)
        print(f"Helmholtz decomposition results saved to '{output_filename}'")

    
    def compute_gradient(phi, dx, dy, dz, mask):
        vpx = np.zeros_like(phi)
        vpy = np.zeros_like(phi)
        vpz = np.zeros_like(phi)

        # Use central differences for interior points
        vpx[1:-1, :, :] = (phi[2:, :, :] - phi[:-2, :, :]) / (2*dx)
        vpy[:, 1:-1, :] = (phi[:, 2:, :] - phi[:, :-2, :]) / (2*dy)
        vpz[:, :, 1:-1] = (phi[:, :, 2:] - phi[:, :, :-2]) / (2*dz)

        # Set gradient components to zero outside the sphere
        vpx[~mask] = 0
        vpy[~mask] = 0
        vpz[~mask] = 0

        return vpx, vpy, vpz

    def compute_divergence(vx, vy, vz, dx, dy, dz, mask):
        div = np.zeros_like(vx)
        # Use central differences for interior points
        div[1:-1,1:-1,1:-1] = (
            (vx[2:,1:-1,1:-1] - vx[:-2,1:-1,1:-1]) / (2*dx) +
            (vy[1:-1,2:,1:-1] - vy[1:-1,:-2,1:-1]) / (2*dy) +
            (vz[1:-1,1:-1,2:] - vz[1:-1,1:-1,:-2]) / (2*dz)
        )
        # Set divergence to zero outside the sphere
        div[~mask] = 0
        return div
    def compute_curl(vx, vy, vz, dx, dy, dz):
        curl_x = np.zeros_like(vx)
        curl_y = np.zeros_like(vy)
        curl_z = np.zeros_like(vz)
        curl_x[:,1:-1,1:-1] = (
            (vz[:,2:,1:-1] - vz[:, :-2,1:-1]) / (2*dy) -
            (vy[:,1:-1,2:] - vy[:,1:-1,:-2]) / (2*dz)
        )
        curl_y[1:-1,:,1:-1] = (
            (vx[1:-1,:,2:] - vx[1:-1,:,:-2]) / (2*dz) -
            (vz[2:,:,1:-1] - vz[:-2,:,1:-1]) / (2*dx)
        )
        curl_z[1:-1,1:-1,:] = (
            (vy[2:,1:-1,:] - vy[:-2,1:-1,:]) / (2*dx) -
            (vx[1:-1,2:,:] - vx[1:-1,:-2,:]) / (2*dy)
        )
        return curl_x, curl_y, curl_z
    
    def perform_helmholtz_decomposition(self, dump, num_neighbors=10):
        """
        Perform Helmholtz decomposition on the velocity field for a given dump without interpolation.

        Parameters:
        - dump: The dump number to process.
        - num_neighbors: Number of nearest neighbors to use for gradient approximations.
        """
        import scipy.spatial

        # Read the sphere data for the given dump
        data = self.read_sphere_data(dump)

        # Extract velocities and positions
        velx = data.velx
        vely = data.vely
        velz = data.velz
        coords = data.coords  # positions (x, y, z)

        # Number of data points
        num_points = coords.shape[1]

        # Build KDTree for efficient neighbor search
        tree = scipy.spatial.cKDTree(coords.T)

        # Precompute neighbor indices for all points
        neighbor_indices = np.zeros((num_points, num_neighbors), dtype=np.int32)
        for i in range(num_points):
            # Find indices of the nearest neighbors (including the point itself)
            distances, idx = tree.query(coords[:, i], k=num_neighbors)
            neighbor_indices[i, :] = idx

        # Initialize arrays for gradients
        grad_vx = np.zeros((num_points, 3))
        grad_vy = np.zeros((num_points, 3))
        grad_vz = np.zeros((num_points, 3))
        divergence_v = np.zeros(num_points)

        print("Computing spatial derivatives using finite difference approximations on unstructured grids...")

        for i in range(num_points):
            idx = neighbor_indices[i, :]
            idx = idx[idx != i]
            if len(idx) < 4:
                continue

            # Local coordinates relative to the point of interest
            dx = coords[0, idx] - coords[0, i]
            dy = coords[1, idx] - coords[1, i]
            dz = coords[2, idx] - coords[2, i]

            # Build the design matrix A
            A = np.column_stack((dx, dy, dz))

            # Differences in velocities
            delta_velx = velx[idx] - velx[i]
            delta_vely = vely[idx] - vely[i]
            delta_velz = velz[idx] - velz[i]

            # Solve least squares problems to find gradients
            try:
                # For velx
                grad_velx_i, _, _, _ = np.linalg.lstsq(A, delta_velx, rcond=None)
                grad_vx[i, :] = grad_velx_i

                # For vely
                grad_vely_i, _, _, _ = np.linalg.lstsq(A, delta_vely, rcond=None)
                grad_vy[i, :] = grad_vely_i

                # For velz
                grad_velz_i, _, _, _ = np.linalg.lstsq(A, delta_velz, rcond=None)
                grad_vz[i, :] = grad_velz_i

                # Compute divergence at this point
                divergence_v[i] = grad_velx_i[0] + grad_vely_i[1] + grad_velz_i[2]

            except np.linalg.LinAlgError:
                continue

        # Now assemble the Laplacian matrix L
        from scipy.sparse import lil_matrix
        L = lil_matrix((num_points, num_points))

        # Build the Laplacian matrix
        for i in range(num_points):
            idx = neighbor_indices[i, :]
            idx = idx[idx != i]
            if len(idx) == 0:
                continue

            xi = coords[:, i]
            xi = xi.reshape(3, 1)
            xj = coords[:, idx]

            # Compute weights w_{ij}
            distances_sq = np.sum((xi - xj)**2, axis=0)
            weights = 1.0 / distances_sq

            sum_weights = np.sum(weights)
            L[i, i] = sum_weights

            for j_idx, j in enumerate(idx):
                L[i, j] = -weights[j_idx]

        # Convert L to CSR format
        L = L.tocsr()

        # Solve the Poisson equation L φ = -div_v
        from scipy.sparse.linalg import spsolve

        phi = spsolve(L, -divergence_v)

        # Now compute grad φ at each point
        grad_phi = np.zeros((num_points, 3))

        for i in range(num_points):
            idx = neighbor_indices[i, :]
            idx = idx[idx != i]
            if len(idx) < 4:
                continue

            # Local coordinates relative to the point of interest
            dx = coords[0, idx] - coords[0, i]
            dy = coords[1, idx] - coords[1, i]
            dz = coords[2, idx] - coords[2, i]

            # Build the design matrix A
            A = np.column_stack((dx, dy, dz))

            # Differences in φ
            delta_phi = phi[idx] - phi[i]

            # Solve least squares problems to find gradients
            try:
                grad_phi_i, _, _, _ = np.linalg.lstsq(A, delta_phi, rcond=None)
                grad_phi[i, :] = grad_phi_i
            except np.linalg.LinAlgError:
                continue

        # Now compute the irrotational component
        vel_irrot = -grad_phi  # Negative gradient of φ

        # Compute the solenoidal component
        vel_original = np.vstack((velx, vely, velz)).T
        vel_solenoidal = vel_original - vel_irrot

        # Verify that vel_irrot + vel_solenoidal ≈ vel_original
        diff = vel_irrot + vel_solenoidal - vel_original
        max_diff = np.max(np.abs(diff))
        print(f"Maximum difference between original and reconstructed velocities: {max_diff}")

        # Save the results to a file
        output_filename = self.plt_fname(dump) + "_helmholtz_decomposition.npz"
        np.savez(output_filename,
                coords=coords,
                velx_original=velx, vely_original=vely, velz_original=velz,
                velx_irrot=vel_irrot[:, 0], vely_irrot=vel_irrot[:, 1], velz_irrot=vel_irrot[:, 2],
                velx_solenoidal=vel_solenoidal[:, 0], vely_solenoidal=vel_solenoidal[:, 1], velz_solenoidal=vel_solenoidal[:, 2])
        print(f"Helmholtz decomposition results saved to '{output_filename}'")








# ===== the following applies in case we are running this in script mode =====
if __name__ == "__main__":

    nif = NIF()
    dump_number =  1500 # Replace with your dump number
    # # nif.perform_helmholtz_decomposition_spherical(dump=dump_number,
    # #                                           grid_resolution_r=32,
    # #                                           grid_resolution_theta=32,
    # # # #                                           grid_resolution_phi=32)
    # dumps_to_process = None  # Example: [0, 10, 20, 30, 40]

    # # Compute vorticity and divergence over time and save the data
    # nif.save_vorticity_divergence_over_time(dumps=dumps_to_process, output_filename='vorticity_divergence_over_time.npz')

    # # Plot the average vorticity and divergence against time
    # # Plot the average vorticity and divergence against time
    # # nif.plot_vorticity_divergence_over_time(input_filename='vorticity_divergence_over_time.npz')
    # dump_number = 1500  # Replace with your dump number
    # nif.compute_stretching_compression(dump=dump_number,
    #                                grid_resolution_x=32,
    #                                grid_resolution_y=32,
    #                                grid_resolution_z=32)
  #  nif.read_and_print_sphere_data()# create new NIF odel object
    # nif.plot_quantities_against_time_from_file()
    
    # # Specify the dump number you want to process
    # dump_number = 1500  # Replace with your dump number

    # # # # # Perform Helmholtz decomposition
    dump_number = 1500
    # nif.perform_helmholtz_decomposition(dump=dump_number, num_neighbors=10)
    nif.compute_energy_spectra(dump = dump_number)

    # nif.plot_quantities_against_time_from_file()
    # #nif_instance = NIF()
    # decomposed_velocity = nif.helmholtz_decomposition(dump=1500, grid_resolution=50)

    # if decomposed_velocity is not None:
    #     # Access the curl-free component
    #     v_curl_free = decomposed_velocity['curl_free']
    #     grid_resolution = 50

    #     # Access the divergence-free component
    #     v_div_free = decomposed_velocity['div_free']

    #     # Access the grid
    #     X = decomposed_velocity['grid']['X']
    #     Y = decomposed_velocity['grid']['Y']
    #     Z = decomposed_velocity['grid']['Z']

    #     # For example, plot the magnitude of the divergence-free component at the mid-plane
    #     import matplotlib.pyplot as plt

    #     magnitude_div_free = np.sqrt(
    #         v_div_free['velx']**2 + v_div_free['vely']**2 + v_div_free['velz']**2
    #     )

    #     plt.imshow(
    #         magnitude_div_free[:, :, grid_resolution // 2],
    #         extent=[X.min(), X.max(), Y.min(), Y.max()],
    #         origin='lower'
    #     )
    #     plt.colorbar(label='Divergence-Free Velocity Magnitude')
    #     plt.xlabel('X')
    #     plt.ylabel('Y')
    #     plt.title('Divergence-Free Velocity Field at Mid-Plane')
    #     plt.show()
    # else:
    #     print("Helmholtz decomposition failed.")
    # # Compute stretching and compression terms
    # nif.compute_stretching_compression(dump=dump_number, num_neighbors=10)
    # # #Extract spherical data
    # nif.extract_sphere_data()
    
    # # sdat = nif.read_sphere_data(0)
    # nif.process_sphere_data()

    # nif.save_quantities_against_time()
    
    # nif.plot_more_quantities_against_time_from_file()
    # nif.plot_even_more_quantities_against_time_from_file()
    # Example of tracking a moving bounding box
    # # Example of tracking a moving bounding sphere
    # center = [0, 0, 0]  # Initial center of the bounding sphere
    # radius = 0.05  # Radius of the bounding sphere
    # nif.track_bounding_sphere_in_time(center=center, radius=radius)

    # # Read the tracked data
    # tracked_data = nif.read_tracked_data("tracked_sphere_data.pkl")

    # # # Save quantities for plotting
    # if tracked_data is not None:
    #     nif.save_tracked_quantities(tracked_data)

    # Plot the saved quantities
    # nif.plot_saved_quantities("tracked_quantities_data.npz")

    

    # Process npy files
    # nif.plot_magnetic_field_pdfs(start_index=500, end_index=500)

    
    # stop()
