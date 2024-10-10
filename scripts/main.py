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
        self.simpath = "../simdata_tracers_long/"
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
                datasets = ["dens", "velx", "vely", "velz", "magx", "magy", "magz","pres"]
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
        
        sq_density = std_density ** 2 + densities ** 2
        C = sq_density/densities 
        
        mean_density = densities# gas clumping factor
        
        
        
        
        fig, axes = plt.subplots(1, 1, figsize=(8, 8), sharex=True)
        t = np.linspace(1e-7, 3e-7, 100)
        ekin = 1e3 * t ** (-1.2)

        # # First subplot: Kinetic Energy
        # axes[0].semilogy(times * 10**9, kinetic_energies, label="Kinetic Energy", color='r', lw=3)
        # axes[0].set_ylabel("$E_{\mathrm{kin}}$", fontsize=20)
        # axes[0].set_xlim(0, 300)
        # #axes[0].set_ylim(1e-12, 1e12)
        # #axes[0].grid(True)

        # # Second subplot: Magnetic Energy
        # axes[1].semilogy(times * 10**9, max_magnetic_energy, label="Magnetic Energy", color='g', lw=3)
        # axes[1].set_xlabel("$t$ (ns)", fontsize=20)
        # axes[1].set_ylabel("$E_{\mathrm{mag}}$", fontsize=20)
        # axes[1].set_xlim(0, 300)
        # #axes[1].grid(True)

        # Apply tight layout and show/save the figure
        plt.tight_layout()

        # plt.subplot(3, 2, 4)
        plt.semilogy(times * 10**9, mean_density, label="Velx",lw = 3)
        plt.xlabel("$t$ (ns)", fontsize=20)
        plt.ylabel(r"$ \langle  \rho \rangle$", fontsize=20)
        plt.xlim(75,300)
        # plt.ylim(1e-20,1e-17)
        
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
        plt.savefig('mean_dens.pdf')
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



# ===== the following applies in case we are running this in script mode =====
if __name__ == "__main__":

    nif = NIF()
    #nif.read_and_print_sphere_data()# create new NIF model object

    # #Extract spherical data
    # nif.extract_sphere_data()
    
    # # sdat = nif.read_sphere_data(0)
    # nif.process_sphere_data()

    # nif.save_quantities_against_time()
    
    # nif.plot_quantities_against_time_from_file()
    # Example of tracking a moving bounding box
    # # Example of tracking a moving bounding sphere
    center = [0, 0, 0]  # Initial center of the bounding sphere
    radius = 0.05  # Radius of the bounding sphere
    nif.track_bounding_sphere_in_time(center=center, radius=radius)

    # Read the tracked data
    tracked_data = nif.read_tracked_data("tracked_sphere_data.pkl")

    # Save quantities for plotting
    if tracked_data is not None:
        nif.save_tracked_quantities(tracked_data)

    # # Plot the saved quantities
    # nif.plot_saved_quantities("tracked_quantities_data.npz")

    

    # Process npy files
    # nif.plot_magnetic_field_pdfs(start_index=500, end_index=500)

    
    # stop()
