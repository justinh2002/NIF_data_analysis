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
matplotlib.use('pdf')

class NIF:

    def __init__(self):
        self.simpath = "../simdata/high_res/"
        self.bname = "NIF"
        self.plt_files = sorted(glob.glob(self.simpath + self.bname + "_hdf5_plt_cnt_????"))

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
                datasets = ["dens", "velx", "vely", "velz", "magx", "magy", "magz"]
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
            dens = data_dict.dens
            velx = data_dict.velx
            vely = data_dict.vely
            velz = data_dict.velz
            magx = data_dict.magx
            magy = data_dict.magy
            magz = data_dict.magz
            coords = data_dict.coords

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
                        # Volume averaging for density
            volume_weighted_density = dens * cell_volume
            volume_averaged_density = np.sum(volume_weighted_density) / total_volume
            velx_mass_avg = np.sum(velx * dens * cell_volume) / total_mass
            vely_mass_avg = np.sum(vely * dens * cell_volume) / total_mass
            velz_mass_avg = np.sum(velz * dens * cell_volume) / total_mass

            # Now subtract the average from the velocity components to get the fluctuating part
            velx_prime = velx - velx_mass_avg  # u' (fluctuating x-component)
            vely_prime = vely - vely_mass_avg  # v' (fluctuating y-component)
            velz_prime = velz - velz_mass_avg  # w' (fluctuating z-component)

            # Now you can calculate the total kinetic energy of the fluctuating velocity
            kinetic_energy_prime = 0.5 * dens * (velx_prime**2 + vely_prime**2 + velz_prime**2) * cell_volume
            total_kinetic_energy_prime = np.sum(kinetic_energy_prime)

            # Calculate magnetic energy (assuming μ0 = 1)
            magnetic_energy = 0.5 * (magx**2 + magy**2 + magz**2) * cell_volume
            total_magnetic_energy = np.sum(magnetic_energy)

            # Volume averaging for magnetic field components
            magx_volume_avg = np.sum(magx * cell_volume) / total_volume
            magy_volume_avg = np.sum(magy * cell_volume) / total_volume
            magz_volume_avg = np.sum(magz * cell_volume) / total_volume

            # Create a new dictionary with processed values and simulation time
            processed_data = {
                "time": sim_time,
                "velx_mass_avg": velx_mass_avg,
                "vely_mass_avg": vely_mass_avg,
                "velz_mass_avg": velz_mass_avg,
                "velx_prime": velx_prime,
                "vely_prime": vely_prime,
                "velz_prime": velz_prime,
                "volume_averaged_density": volume_averaged_density,
                "total_kinetic_energy_prime": total_kinetic_energy_prime,
                "magx_volume_avg": magx_volume_avg,
                "magy_volume_avg": magy_volume_avg,
                "magz_volume_avg": magz_volume_avg,
                "total_magnetic_energy": total_magnetic_energy,
                "coords": coords
            }

            # Save the processed data into a new .npy file
            np.save(output_filename, processed_data)

            print(f"Processed data with time={sim_time} saved to {output_filename}")

            
    def plot_quantities_against_time(self, files=None):
        if files is None:
            files = self.plt_files  # Default to processing all available plt files

        times = []
        densities = []
        kinetic_energies = []
        magnetic_energies = []
        velx_avgs = []
        vely_avgs = []
        velz_avgs = []
        velx_dispersion = []
        vely_dispersion = []
        velz_dispersion = []
    

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

            # # Append time and other quantities to lists
            times.append(data["time"])
            densities.append(data["volume_averaged_density"])
            kinetic_energies.append(data["total_kinetic_energy_prime"])
            magnetic_energies.append(data["total_magnetic_energy"])           
                    # Compute the velocity dispersion (standard deviation of the fluctuating velocity components)

                    # Calculate the velocity dispersion for each time step
            velx_dispersion.append(np.sqrt(np.mean(data["velx_prime"]**2)))
            vely_dispersion.append(np.sqrt(np.mean(data["vely_prime"]**2)))
            velz_dispersion.append(np.sqrt(np.mean(data["velz_prime"]**2)))

        # Create the plots
        plt.figure(figsize=(10, 8))
        t = np.linspace(1e-7,3e-7,100)
        ekin =   1e1 * t ** (-1.2) 

        plt.subplot(3, 2, 1)
        plt.loglog(times, densities, label="Density")
        plt.xlabel("Time")
        plt.ylabel("Density")
        plt.title("Volume-averaged Density vs Time")
        plt.xlim(1e-7,5e-7)

        plt.subplot(3, 2, 2)
        plt.plot(t,ekin,ls = '--',label = r'$t^{-1.2}$')
        plt.loglog(times, kinetic_energies, label="Kinetic Energy", color='r')
        plt.xlabel("Time")
        plt.ylabel("Kinetic Energy")
        plt.legend()
        plt.title("Total Kinetic Energy vs Time")
        plt.xlim(1e-7,5e-7)
        plt.ylim(1e3,1e12)

        plt.subplot(3, 2, 3)
        plt.loglog(times, magnetic_energies, label="Magnetic Energy", color='g')
        plt.xlabel("Time")
        plt.ylabel("Magnetic Energy")
        plt.title("Total Magnetic Energy vs Time")
        plt.xlim(1e-7,5e-7)
        # plt.ylim(1e-8,1e-7)

        plt.subplot(3, 2, 4)
        plt.semilogy(times, velx_dispersion, label="Velx", color='b')
        plt.xlabel("Time")
        plt.ylabel("Velx dispersion")
        plt.title("Velx dispersion vs Time")

        plt.subplot(3, 2, 5)
        plt.semilogy(times, vely_dispersion, label="Vely", color='c')
        plt.xlabel("Time")
        plt.ylabel("Vely dispersion")
        plt.title("Vely dispersion vs Time")

        plt.subplot(3, 2, 6)
        plt.semilogy(times, velz_dispersion, label="Velz", color='m')
        plt.xlabel("Time")
        plt.ylabel("Velz dispersion")
        plt.title("Velz dispersion vs Time")
        

        plt.tight_layout()
        plt.savefig('time_quantities.pdf')
        
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








# ===== the following applies in case we are running this in script mode =====
if __name__ == "__main__":

    nif = NIF()
    #nif.read_and_print_sphere_data()# create new NIF model object

    # Extract spherical data
    # nif.extract_sphere_data()
    
    # sdat = nif.read_sphere_data(0)
    # nif.process_sphere_data()

    nif.plot_quantities_against_time()
    

    # Process npy files
    # nif.plot_magnetic_field_pdfs(start_index=500, end_index=500)

    
    # stop()
