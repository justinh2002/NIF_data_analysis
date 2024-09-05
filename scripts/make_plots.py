import numpy as np
import matplotlib.pyplot as plt
import flashlib as fl
import os
import matplotlib
matplotlib.use('pdf')

# Define the range of timesteps (modify if necessary)
timesteps = range(0, 2499, 10)
times = np.loadtxt('times.txt')[::10]
# Arrays to store the extracted data for each timestep
# times = []
total_kinetic_energies = []
total_magnetic_energies = []
volume_averaged_densities = []
velx_mass_avg = []
vely_mass_avg = []
velz_mass_avg = []

# Loop over the timesteps and extract data
for i in timesteps:
    npy_filename = "NIF_hdf5_plt_cnt_{:04d}_sphere_data.npy".format(i)

    # Check if the file exists before trying to load it
    if not os.path.exists(npy_filename):
        print(f"File {npy_filename} does not exist. Skipping.")
        continue

    # Load the .npy file
    try:
        data_dict = np.load(npy_filename, allow_pickle=True).item()
        print(f"Loading {npy_filename}")
    except Exception as e:
        print(f"Failed to load {npy_filename}: {e}")
        continue

    # Load the FlashGG object to extract time
    # try:
    #     a = fl.FlashGG("NIF_hdf5_plt_cnt_{:04d}".format(i))
    #     time = a.scalars['time']  # Extract the simulation time
    # except Exception as e:
    #     print(f"Failed to extract time from file {i}: {e}")
    #     continue

    # Extract relevant quantities from the loaded data
    total_kinetic_energy = data_dict.get("total_kinetic_energy", None)
    total_magnetic_energy = data_dict.get("total_magnetic_energy", None)
    volume_averaged_density = data_dict.get("volume_averaged_density", None)
    velx_mass = data_dict.get("velx_mass_avg", None)
    vely_mass = data_dict.get("vely_mass_avg", None)
    velz_mass = data_dict.get("velz_mass_avg", None)
    print(f"Successfully loaded {npy_filename}")

    # If all required quantities are available, append them
    if total_kinetic_energy is not None and total_magnetic_energy is not None and volume_averaged_density is not None:
        # times.append(time)
        total_kinetic_energies.append(total_kinetic_energy)
        total_magnetic_energies.append(total_magnetic_energy)
        volume_averaged_densities.append(volume_averaged_density)
        velx_mass_avg.append(velx_mass)
        vely_mass_avg.append(vely_mass)
        velz_mass_avg.append(velz_mass)

# Plot Magnetic Energy vs. Time
plt.figure(figsize=(10, 6))
plt.plot(times, total_magnetic_energies, label="Magnetic Energy", color="blue")
plt.xlabel('Time')
plt.ylabel('Total Magnetic Energy')
plt.title('Magnetic Energy as a Function of Time')

plt.legend()
plt.savefig('emag.pdf')

# Plot Kinetic Energy vs. Time
plt.figure(figsize=(10, 6))
plt.plot(times, total_kinetic_energies, label="Kinetic Energy", color="red")
plt.xlabel('Time')
plt.ylabel('Total Kinetic Energy')
plt.title('Kinetic Energy as a Function of Time')

plt.legend()
plt.savefig('ekin.pdf')


# Plot Volume-Averaged Density vs. Time
plt.figure(figsize=(10, 6))
plt.plot(times, volume_averaged_densities, label="Volume-Averaged Density", color="green")
plt.xlabel('Time')
plt.ylabel('Volume-Averaged Density')
plt.title('Volume-Averaged Density as a Function of Time')

plt.legend()
plt.savefig('density.pdf')


# Plot Velocity Components vs. Time (Mass-Averaged)
plt.figure(figsize=(10, 6))
plt.plot(times, velx_mass_avg, label="Mass-Averaged Velx", color="purple")
plt.plot(times, vely_mass_avg, label="Mass-Averaged Vely", color="orange")
plt.plot(times, velz_mass_avg, label="Mass-Averaged Velz", color="cyan")
plt.xlabel('Time')
plt.ylabel('Mass-Averaged Velocity')
plt.title('Mass-Averaged Velocity Components as a Function of Time')

plt.legend()
plt.savefig('velocities.pdf')
