import numpy as np

# Loop over the desired range of files
for i in range(0, 2499, 1):
    npy_filename = "NIF_hdf5_plt_cnt_{:04d}_sphere_data.npy".format(i)

    # Load the .npy file
    try:
        data_dict = np.load(npy_filename, allow_pickle=True).item()
    except Exception as e:
        print(f"Failed to load {npy_filename}: {e}")
        continue

    # Access the data arrays directly from the dictionary
    dens = data_dict["dens"]["data"]
    velx = data_dict["velx"]["data"]
    vely = data_dict["vely"]["data"]
    velz = data_dict["velz"]["data"]
    magx = data_dict["magx"]["data"]
    magy = data_dict["magy"]["data"]
    magz = data_dict["magz"]["data"]
    coords = data_dict["dens"]["coords"]

    # Assume a regular grid 
    dx = np.abs(coords[0, 1] - coords[0, 0])  # Difference in x-direction
    dy = np.abs(coords[1, 1] - coords[1, 0])  # Difference in y-direction
    dz = np.abs(coords[2, 1] - coords[2, 0])  # Difference in z-direction

    cell_volume = dx * dy * dz

    # Volume averaging for density
    volume_weighted_density = dens * cell_volume
    total_volume = np.sum(cell_volume)
    volume_averaged_density = np.sum(volume_weighted_density) / total_volume

    # Calculate the momentum for each velocity component at each cell
    momentum_x = velx * dens
    momentum_y = vely * dens
    momentum_z = velz * dens

    # Calculate the kinetic energy (1/2 * density * velocity^2) for each cell
    kinetic_energy = 0.5 * dens * (velx**2 + vely**2 + velz**2) * cell_volume
    total_kinetic_energy = np.sum(kinetic_energy)

    # Calculate the magnetic energy (1/2 * B^2 / μ0) for each cell, assuming μ0 = 1 for simplicity
    magnetic_energy = 0.5 * (magx**2 + magy**2 + magz**2) * cell_volume
    total_magnetic_energy = np.sum(magnetic_energy)
    # Mass averaging for velocities
    total_mass = np.sum(dens * cell_volume)
    velx_mass_avg = np.sum(momentum_x * cell_volume) / total_mass
    vely_mass_avg = np.sum(momentum_y * cell_volume) / total_mass
    velz_mass_avg = np.sum(momentum_z * cell_volume) / total_mass

    # Volume averaging for magnetic field components
    magx_volume_avg = np.sum(magx * cell_volume) / total_volume
    magy_volume_avg = np.sum(magy * cell_volume) / total_volume
    magz_volume_avg = np.sum(magz * cell_volume) / total_volume

    # Update the dictionary with mass-averaged velocities, volume-averaged magnetic field components, and volume-averaged density
    data_dict["velx_mass_avg"] = velx_mass_avg
    data_dict["vely_mass_avg"] = vely_mass_avg
    data_dict["velz_mass_avg"] = velz_mass_avg
    data_dict["magx_volume_avg"] = magx_volume_avg
    data_dict["magy_volume_avg"] = magy_volume_avg
    data_dict["magz_volume_avg"] = magz_volume_avg
    data_dict["volume_averaged_density"] = volume_averaged_density
    data_dict["total_kinetic_energy"] = total_kinetic_energy
    data_dict["total_magnetic_energy"] = total_magnetic_energy

    # Save the updated dictionary back to the .npy file
    np.save(npy_filename, data_dict)

    # Optionally print progress
    print(f"Updated and saved mass-averaged and volume-averaged data to {npy_filename}")
