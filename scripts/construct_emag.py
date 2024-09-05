import numpy as np
import flashlib as fl
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('pdf')


domain_bounds = [[-0.4, 0.4], [-0.2, 0.2], [-0.1, 0.1]]

# Define the range of intervals
start_interval = 0
end_interval = 2499
step_size = 1

# Cylinder radius and extraction range
cylinder_radius_y = 0.05  # Cylinder radius in y direction
extraction_range = [[-0.03, 0.03], [-0.05, 0.05], [-0.15, 0.15]]  # Example extraction range

# Initialize lists to store time intervals and corresponding spatial averages of magnetic energy
time_intervals = []
avg_mag_energy_vals = []

# Loop through the intervals and process the data
for interval in range(start_interval, end_interval + step_size, step_size):
    # Instantiate the class for each interval
    a = fl.FlashGG(f"NIF_hdf5_plt_cnt_{interval:04d}")
    time = a.scalars["time"]
    
    # Extract magnetic field components
    magx_profile = a.GetVolumeSphereWeightedAverageProfile(dsets="magx", extraction_range=extraction_range, sphere_radius=cylinder_radius_y)
    magy_profile = a.GetVolumeSphereWeightedAverageProfile(dsets="magy", extraction_range=extraction_range, sphere_radius=cylinder_radius_y)
    magz_profile = a.GetVolumeSphereWeightedAverageProfile(dsets="magz", extraction_range=extraction_range, sphere_radius=cylinder_radius_y)
    
    # Extract velocity components
    velx_profile = a.GetVolumeSphereWeightedAverageProfile(dsets="velx", extraction_range=extraction_range, sphere_radius=cylinder_radius_y)
    vely_profile = a.GetVolumeSphereWeightedAverageProfile(dsets="vely", extraction_range=extraction_range, sphere_radius=cylinder_radius_y)
    velz_profile = a.GetVolumeSphereWeightedAverageProfile(dsets="velz", extraction_range=extraction_range, sphere_radius=cylinder_radius_y)
    
    # Extract density
    dens_profile = a.GetVolumeSphereWeightedAverageProfile(dsets="dens", extraction_range=extraction_range, sphere_radius=cylinder_radius_y)
    
    # Unzip the profile data
    x_vals, magx_vals = magx_profile
    _, magy_vals = magy_profile
    _, magz_vals = magz_profile
    _, velx_vals = velx_profile
    _, vely_vals = vely_profile
    _, velz_vals = velz_profile
    _, dens_vals = dens_profile
    
    # Calculate magnetic energy (E_mag = (magx^2 + magy^2 + magz^2)/2)
    mag_energy = 0.5 * (np.array(magx_vals)**2 + np.array(magy_vals)**2 + np.array(magz_vals)**2)
    
    # Compute the spatial average of the magnetic energy
    avg_mag_energy = np.mean(mag_energy)
    
    # Store the time interval and corresponding average magnetic energy
    time_intervals.append(time)
    avg_mag_energy_vals.append(avg_mag_energy)
    
    # Save magnetic field, velocity, and density data into a single .npy file
    data_dict = {
        "x_vals": x_vals,
        "magx": magx_vals,
        "magy": magy_vals,
        "magz": magz_vals,
        "velx": velx_vals,
        "vely": vely_vals,
        "velz": velz_vals,
        "dens": dens_vals,
        "mag_energy": mag_energy
    }
    np.save(f'NIF_hdf5_plt_cnt_{interval:04d}_sphere_data.npy', data_dict)

# Plot the spatially averaged magnetic energy as a function of time
plt.figure(figsize=(10, 6))
plt.plot(time_intervals, avg_mag_energy_vals, marker='o', linestyle='-', color='r', label='Spatially Averaged Magnetic Energy')
plt.xlabel('$t$ (ns)')
plt.ylabel(r'$E_{mag}$')

# Save the plot as a PDF
plt.savefig(f'spatially_averaged_magnetic_energy_vs_time.pdf')
plt.close()

print("Data for each interval saved to .npy files and spatially averaged magnetic energy over time plotted successfully.")
