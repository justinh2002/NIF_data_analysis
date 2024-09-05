import numpy as np
import flashlib as fl
import matplotlib.pyplot as plt


domain_bounds = [[-0.4, 0.4], [-0.2, 0.2], [-0.1, 0.1]]

# Define the range of intervals
start_interval = 1000
end_interval = 1500
step_size = 100

# Cylinder radius and extraction range
cylinder_radius_y = 0.05  # Cylinder radius in y direction
extraction_range = [[-0.03, 0.03], [-0.05, 0.05], [-0.15, 0.15]]  # Example extraction range

# Loop through the intervals and process the data
for interval in range(start_interval, end_interval + step_size, step_size):
    # Instantiate the class for each interval
    a = fl.FlashGG(f"NIF_hdf5_plt_cnt_{interval}")
    
    # Call the function to get the volume-weighted average profile within the specified cylinder
    profile = a.GetVolumeSphereWeightedAverageProfile(dsets="magx", extraction_range=extraction_range, sphere_radius=cylinder_radius_y)
    print(f"Profile for interval {interval}: {profile}")
    
    # Unzip the profile data
    x_vals, rho_vals = zip(*profile)
    
    # Plot the profile
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, rho_vals, marker='o', linestyle='-', color='b')
    plt.xlabel('$x$')
    plt.ylabel(r'$\langle \rho \rangle_V$')
    
    
    # Save the plot as a PDF with the interval in the filename
plt.savefig(f'jps_new.pdf')
