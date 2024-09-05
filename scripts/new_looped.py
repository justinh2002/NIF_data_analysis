import numpy as np
import flashlib as fl
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


domain_bounds = [[-0.4, 0.4], [-0.2, 0.2], [-0.1, 0.1]]

# Define the range of intervals
start_interval = 0000
end_interval = 2490
step_size = 10

# Cylinder radius and extraction range
cylinder_radius_y = 0.05  # Cylinder radius in y direction
extraction_range = [[-0.03, 0.03], [-0.05, 0.05], [-0.15, 0.15]]  # Example extraction range

# Loop through the intervals and process the data
for interval in range(start_interval, end_interval + step_size, step_size):
    # Instantiate the class for each interval
    a = fl.FlashGG(f"NIF_hdf5_plt_cnt_{interval:04d}")
    
    # Call the function to get the volume-weighted average profile within the specified cylinder
    profile = a.GetVolumeWeightedAverageProfile(dsets="dens", extraction_range=extraction_range, cylinder_radius_y=cylinder_radius_y)
    print(f"Profile for interval {interval}: {profile}")
    
    # Unzip the profile data
    x_vals, rho_vals = profile
    
    # Plot the profile
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, rho_vals, marker='o', linestyle='-', color='b')
    plt.xlabel('$x$')
    plt.ylabel(r'$\langle \rho \rangle_V$')
    
    
    # Save the plot as a PDF with the interval in the filename
    plt.savefig(f'jps_{interval}.pdf')
