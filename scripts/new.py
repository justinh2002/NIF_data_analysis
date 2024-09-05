import matplotlib
matplotlib.use('pdf')  # Set the backend before importing pyplot

import matplotlib.pyplot as plt
import flashlib as fl
import numpy as np

domain_bounds = [[-0.4, 0.4], [-0.2, 0.2], [-0.1, 0.1]]

for i in range(0, 2499, 1):
    a = fl.FlashGG("NIF_hdf5_plt_cnt_{i:04d}".format(i=i))
    time = a.scalars['time']# Corrected formatting
    # Instantiate the class

    cylinder_radius_y = 0.05  # Cylinder radius in y direction
    extraction_range = [[-0.2, 0.2], [-0.05, 0.05], [-0.1, 0.1]]  # Example extraction range
    num_bins = 50  # Number of bins for the profile

    bin_centers, bin_means = a.GetVolumeWeightedAverageProfile(
        dsets="dens",
        extraction_range=extraction_range,
        cylinder_radius_y=cylinder_radius_y,
        num_bins=num_bins
    )

    # Plot the profile
    plt.figure(figsize=(10, 6))
    plt.semilogy(bin_centers, bin_means, marker='o', linestyle='-')
    plt.xlabel(r'$x$')
    plt.ylim(1e-2,4e-1)
    plt.ylabel(r'$\langle \rho \rangle_V$')
    plt.title(f't: {time} (ns)')

    plt.savefig('jps_{i:04d}.png'.format(i=i))  # Corrected formatting

import glob as glob
import os
import subprocess

#  # Close the plot to avoid memory issues
# png_files = sorted(glob.glob("jps_*.png"))

# # Rename PNG files to a sequential pattern
# for i, png_file in enumerate(png_files):
#     new_name = f"jps{i:04d}.png"
#     os.rename(png_file, new_name)

# # FFmpeg command
# ffmpeg_command = [
#     "ffmpeg",
#     "-framerate", "30",
#     "-i", "jps%04d.png",
#     "-c:v", "mpeg4",
#     "-pix_fmt", "yuv420p",
#     "dens.mp4"
# ]

# # Run the FFmpeg command
# subprocess.run(ffmpeg_command)

# print("Video created successfully!")