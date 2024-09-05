import flashlib as fl   
import numpy as np

# Loop over the desired range of files
for i in range(0, 2499, 1):
    # Create an instance of the FlashGG class for each file
    filename = "NIF_hdf5_plt_cnt_{:04d}".format(i)
    fn = fl.FlashGG(filename)

    # Define the center and radius of the sphere
    sphere_center = [0.0, 0.0, 0.0]  # Center of the sphere at the origin
    sphere_radius = 0.05  # Radius of the sphere

    # Extract data for the required datasets
    datasets = ["dens", "magx", "magy", "magz", "velx", "vely", "velz"]
    data_dict = {}

    for dset in datasets:
        # Extract the data and coordinates for the current dataset
        cell_data, cell_coords = fn.GetCellsInSphere(dsets=dset, sphere_center=sphere_center, sphere_radius=sphere_radius)
        # Store the data and coordinates in the dictionary
        data_dict[dset] = {
            "data": cell_data,
            "coords": cell_coords
        }

    # Save the dictionary to a .npy file
    npy_filename = "NIF_hdf5_plt_cnt_{:04d}_sphere_data.npy".format(i)
    np.save(npy_filename, data_dict)

    # Optionally print progress
    print(f"Saved data to {npy_filename}")
