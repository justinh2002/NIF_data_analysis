from mpi4py import MPI
import flashlib as fl  
import glob


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Function to get and sort the plot files
def get_plot_files(pattern="NIF_hdf5_plt_cnt_????"):
    files = sorted(glob.glob(pattern))
    print(f"Rank {rank}: Found {len(files)} plot files.")
    return files

# Function to extract time scalar from a single plot file
def extract_time(file):
    gg = fl.FlashGG(file)
    time = gg.scalars['time']
    print(f"Rank {rank}: Extracted time {time} from {file}.")
    return time

# Main execution
if __name__ == "__main__":
    plot_files = get_plot_files()

    
    files_per_process = len(plot_files) // size
    start_index = rank * files_per_process
    end_index = start_index + files_per_process if rank != size - 1 else len(plot_files)
    
    times = [extract_time(file) for file in plot_files[start_index:end_index]]

    # Gathering times from all processes
    all_times = comm.gather(times, root=0)

    # Writing times to a txt file in the root process
    if rank == 0:
        
        all_times_flat = [time for sublist in all_times for time in sublist]

        with open("times.txt", "w") as file:
            for time in all_times_flat:
                file.write(f"{time}\n")

        print("Rank 0: Written times to times.txt.")
