#!/usr/bin/env python
# -*- coding: utf-8 -*-
# written by Christoph Federrath, 2021

import glob
from cfpack import stop, print
import timeit
import flashplotlib as fpl
import subprocess
import os

# ===== the following applies in case we are running this in script mode =====
if __name__ == "__main__":

    # time the script
    start_time = timeit.default_timer()

    # files
    files = sorted(glob.glob("*_hdf5_plt_cnt_????"))

    # loop over files
    for filen in files:
        # parse arguments
        args = fpl.parse_args()
        
        args.vmin = 1e-11
        args.vmax = 1e-7
        args.datasetname = 'emag'
        args.slice = True
        args.outtype = ['png']
        args.cmap_label = '$|\mathbf{B}|^2$'
        # process file
        fpl.process_file(filen, args)

    # time the script
    stop_time = timeit.default_timer()
    total_time = stop_time - start_time
    print("***************** time to finish = "+str(total_time)+"s *****************")

    
    png_files = sorted(glob.glob("*_dens_slice_z.h5_dens_slice.png"))

    
    for i, png_file in enumerate(png_files):
        new_name = f"frame_{i:04d}.png"
        os.rename(png_file, new_name)

    # FFmpeg command
    ffmpeg_command = [
        "ffmpeg",
        "-framerate", "30",
        "-i", "frame_%04d.png",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "output.mp4"
    ]

    # Run the FFmpeg command
    subprocess.run(ffmpeg_command)

    print("video created")
