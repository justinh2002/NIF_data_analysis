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
import timeit
import glob
import os
import flashplotlib as fpl

if __name__ == "__main__":

    # time the script
    start_time = timeit.default_timer()

    # files
    files = sorted(glob.glob("*_hdf5_plt_cnt_????"))

    # filter files to start from file number 1412
    files = [f for f in files if int(f.split('_')[-1]) >= 1412]

    # loop over files
    for filen in files:
        # parse arguments
        args = fpl.parse_args()
        args.vmin = 1e-3
        args.vmax = 1e0
        args.slice = True
        args.outtype = ['png']
        # process file
        fpl.process_file(filen, args)

    # time the script
    stop_time = timeit.default_timer()
    total_time = stop_time - start_time
    print("***************** time to finish = "+str(total_time)+"s *****************")

    png_files = sorted(glob.glob("*_dens_slice_z.h5_dens_slice.png"))

    

