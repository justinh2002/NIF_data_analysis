#!/usr/bin/env python

from cfpack.defaults import *
import cfpack as cfp
import flashlib as fl
import numpy as np
import glob
import os
import dill


class NIF:

    def __init__(self):
        self.simpath = "../simdata/high_res/"
        self.bname = "NIF"
        self.plt_files = sorted(glob.glob(self.simpath+self.bname+"_hdf5_plt_cnt_????"))

    def plt_fname(self, dump, withpath=True):
        fname = self.bname+"_hdf5_plt_cnt_{:04d}".format(dump)
        if withpath: fname = self.simpath+fname
        return fname

    def extract_sphere_data(self, sphere_radius=0.05, sphere_center=[0,0,0], files=None, verbose=1):
        # loop over plt files
        if files is None: files = self.plt_files # default is all plt files
        for fname in files:
            outfile = fname+"_sphere_dat.pkl"
            if not os.path.isfile(outfile):
                print("extracting data from '"+fname+"'...", color="green")
                # create an instance of the FlashGG class for each file
                gg = fl.FlashGG(fname, verbose=verbose-1)
                # extract data for the required datasets
                datasets = ["dens", "velx", "vely", "velz", "magx", "magy", "magz"]
                cell_obj = gg.GetCells(dsets=datasets, sphere_radius=sphere_radius, sphere_center=sphere_center)
                class ret:
                    time = gg.scalars["time"]
                    dens = cell_obj.cell_dat[0]
                    velx = cell_obj.cell_dat[1]
                    vely = cell_obj.cell_dat[2]
                    velz = cell_obj.cell_dat[3]
                    magx = cell_obj.cell_dat[4]
                    magy = cell_obj.cell_dat[5]
                    magz = cell_obj.cell_dat[6]
                    coords = cell_obj.cell_pos
                # write to file
                with open(outfile, "wb") as fobj:
                    dill.dump(ret, fobj)
                    print("'"+outfile+"' written.", color="magenta")

    def read_sphere_data(self, dump):
        fname = self.plt_fname(dump)+"_sphere_dat.pkl"
        print("reading from '"+fname+"'...")
        ret = dill.load(open(fname, "rb"))
        return ret


# ===== the following applies in case we are running this in script mode =====
if __name__ == "__main__":

    nif = NIF() # create new NIF model object

    # generate data files with cell information inside spherical control volume
    nif.extract_sphere_data()

    sdat = nif.read_sphere_data(0)

    stop()
