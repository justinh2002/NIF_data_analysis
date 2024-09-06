#!/usr/bin/env python
# -*- coding: utf-8 -*-
# written by Christoph Federrath, 2020-2022

import numpy as np
import argparse
import cfpack as cfp
from cfpack import print, hdfio, stop
import h5py
import shutil
import scipy
import os
import scipy.stats as scstats
import inspect
from scipy.stats import binned_statistic

debug = False

# class similar to C++ FlashGG to handle Flash block data
class FlashGG:

    def __init__(self, filename, verbose=1):
        self.filename = filename
        self.verbose = verbose
        # determine grid type
        self.GridType = self.GetGridType()
        if self.verbose > 1: print("GridType = "+self.GridType)
        # determine boundary type
        self.BoundaryType = self.GetBoundaryType()
        if self.verbose > 1: print("BoundaryType = "+self.BoundaryType)
        # extracted grid not fully implemented
        if self.GridType == 'E': return
        # read bounding box
        self.BoundingBox = hdfio.read(self.filename, "bounding box")
        # set number of blocks
        self.NumBlocks = np.shape(self.BoundingBox)[0]
        self.NumBlocksRep = self.NumBlocks
        # read node type
        self.NodeType = self.ReadNodeType()
        # create list of leaf blocks
        leaf_blocks = []
        for b in range(self.NumBlocks): # loop over all blocks
            if self.NodeType[b] == 1: # if it is a leaf block
                leaf_blocks.append(b)
        self.LeafBlocks = np.array(leaf_blocks)
        # number of leaf blocks
        self.NumLeafBlocks = len(self.LeafBlocks)
        # read refine level
        self.RefineLevel = self.ReadRefineLevel()
        # read runtime parameters
        self.runtime_params = read_runtime_parameters(self.filename)
        # read scalars
        self.scalars = read_scalars(self.filename)
        # Ndim
        self.Ndim = self.scalars['dimensionality']
        # set number of cells per block (NB)
        self.NB = np.zeros(3).astype(int)
        self.NB[0] = self.scalars['nxb']
        self.NB[1] = self.scalars['nyb']
        self.NB[2] = self.scalars['nzb']
        # base grid resolution (number of cells)
        self.NBaseGrid = np.array([])
        for i in range(3):
            if self.GridType == 'A': parname = ["nblockx", "nblocky", "nblockz"]
            if self.GridType == 'U': parname = ["iprocs", "jprocs", "kprocs"]
            self.NBaseGrid = np.append(self.NBaseGrid, self.runtime_params[parname[i]]*self.NB[i])
        self.NBaseGrid = self.NBaseGrid.astype(int) * 2**(self.RefineLevel.min()-1)
        # maximum effective resolution (number of cells)
        self.NMax = self.NBaseGrid * 2**(self.RefineLevel.max()-1)
        # size of each block
        self.LB = self.BoundingBox[:,:,1]-self.BoundingBox[:,:,0]
        # set cell size of each block
        self.D = self.LB / self.NB
        # size of domain
        self.L = np.array([np.max(self.BoundingBox[:,x,:])-np.min(self.BoundingBox[:,x,:]) for x in range(3)])
        # domain bounds
        self.domain_bounds = np.array([[self.runtime_params['xmin'], self.runtime_params['xmax']],
                                       [self.runtime_params['ymin'], self.runtime_params['ymax']],
                                       [self.runtime_params['zmin'], self.runtime_params['zmax']]])
        # define an analysis centre (e.g., for radial profiles; call to SetAnalysisCentre)
        self.centre = np.mean(self.domain_bounds, axis=1) # center of the simulation box
        #self.hist_2d_buffer = None # buffer storage for 2D histograms

    # set centre (e.g., for making radial profiles); centre must be list or array of 3 elements
    def SetCentre(self, centre=None):
        c = np.array(centre)
        if len(c) != 3: print("centre must have 3 elements for x, y, z", error=True)
        self.centre = c

    # get grid type (extracted: 'E', AMR: 'A', uniform: 'U')
    def GetGridType(self):
        grid_type = None
        if "minmax_xyz" in hdfio.get_dataset_names(self.filename): grid_type = 'E'
        else:
            runtime_params = read_runtime_parameters(self.filename)
            if 'lrefine_max' in runtime_params.keys(): grid_type = 'A'
            else: grid_type = 'U'
        return grid_type

    # get boundary type (extracted: 'E', AMR: 'A', uniform: 'U')
    def GetBoundaryType(self):
        boundary_type = 'isolated'
        if self.GridType == 'E': # E grid
            if "pbc" in hdfio.get_dataset_names(self.filename):
                if hdfio.read(self.filename, "pbc") == 1: boundary_type = 'periodic'
        else: # A or U grid
            runtime_params = read_runtime_parameters(self.filename)
            if runtime_params["xl_boundary_type"] == 'periodic': boundary_type = 'periodic'
        return boundary_type

    # get uniform grid
    def GetUniformGrid(self, dset="dens"):
        if self.NumBlocks != self.NumLeafBlocks:
            print("highest refinement level is not a uniform grid", error=True)
        ret = np.zeros(self.NMax) # empty return array of full size
        LB = self.LB # fix 1D and 2D, which have zero-sixed LB directions
        # get block position mulitplier in domain
        for b in range(self.NumBlocks): LB[b,np.where(LB[b]==0.0)[0]] = 1.0
        block_pos_ind = np.round((self.BoundingBox[:,:,0] - self.domain_bounds[:,0]) / LB).astype(int)
        for b in range(self.NumBlocks):
            # put block into ret array
            ib = block_pos_ind[b]*self.NB
            ie = ib + self.NB
            ret[ib[0]:ie[0],ib[1]:ie[1],ib[2]:ie[2]] = self.ReadBlockVar(b, dsets=dset)
        return ret.squeeze() # return with irrelevant dimensions removed

    # read entire variable (all blocks)
    # Note that this will create an array that may be too big if the simulation contains too many blocks and cells
    # Use ReadBlockVar instead, to read the data block-by-block
    def ReadVar(self, dsets="dens"):
        if type(dsets) != list: dsets = [dsets] # if user requests only a single dset, turn into list of 1 element for internal processing
        data = [] # prep return array
        for dset in dsets:
            all_blocks_data = []
            for b in range(self.NumBlocks):
                all_blocks_data.append(self.ReadBlockVar(b, dset))
            data.append(all_blocks_data) # append to output list
        if len(dsets) == 1: data = data[0] # strip first dim if only 1 dset was requested
        return np.array(data)

    # read block variable
    def ReadBlockVar(self, block, dsets="dens"):
        if type(dsets) != list: dsets = [dsets] # if user requests only a single dset, turn into list of 1 element for internal processing
        b = block % self.NumBlocks # dealing with PBCs if looping on block replica indices
        data = [] # prep return array
        for dset in dsets:
            # check if dset is in file
            if dset in hdfio.get_dataset_names(self.filename):
                data.append(hdfio.read(self.filename, dset, ind=b).T) # read directly from file
            else:
                data.append(self.CreateDerivedBlockVar(block, dset)) # create a derived variable
        if len(dsets) == 1: data = data[0] # strip first dim if only 1 dset was requested
        return np.array(data)

    # create derived block variable
    def CreateDerivedBlockVar(self, block, dset="radius"):
        if dset == "radius":
            coords = np.array(self.GetCellCoords(block)) # get cell coordinates (x, y, z in first index)
            coords = (coords.T - self.centre).T # subtract the centre
            return np.sqrt(np.sum(coords**2, axis=0)) # return radial distance to centre
        else:
            print("derived dset = '"+dset+"' unknown; not implemented yet", error=True)

    # read bounding box
    def ReadBoundingBox(self):
        return hdfio.read(self.filename, "bounding box")

    # read node type
    def ReadNodeType(self):
        return hdfio.read(self.filename, "node type")

    # read refinement level
    def ReadRefineLevel(self):
        return hdfio.read(self.filename, "refine level")

    # get information on refinement (number of blocks and volume filling fraction on each level)
    def GetRefinementInfo(self):
        print("min/max level = "+str(self.RefineLevel.min())+", "+str(self.RefineLevel.max()))
        levels = np.unique(self.RefineLevel)
        nbl = []
        for level in levels: nbl.append((self.RefineLevel == level).sum())
        nbl = np.array(nbl)
        frac = nbl/self.NumBlocks
        for il, level in enumerate(levels):
            print("Level "+str(level)+": nblocks="+str(nbl[il])+" (fraction: "+cfp.round(frac[il]*100, str_ret=True)+"%)")
        return nbl, frac

    # get the cell coordinates of a block
    def GetCellCoords(self, block):
        b = block % self.NumBlocks # dealing with PBCs if looping on block replica indices
        x,y,z = cfp.get_3d_coords(cmin=self.BoundingBox[b,:,0], cmax=self.BoundingBox[b,:,1], ndim=self.NB, cell_centred=True)
        return x,y,z

    # get integral quantity of a dataset
    def GetIntegralQuantity(self, datasetname="dens", statistic="mean", weighting="volume"):
        # catch user settings
        statistics = ["min", "max", "mean", "std"]
        if statistic not in statistics: print("statistic='"+statistic+"' not supported; statistic must be any of", statistics, error=True)
        weightings = ["volume", "mass"]
        if weighting not in weightings: print("weighting='"+weighting+"' not supported; weighting must be any of", weightings, error=True)
        # init accumulators
        dsum1 = 0.0
        dsum2 = 0.0
        wsum = 0.0
        dmin = +1e99
        dmax = -1e99
        # accumulate (loop over all blocks)
        for b in range(self.NumBlocks):
            if self.NodeType[b] == 1: # leaf block
                dset = self.ReadBlockVar(b, datasetname)
                cell_vol = np.prod(self.D[b])
                if weighting == "volume": weight = np.full(dset.shape, cell_vol)
                if weighting == "mass": weight = self.ReadBlockVar(b, "dens") * cell_vol
                if statistic == "mean":
                    dsum1 += np.sum(dset*weight)
                    wsum += np.sum(weight)
                if statistic == "std":
                    dsum1 += np.sum(dset*weight)
                    dsum2 += np.sum(dset**2*weight)
                    wsum += np.sum(weight)
                if statistic == "min":
                    dmin = np.min([dmin, np.min(dset)])
                if statistic == "max":
                    dmax = np.max([dmax, np.max(dset)])
        # finally, define the return values for each statistics case
        ret = None # default return value
        if statistic == "mean": ret = dsum1 / wsum
        if statistic == "std": ret = np.sqrt( (dsum2/wsum) - (dsum1/wsum)**2 )
        if statistic == "min": ret = dmin
        if statistic == "max": ret = dmax
        return ret

    # get min of a dataset
    def GetMin(self, datasetname="dens"):
        return self.GetIntegralQuantity(datasetname, statistic='min')
    # get max of a dataset
    def GetMax(self, datasetname="dens"):
        return self.GetIntegralQuantity(datasetname, statistic='max')

    # Following function perform different statistical analyses (mean, percentile, standard deviation) of one dataset w.r.t. other based on 2D histogram
    # Look at https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic_2d.html for detailed documentation
    # dsetx/dsety includes dsets available from pltfiles and 'radius'. Statistical analysis performed across dsety for different dsetx.
    # For instance, in the default scenario, this function outputs vol-weighted mean of densities as a function of radius.
    # Any radial analysis can be performed over different centres in the domain - specify centre (1D-array), if None, domain centre is taken.
    # weight should be 'vol', 'mass'. stat should be 'mean', 'std', 'per_<insert percentile desired>' (e.g. per_50 for median).
    # bins should be integers (no. of bins) or arrays (bin edges). bin_range should contain binning range of dsetx and dsety.
    # If bin_range is None, global max, min of dsetx, dsety used. For fast computation, define either bin_range or bin edges (in bins).
    # x_bintype, y_bintype should be 'uni' (uniform bins), 'log' (logarithmic bins). To remove NaN from output make remove_nan=True.

    def binned_statistic_2d(self, dsetx='radius', dsety='dens', centre=None, weight='vol', stat='mean', bins=[300, 200], bin_range=[[None, None], [None, None]],
                            x_bintype='uni', y_bintype='log', remove_nan=False):

        def hist2d(dsetx='radius', dsety='dens', centre=None, weight='vol', stat='mean', bins=[300, 200], bin_range=[[None, None], [None, None]],
                   x_bintype='uni', y_bintype='log', remove_nan=False):
            frame = inspect.currentframe(); _, _, _, values = inspect.getargvalues(frame) # Storing the arguments for buffer
            if centre is not None: self.SetCentre(centre) # defining the centre
            if isinstance(bins[0], int) == True or isinstance(bins[1], int) == True: # if bins are integers
                if bin_range==[[None, None] , [None, None]]: # define bin_range if not defined
                    if dsetx=='radius' or dsety=='radius':
                        rmin = 0
                        rmax = np.linalg.norm(np.max(np.abs((self.centre-self.domain_bounds.T).T), axis=1))
                        if dsetx=='radius': print('binning r VS '+dsety); bin_range = [[rmin, rmax], [0.8*self.GetMin(dsety), 1.2*self.GetMax(dsety)]]
                        if dsety=='radius': print('binning '+dsetx+' VS r'); bin_range = [[0.8*self.GetMin(dsetx), 1.2*self.GetMax(dsetx)], [rmin, rmax]]
                    else: print('binning '+dsetx+' VS '+dsety); bin_range = [[0.8*self.GetMin(dsetx), 1.2*self.GetMax(dsetx)], [0.8*self.GetMin(dsety), 1.2*self.GetMax(dsety)]]
                # define bin_xedges, bin_yedges if bin_range is defined
                if x_bintype=='uni': bin_xedges = np.linspace(bin_range[0][0], bin_range[0][1], bins[0])
                if x_bintype=='log': bin_xedges = np.logspace(np.log10(bin_range[0][0]), np.log10(bin_range[0][1]), num=bins[0])
                if y_bintype=='uni': bin_yedges = np.linspace(bin_range[1][0], bin_range[1][1], bins[1])
                if y_bintype=='log': bin_yedges = np.logspace(np.log10(bin_range[1][0]), np.log10(bin_range[1][1]), num=bins[1])
                print('bin edges determined!')
            else: print('bin edges already defined.'); bin_xedges = np.array(bins[0]); bin_yedges = np.array(bins[1])
            leaf_blocks = np.where(self.NodeType==1)[0].astype('int') # only leaf blocks
            final_weight_hist_arr = np.zeros([len(bin_xedges)-1, len(bin_yedges)-1]) # pre-allocating the final 2D weights array
            print('Computing 2D histogram of '+dsetx+' VS '+dsety)
            num = 0; lim = 10 # Variables to display the loading screen
            for i in leaf_blocks:
                num = num+1
                if num/len(leaf_blocks)*100>=lim: print(str(lim)+'% blocks analyzed...'); lim=lim+10 # Loading screen
                dsetx_arr = self.ReadBlockVar(i, dsets=dsetx); dsety_arr = self.ReadBlockVar(i, dsets=dsety) # contains all dataset values
                vol = np.prod(self.D[i]); vol_arr = np.zeros([self.NB[0], self.NB[1], self.NB[2]])+vol # contains all volume values
                # weight array is calculated based on masses or volumes
                if weight=='vol': weight_arr = vol_arr
                if weight=='mass': weight_arr = vol_arr*self.ReadBlockVar(i, dsets='dens')
                # For 2D binning, making the dset_arr, weight_arr 1D is necessary
                dsetx_arr = dsetx_arr.ravel(); dsety_arr = dsety_arr.ravel(); weight_arr = weight_arr.ravel()
                # Computing the 2D histogram of dsetx VS dsety with weights summed
                weight_hist_arr = scstats.binned_statistic_2d(dsetx_arr, dsety_arr, statistic='sum', values=weight_arr, bins=[bin_xedges, bin_yedges])[0]
                final_weight_hist_arr = final_weight_hist_arr+weight_hist_arr
            self.hist_2d_buffer = {'args':values.keys(), 'values':values, 'weights2d':final_weight_hist_arr, 'bin_xedges':bin_xedges, 'bin_yedges':bin_yedges}

        frame = inspect.currentframe(); args, _, _, values = inspect.getargvalues(frame) # Getting the input arguments
        exclude_keys = ['self', 'stat', 'frame', 'remove_nan', 'hist2d'] # This keys are to be excluded when checking if 2d histogram needs to be computed based on its arguments
        if self.hist_2d_buffer is None:
            print('generating the first 2d histogram')
            hist2d(dsetx=dsetx, dsety=dsety, centre=centre, weight=weight, stat=stat, bins=bins, bin_range=bin_range, x_bintype=x_bintype, y_bintype=y_bintype, remove_nan=remove_nan)
        else:
            chk = []
            for i in values.keys():
                if i in exclude_keys: continue
                else: chk.append(self.hist_2d_buffer['values'][i]==values[i])
            if all(chk): print('2d histogram already exists')
            else:
                print('generating a different 2d histogram')
                hist2d(dsetx=dsetx, dsety=dsety, centre=centre, weight=weight, stat=stat, bins=bins, bin_range=bin_range, x_bintype=x_bintype, y_bintype=y_bintype, remove_nan=remove_nan)
        weights2d = self.hist_2d_buffer['weights2d']; bin_xedges = self.hist_2d_buffer['bin_xedges']; bin_yedges = self.hist_2d_buffer['bin_yedges'] # variable assignment
        def CDF_per(xarray, yarray, per):
            ycdf = 100*(np.cumsum(yarray)/np.sum(yarray)); idx = np.max(np.where(ycdf<=per)[0])
            return xarray[idx]
        def edge2cen(array, cond):
            array1 = np.roll(array, 1)
            if cond=='uni': cen_array = (array[1:]+array1[1:])/2
            if cond=='log': cen_array = 10**((np.log10(array[1:])+np.log10(array1[1:]))/2)
            return cen_array
        good_ind = np.sum(weights2d, axis=1)>0
        dsetx_cen = edge2cen(bin_xedges, x_bintype); dsety_cen = edge2cen(bin_yedges, y_bintype)
        ret = np.full(shape=len(dsetx_cen), fill_value=np.nan)
        if stat=='mean':
            print('Computing the '+weight+'-weighted mean of '+dsetx+' VS '+dsety)
            # First term is the final weights for the mean, second term are the densities
            ret[good_ind] = np.sum(weights2d*dsety_cen, axis=1)[good_ind]/np.sum(weights2d, axis=1)[good_ind]
        if stat=='std':
            print('Computing the standard deviation of '+dsetx+' VS '+dsety)
            mean = np.full(shape=len(dsetx_cen), fill_value=np.nan)
            mean[good_ind] = np.sum(weights2d*dsety_cen, axis=1)[good_ind]/np.sum(weights2d, axis=1)[good_ind]
            ret[good_ind] = np.sum(weights2d*dsety_cen**2, axis=1)[good_ind]/np.sum(weights2d, axis=1)[good_ind]
            temp = ret[good_ind]-mean[good_ind]**2; temp[temp<0]=0
            ret[good_ind] = np.sqrt(temp)
        if stat.split('_')[0]=='per':
            print('Computing the '+stat.split('_')[1]+'th percentile of '+dsety+' VS '+dsetx)
            percentile = float(stat.split('_')[1])
            for i in range(len(good_ind)):
                if good_ind[i]: ret[i] = CDF_per(dsety_cen, weights2d[i,:], percentile)
        if remove_nan: dsetx_cen = dsetx_cen[good_ind]; ret = ret[good_ind]
        return dsetx_cen, ret

    # get location of maximum of a dataset
    def GetMaxLoc(self, datasetname="dens"):
        max_loc = None
        maximum = 0.0
        for b in range(self.NumBlocks):
            if self.NodeType[b] == 1: # leaf block
                dset = self.ReadBlockVar(b, datasetname)
                if np.max(dset) > maximum:
                    maximum = np.max(dset)
                    array = np.where(dset == maximum)
                    max_ind = np.array([val[0] for val in array]) # take the first element that is at max (in case there are more)
                    max_loc = self.BoundingBox[b,:,0] + (max_ind[:]+0.5)*self.D[b,:]
        return max_loc

    # add block replicas to simplify handling of periodic boundary conditions (PBCs)
    def AddBlockReplicasPBCs(self):
        bb = np.copy(self.BoundingBox)
        nt = np.copy(self.NodeType)
        rl = np.copy(self.RefineLevel)
        pbc = [0,0,0]
        pbc_factor = [[0],[0],[0]]
        for dim in range(self.Ndim): pbc_factor[dim] = [-1,0,1]
        for pbc[2] in pbc_factor[2]:
            for pbc[1] in pbc_factor[1]:
                for pbc[0] in pbc_factor[0]:
                    if pbc[0]==0 and pbc[1]==0 and pbc[2]==0: continue # do not process the original
                    bbc = np.copy(bb)
                    for dim in range(self.Ndim): bbc[:,dim,:] += pbc[dim]*self.L[dim]
                    self.BoundingBox = np.append(self.BoundingBox, bbc, axis=0) # append to BoundingBox
                    self.NodeType = np.append(self.NodeType, nt, axis=0) # append to NodeType
                    self.RefineLevel = np.append(self.RefineLevel, rl, axis=0) # append to RefineLevel
        self.NumBlocksRep = np.shape(self.BoundingBox)[0] # update NumBlocksRep

    # return list of blocks overlapping with bounds
    def GetAffectedBlocks(self, bounds=None):
        if bounds is None: use_bounds = np.copy(self.domain_bounds)
        else: use_bounds = np.copy(bounds)
        affected_blocks = [] # start with empty return container
        for b in range(self.NumBlocks): # loop over all blocks
            if self.NodeType[b] == 1: # if it is a leaf block
                overlap = True
                for dim in range(3):
                    if self.BoundingBox[b,dim,1] <= use_bounds[dim,0]:
                        overlap = False; break
                    if self.BoundingBox[b,dim,0] >= use_bounds[dim,1]:
                        overlap = False; break
                if overlap: affected_blocks.append(b)
        return np.array(affected_blocks)
    
    def GetAffectedBlocksInSphere(self, sphere_center, sphere_radius):
        affected_blocks = []  # start with empty return container
        for b in range(self.NumBlocks):  # loop over all blocks
            if self.NodeType[b] == 1:  # if it is a leaf block
                overlap = False
                min_dist_squared = 0  # Initialize squared minimum distance to 0
                for dim in range(3):
                    # Check the distance between the sphere center and the block's bounding box in each dimension
                    if sphere_center[dim] < self.BoundingBox[b, dim, 0]:
                        min_dist_squared += (self.BoundingBox[b, dim, 0] - sphere_center[dim]) ** 2
                    elif sphere_center[dim] > self.BoundingBox[b, dim, 1]:
                        min_dist_squared += (sphere_center[dim] - self.BoundingBox[b, dim, 1]) ** 2
                
                # If the minimum distance from the sphere center to the block is less than the sphere radius, we have an overlap
                if min_dist_squared <= sphere_radius ** 2:
                    overlap = True
                
                if overlap:
                    affected_blocks.append(b)
                    
        return np.array(affected_blocks)


    # extract data in rectangular region
    @cfp.timer_decorator
    def GetCellsInRectangle(self, dsets="dens", extraction_range=None): #bounding box keyword
        if type(dsets) != list: dsets = [dsets] # if user only supplies a single dset, turn into list of 1 element for internal processing
        if extraction_range is None: extraction_range = self.domain_bounds # use full domain if extraction_range not set
        blocks = self.GetAffectedBlocks(extraction_range) # get overlapping blocks
        # init return data containers
        n_max_out = len(blocks)*self.NB.prod()
        cell_datas = np.empty((len(dsets),n_max_out)); cell_datas.fill(np.nan)
        cell_coord = np.empty((3,n_max_out)); cell_coord.fill(np.nan)
        ibeg = 0 # start index of selection for output array
        # loop blocks
        for ib, b in enumerate(blocks): # loop over all blocks
            if self.verbose: print("working on block # "+str(ib+1)+" of total affected blocks "+str(len(blocks)))
            x, y, z = self.GetCellCoords(b) # get cell coordinate of this block
            # check if in range
            ind =   (x > extraction_range[0][0]) & (x < extraction_range[0][1]) & \
                    (y > extraction_range[1][0]) & (y < extraction_range[1][1]) & \
                    (z > extraction_range[2][0]) & (z < extraction_range[2][1])
            n_cells = ind.sum() # number of affected cells (number of True values in ind)
            if n_cells > 0: # only extract if we have selected cells in extraction range
                iend = ibeg + n_cells # end index of output array for this selection of cells
                block_datas = self.ReadBlockVar(b, dsets) # read full block variable
                if len(dsets) == 1: block_datas = np.array([block_datas]) # turn into list if we only extract a single dset
                cell_datas[:,ibeg:iend] = block_datas[:,ind] # only select cells within the range and copy into right place in output
                cell_coord[:,ibeg:iend] = [x[ind], y[ind], z[ind]] # coordinates
                ibeg = iend # new start index for next selection output
        cell_coord = cell_coord[:,0:iend] # trim away all NaNs at the end of the array
        cell_datas = cell_datas[:,0:iend] # trim away all NaNs at the end of the array
        if len(dsets) == 1: cell_datas = cell_datas[0,:] # strip first dim if only 1 dset was requested
        return cell_datas, cell_coord
    
    def GetCellsInSphere(self, dsets="dens", sphere_center=None, sphere_radius=None):
        if type(dsets) != list: dsets = [dsets]  # If user only supplies a single dataset, turn into list of 1 element for internal processing
        if sphere_center is None or sphere_radius is None:
            raise ValueError("sphere_center and sphere_radius must be provided.")
        
        blocks = self.GetAffectedBlocksInSphere(sphere_center, sphere_radius)  # Get overlapping blocks
        
        # Initialize return data containers
        n_max_out = len(blocks) * self.NB.prod()
        cell_datas = np.empty((len(dsets), n_max_out))
        cell_datas.fill(np.nan)
        cell_coord = np.empty((3, n_max_out))
        cell_coord.fill(np.nan)
        
        ibeg = 0  # start index of selection for output array
        
        # Loop blocks
        for ib, b in enumerate(blocks):  # loop over all blocks
            if self.verbose: 
                print("working on block # " + str(ib + 1) + " of total affected blocks " + str(len(blocks)))
            
            x, y, z = self.GetCellCoords(b)  # get cell coordinates of this block
            
            # Check if in range (inside the sphere)
            distances_squared = (x - sphere_center[0]) ** 2 + (y - sphere_center[1]) ** 2 + (z - sphere_center[2]) ** 2
            ind = distances_squared <= sphere_radius ** 2
            n_cells = ind.sum()  # number of affected cells (number of True values in ind)
            
            if n_cells > 0:  # only extract if we have selected cells in extraction range
                iend = ibeg + n_cells  # end index of output array for this selection of cells
                block_datas = self.ReadBlockVar(b, dsets)  # read full block variable
                if len(dsets) == 1: 
                    block_datas = np.array([block_datas])  # turn into list if we only extract a single dataset
                
                cell_datas[:, ibeg:iend] = block_datas[:, ind]  # only select cells within the range and copy into the right place in the output
                cell_coord[:, ibeg:iend] = [x[ind], y[ind], z[ind]]  # coordinates
                ibeg = iend  # new start index for the next selection output
        
        cell_coord = cell_coord[:, 0:iend]  # trim away all NaNs at the end of the array
        cell_datas = cell_datas[:, 0:iend]  # trim away all NaNs at the end of the array
        
        if len(dsets) == 1: 
            cell_datas = cell_datas[0, :]  # strip first dimension if only 1 dataset was requested
        
        return cell_datas, cell_coord



    def IsWithinCylinder(self, y, z, cylinder_radius_y):
        # Calculate the distance from each cell to the cylinder axis in the y direction
        dist_to_axis = np.sqrt(y**2 + z**2)
        
        # Check if the cells are within the radius in y and z
        within_radius = dist_to_axis < cylinder_radius_y
        
        return within_radius
    def IsWithinSphere(self, x, y, z, sphere_radius):
        # Calculate the distance from each cell to the sphere center
        dist_to_center = np.sqrt(x**2 + y**2 + z**2)

        # Check if the cells are within the radius
        within_radius = dist_to_center < sphere_radius

        return within_radius

    def VolumeSphereWeightedAverageProfile(self, cell_datas, cell_coord, sphere_radius, num_bins=50):
        x_coords = cell_coord[0]
        y_coords = cell_coord[1]
        z_coords = cell_coord[2]
        rho_vals = cell_datas

        within_sphere = self.IsWithinSphere(x_coords, y_coords, z_coords, sphere_radius)

        # Filter the data to those within the sphere
        x_within = x_coords[within_sphere]
        rho_within = rho_vals[within_sphere]

        # Compute the binned statistic (mean) for the profile
        bin_means, bin_edges, binnumber = binned_statistic(x_within, rho_within, statistic='mean', bins=num_bins)

        # Calculate bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        return bin_centers, bin_means

    # Example usage
    def GetVolumeSphereWeightedAverageProfile(self, dsets="dens", extraction_range=None, sphere_radius=0.05, num_bins=50):
        cell_datas, cell_coord = self.GetCellsInSphere(dsets, extraction_range)
        bin_centers, bin_means = self.VolumeWeightedAverageProfile(cell_datas, cell_coord, sphere_radius, num_bins)
        return bin_centers, bin_means    
    
    def VolumeWeightedAverageProfile(self, cell_datas, cell_coord, cylinder_radius_y, num_bins=50):
        x_coords = cell_coord[0]
        y_coords = cell_coord[1]
        z_coords = cell_coord[2]
        rho_vals = cell_datas

        within_cylinder = self.IsWithinCylinder(y_coords, z_coords, cylinder_radius_y)
        
        # Filter the data to those within the cylinder
        x_within = x_coords[within_cylinder]
        rho_within = rho_vals[within_cylinder]

        # Compute the binned statistic (mean) for the profile
        bin_means, bin_edges, binnumber = binned_statistic(x_within, rho_within, statistic='mean', bins=num_bins)
        
        # Calculate bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        return bin_centers, bin_means
    
    # Example usage
    def GetVolumeWeightedAverageProfile(self, dsets="dens", extraction_range=None, cylinder_radius_y=0.05, num_bins=50):
        cell_datas, cell_coord = self.GetCellsInRectangle(dsets, extraction_range)
        bin_centers, bin_means = self.VolumeWeightedAverageProfile(cell_datas, cell_coord, cylinder_radius_y, num_bins)
        return bin_centers, bin_means



# class similar to C++ FlashParticles to handle Flash particles
class Particles:

    def __init__(self, filename, verbose=1):
        self.verbose = verbose
        filen = filename
        if is_plt_file(filen): # if plt file, we assume the part file has the same number
            filen = filen.replace("_hdf5_plt_cnt_", "_hdf5_part_")
        if not os.path.isfile(filen): # get out of here if there is no particle file
            self.n = 0
            return
        if self.verbose: print("Using '"+filen+"' as input file.")
        self.filename = filen
        # set number of particles and particle properties
        self.n = 0
        self.props = None
        dsets = hdfio.get_dataset_names(self.filename)
        if "particle names" in dsets:
            names = hdfio.read(self.filename, "particle names")
            # set particle properties
            self.props = [x[0].strip().decode() for x in names]
            self.n = hdfio.get_shape(self.filename, "tracer particles")[0]
        # guess the particle types
        self.set_type()

    # set/guess the particle types
    def set_type(self):
        if self.verbose: print("determining particle types...")
        self.tracer_type = None
        self.sink_type = None
        if self.props is None: return # return if there are no particles
        self.types = [1]
        self.n_by_type = [self.n]
        if 'type' in self.props:
            types = self.read(prop='type')
            self.types = np.unique(types).astype(int)
            if self.verbose: print("file '"+self.filename+"' contains "+str(len(self.types))+" particle type(s) = "+str(self.types))
            if 'accr_rate' in self.props:
                self.tracer_type = 1
                self.sink_type = 2
            self.n_by_type = [(types==self.tracer_type).sum(), (types==self.sink_type).sum()]
            if self.verbose:
                if self.tracer_type is not None:
                    print("Guessing tracer / dark matter particle type is "+str(self.tracer_type)+" with "+str(self.n_by_type[self.tracer_type-1])+" particle(s).")
                if self.sink_type is not None: print("Guessing sink particle type is "+str(self.sink_type)+" with "+str(self.n_by_type[self.sink_type-1])+" particle(s).")
        else:
            if 'accr_rate' in self.props:
                self.sink_type = 1
            else:
                self.tracer_type = 1
            if self.verbose: print("There is only one particle type present.")
        return

    # print info
    def print_info(self):
        print("Particle filename = '"+self.filename+"'")
        print("Total number of particles = "+str(self.n))
        print("Particle types = ", self.types)
        if len(self.types) > 1:
            print("Number of tracer / dark matter particles = "+str(self.n_by_type[self.tracer_type-1])+" (tracer_type="+str(self.tracer_type)+")")
            print("Number of sink particles = "+str(self.n_by_type[self.sink_type-1])+" (sink_type="+str(self.sink_type)+")")
        return

    # read particles into dictionary (if prop is provided, read only that single particle property)
    def read(self, prop=None, type=None):
        if self.n == 0: return None # return if we don't have any particles
        # if user wants a specific type, we first check whether the type is actually present
        if type is not None:
            error = False
            if len(self.types) > 1: # there are actually mutliple types
                if type not in self.types: error = True # user requested a non-existent type
            elif type != 1: error = True # user requested a non-existent type
            if error: # return
                print("Error. Particle type "+str(type)+" not in file (available types are", self.types, "). Returning None.")
                return None
        # user requests a specific particle property (prop)
        if prop is not None:
            # find requested particle property index
            index = np.argwhere(np.array(self.props) == prop).flatten()[0]
            ind = np.s_[:,index] # use index to pass to hdfio for hyperslab selection
            particles = hdfio.read(self.filename, "tracer particles", ind=ind)
            if (type is None) or (len(self.types)==1):
                return particles # return particle property without type distinction
            else: # there are in fact different types in file
                index = np.argwhere(np.array(self.props) == "type").flatten()[0]
                ind = np.s_[:,index] # use index to pass to hdfio for hyperslab selection
                particles_type = hdfio.read(self.filename, "tracer particles", ind=ind)
                return particles[particles_type==type]
        else: # we read all particles properties and return a dictionary
            particles = hdfio.read(self.filename, "tracer particles")
            p_dict = dict(zip(self.props, particles.T))
            if (type is None) or (len(self.types)==1):
                return p_dict # return the whole particle dict without type distinction
            else: # there are in fact different types in file
                ind = np.where(p_dict["type"]==type)
                p_dict_type = {k:p_dict[k][ind] for k in p_dict.keys()}
                if self.verbose: print("Returning dictionary only containing type "+str(type))
                return p_dict_type

    # return indices of particles in requested bounding box (bb[3,2] with 1st index x,y,z, and 2nd index min,max)
    def find(self, bb=None, type=None):
        if bb is None:
            print("Need to provide bounding box for search; keyword bb=[3,2], with 1st index x,y,z, and 2nd index min,max.")
            return None
        bb = np.array(bb) # turn into numpy array
        particles = self.read(type=type) # read particles
        pos = np.array([particles["posx"], particles["posy"], particles["posz"]]) # positions
        ind = np.full(particles["posx"].shape, True) # start with all True elements
        for dir in range(3): # find in bb
            ind = np.logical_and(ind, np.logical_and(pos[dir,:] >= bb[dir,0], pos[dir,:] <= bb[dir,1]))
        return ind # return indices of particles in bb

    # return the total sink-gas gravitational interaction potential due to all sinks at a cell position
    def GetSinkGasPot(self, cell_pos):
        if self.n == 0: return None # return if we don't have any particles
        if read_runtime_parameters(self.filename)["grav_boundary_type"] == "periodic":
            print("Obtaining the gas potential with periodic boundary conditions not implemented yet. Returning None.")
            return None
        # Get sink-gas softening type
        soft_type = read_runtime_parameters(self.filename)["sink_softening_type_gas"]
        soft_radius = read_runtime_parameters(self.filename)["sink_softening_radius"]
        # Get min cell size to convert to physical units
        filen = self.filename
        filen = filen.replace("_hdf5_part_","_hdf5_plt_cnt_")
        dx_min = np.min(FlashGG(filen).D)
        # Convert soft_radius to dx_min
        soft_radius *= dx_min
        # Obtain radial distance to position
        sinks_xpos = self.read(prop="posx", type=self.sink_type)
        sinks_ypos = self.read(prop="posy", type=self.sink_type)
        sinks_zpos = self.read(prop="posz", type=self.sink_type)
        sinks_mass = self.read(prop="mass", type=self.sink_type)
        # Get radial distances to each sink
        rdist = (np.ones(len(sinks_xpos)) * cell_pos[0] - sinks_xpos)**2 + \
                (np.ones(len(sinks_ypos)) * cell_pos[1] - sinks_ypos)**2 + \
                (np.ones(len(sinks_zpos)) * cell_pos[2] - sinks_zpos)**2
        # Linear softening
        if soft_type == 'linear':
            def linear_soft(r, r_soft):
                # linear acceleration kernel for r<r_soft: i.e. a = -GMr/r_soft^3
                # this gives phi = -3GM/(2r_soft) + GMr^2/(2r_soft^3) - the second term is to ensure continuity at r_soft
                if r < r_soft:
                    return -3/(2*r_soft) + r**2/(2*r_soft**3)
                # else usual potential for point mass, i.e. phi = -GM/r
                else:
                    return -1/r
            prefactor_grav = cfp.constants.g_n * sinks_mass
            phi_soften = np.vectorize(linear_soft, excluded=['r_soft'])
            return phi_soften(rdist, r_soft=soft_radius)
        else:
            print("Only linear softening implemented yet, spline softening not yet implemented. Returning None.")
            return None

# class to handle time evolution data files
class datfile:

    @cfp.timer_decorator
    def __init__(self, filename, verbose=1, max_num_lines=1e7, read=True, clean=True):
        self.classname = self.__class__.__name__ # class name
        self.filename = filename # data filename
        self.verbose = verbose # for printing to stdout
        self.max_num_lines = int(max_num_lines) # maximum number of lines in data file
        self.header = None # columns header
        self.dat = np.array([]) # 2D data array
        if read: self.read() # read data
        self.sinks_evol = False
        if "sinks_" in self.filename: self.sinks_evol = True # for sinks_evol.dat, sinks_evol_after_outflow.dat, sinks_stellar_evolution.dat
        if clean: self.clean() # clean data
        if self.verbose > 1: print(self.classname+": class instance created.")

    # write cleaned file
    def write_cleaned(self):
        out_filename = self.filename+'_cleaned'
        backup_filename = self.filename+'_sav'
        if self.verbose: print("creating backup copy of '"+self.filename+"' as '"+backup_filename+"'")
        shutil.copyfile(self.filename, backup_filename)
        self.clean()
        self.write(out_filename)

    # remove data lines that were overwritten after FLASH restarts based on time_col
    def clean(self):
        time_col = 0
        if self.sinks_evol: time_col = 1
        completely_cleaned = False
        while not completely_cleaned:
            completely_cleaned = True # first set to True, but can switch to False below
            tmp = np.empty(self.dat.shape) # create temporary work array
            il = 0 # running index for adding cleaned data to tmp
            start_index = len(self.dat) # start index for looping through original data
            done_cleaning = False
            while not done_cleaning:
                for i in reversed(range(start_index)): # loop through all data times in reverse order
                    if i==0: # reached the first data point in time -> finished cleaning
                        done_cleaning = True; break # signal that we are finished cleaning
                    test = self.dat[i-1][time_col] / self.dat[i][time_col] # time ratio of previous line to current line
                    if test < 1:
                        tmp[il] = self.dat[i]; il += 1 # append to output data and increase output counter il
                    if self.sinks_evol and (test == 1):
                        if self.dat[i-1][0] != self.dat[i][0]: # time is the same, but is sink tag different?
                            tmp[il] = self.dat[i]; il += 1 # append to output data and increase output counter il
                    if (test > 1) or ((not self.sinks_evol) and (test == 1)): # keep moving backwards until we hit good data again
                        for j in reversed(range(i-1)):
                            if self.dat[j][time_col] < self.dat[i][time_col]:
                                if self.verbose > 1: print(">>> copied clean data between lines "+str(i+1)+" and "+str(start_index))
                                start_index = j+2; completely_cleaned = False; break
                        if j==0: done_cleaning = True # signal that we are finished cleaning
                        break # break loop over i
            if self.verbose: print("copied clean data between lines "+str(i+1)+" and "+str(start_index))
            tmp[il] = self.dat[i] # copy final missing element
            self.dat = (tmp[:il+1])[::-1] # resize table to correct size and reverse

    # get statistical moments over an interval ([xcol] from xs to xe) of quantity in column ycol
    def get_moments(self, ycol, xcol=0, xs=None, xe=None):
        x = self.dat[:,self.col_ind(xcol)] # extract x column
        y = self.dat[:,self.col_ind(ycol)] # extract y column
        moments = cfp.get_moments(x, y, xs=xs, xe=xe)
        return moments

    # interpolate column ycol in the datafile onto xnew, where x is in xcol (usually time)
    def interpolate(self, ycol, xnew=None, xcol=0):
        tab = self.dat
        x = tab[:,self.col_ind(xcol)] # extract x column
        y = tab[:,self.col_ind(ycol)] # extract y column
        f = scipy.interpolate.interp1d(x, y, kind='cubic')
        if xnew is None: # if xnew is not given, interpolate onto uniform grid with the same number of points as the input
            xnew = np.linspace(np.nanmin(x), np.nanmax(x), len(x))
        ynew = f(xnew)
        return ynew

    # plot
    def plot_column(self, ycol, xcol=0):
        print(self.get_moments(ycol, xcol=xcol))
        #cfp.plot(x=)
        return

    # get column index
    def col_ind(self, col_id):
        if type(col_id) == int: return col_id # if int, return int as index
        if type(col_id) == str: # if string
            if col_id.isnumeric(): return int(col_id) # if string is number, return as int
            else: # it really is a string id for the column
                str_match_list = np.array([x.find(col_id)!=-1 for x in self.header]) # find wildcard match in header list
                if str_match_list.sum() == 1: # found unique column match
                    return np.argwhere(str_match_list).flatten()[0] # return index of matched col_id str
                else: # either no match at all or multiple matching columns by name
                    print("Error: no (unique) match for column id '"+col_id+"'")
                    print("Matched columns include ", self.header[str_match_list])
                    stop()

    # read datafile
    def read(self, read_header=True, to_float=True):
        with open(self.filename, 'r') as f:
            if read_header: self.header = f.readline() # read header (first line)
            self.header = np.array(self.header.split()) # make header numpy array
            self.dat = np.empty((int(self.max_num_lines),len(self.header))) # init output data table
            il = 0 # index to append line to output table
            for line in f: # loop through all lines in file
                try: self.dat[il] = np.asarray(line.split(), dtype=float); il += 1 # fill table with floats
                except: pass # skip bad lines (some element on the line could not be converted to float)
        self.dat = self.dat[:il] # resize table to correct size
        if self.verbose: print("lines in table   : "+str(len(self.dat)))
        if self.verbose: print("columns in table : "+str(len(self.dat[0])))

    # write datafile
    def write(self, out_filename):
        if self.verbose: print("writing '"+out_filename+"'...")
        header = ''.join([' '+x.rjust(23)[:23] for x in self.header])[1:]
        np.savetxt(out_filename, self.dat, header=header, comments='', fmt='%23.16E')
        if self.verbose: print("'"+out_filename+"' written with "+str(len(self.dat))+" lines.", highlight=3)


# ==================== check for file types ===================
def is_movie_slice(filename):
    return filename.find('_slice_') != -1
def is_movie_proj(filename):
    return filename.find('_proj_') != -1
def is_movie_file(filename):
    return is_movie_slice(filename) or is_movie_proj(filename)
def is_part_file(filename):
    searchstr = '_hdf5_part_'
    index = filename.find(searchstr)
    return (index != -1) and (len(filename)-len(searchstr)-index == 4)
def is_plt_file(filename):
    searchstr = '_hdf5_plt_cnt_'
    index = filename.find(searchstr)
    return (index != -1) and (len(filename)-len(searchstr)-index == 4)
def is_chk_file(filename):
    searchstr = '_hdf5_chk_'
    index = filename.find(searchstr)
    return (index != -1) and (len(filename)-len(searchstr)-index == 4)
def is_extracted_file(filename):
    return filename.find('_extracted.h5') != -1
# ================= end: check for file types =================


# ================= read_runtime_parameters ===================
def read_runtime_parameters(flash_file):
    params_dsets = ['integer runtime parameters', \
                    'real runtime parameters', \
                    'logical runtime parameters', \
                    'string runtime parameters']
    runtime_parameters = dict()
    for dset in params_dsets:
        data = hdfio.read(flash_file, dset)
        for i in range(0, len(data)):
            datstr = data[i][0].strip().decode()
            if dset == 'string runtime parameters':
                datval = data[i][1].strip().decode()
            else:
                datval = data[i][1]
            runtime_parameters[datstr] = datval
    return runtime_parameters
# ================ end: read_runtime_parameters ===============


# ======================= read_scalars ========================
def read_scalars(flash_file):
    scalars_dsets = ['integer scalars', \
                     'real scalars', \
                     'logical scalars', \
                     'string scalars']
    scalars = dict()
    for dset in scalars_dsets:
        data = hdfio.read(flash_file, dset)
        for i in range(0, len(data)):
            datstr = data[i][0].strip().decode()
            if dset == 'string scalars':
                datval = data[i][1].strip().decode()
            else:
                datval = data[i][1]
            scalars[datstr] = datval
    return scalars
# ==================== end: read_scalars ======================


# ======================= write_unk_names ========================
# for flash_file, write "unknown names" given unput array unk_names
def write_unk_names(flash_file, unk_names=["dens", "velx", "vely", "velz"], strlen=40, overwrite=True):
    dsetname = "unknown names"
    unn = np.array([unk_names], dtype='|S'+str(strlen)+'') # create numpy string array
    # create datatype and dataspace
    type_id = h5py.h5t.TypeID.copy(h5py.h5t.C_S1)
    type_id.set_size(strlen)
    type_id.set_strpad(h5py.h5t.STR_NULLTERM)
    space = h5py.h5s.create_simple((len(unn[0]),1))
    # open file for read/write
    f = h5py.File(flash_file, "r+")
    if overwrite: del f[dsetname] # delete old "unknown names" dataset
    dset = h5py.h5d.create(f.id, dsetname.encode('utf-8'), type_id, space) # create "unknown names" dataset
    dset.write(space, dset.get_space(), unn) # write dataset
    dset.close() # close dataset
    f.close() # close file
    print("'"+dsetname+"' written in file '"+flash_file+"' with names = ", unk_names, highlight=2)
# ==================== end: write_unk_names ======================


# ======================= read_num_sinks ======================
def read_num_sinks(flash_file):
    num_sinks = 0
    # movie file
    if is_movie_file(flash_file):
        f = h5py.File(flash_file, "r")
        if 'numpart' in f.keys():
            num_sinks = hdfio.read(flash_file, 'numpart')[0]
        f.close()
    # flash particle file
    if is_part_file(flash_file) or is_plt_file(flash_file) or is_chk_file(flash_file):
        particles_obj = Particles(flash_file, verbose=0)
        if particles_obj.sink_type is not None:
            num_sinks = particles_obj.n_by_type[particles_obj.sink_type-1]
    return num_sinks
# ==================== end: read_num_sinks ====================

# ===================== read_sink_masses ======================
def read_sink_masses(flash_file):
    sink_masses = None
    num_sinks = read_num_sinks(flash_file)
    if num_sinks == 0: return
    # movie file
    if is_movie_file(flash_file):
        sink_masses = hdfio.read(flash_file, 'particlemasses')
    # flash particle file
    if is_part_file(flash_file) or is_plt_file(flash_file) or is_chk_file(flash_file):
        particles_obj = Particles(flash_file, verbose=0)
        if particles_obj.sink_type is not None:
            sink_masses = particles_obj.read(prop="mass", type=particles_obj.sink_type)
    return sink_masses
# ================== end: read_sink_masses ====================

# ================= get_min ==================
def get_min(flash_file, dset):
    hdf = h5py.File(flash_file, "r")
    minimum = hdf[dset].attrs["minimum"]
    hdf.close()
    return float(minimum)
# ================= get_max ==================
def get_max(flash_file, dset):
    hdf = h5py.File(flash_file, "r")
    maximum = hdf[dset].attrs["maximum"]
    hdf.close()
    return float(maximum)

# ================= get_time ==================
def get_time(flash_file):
    if is_movie_file(flash_file):
        time = hdfio.read(flash_file, 'time')[0]
    if is_part_file(flash_file) or is_plt_file(flash_file) or is_chk_file(flash_file):
        time = read_scalars(flash_file)['time']
    return time

# ================= get_dump ==================
def get_dump(flash_files, time=None, sink_mass=None, low=None, high=None, quiet=True):
    # sort input list; note that this algorithm only works for sink masses, if sinks are growing in mass for increasing dumps
    flash_files = sorted(flash_files)
    # if not provided, extract low and high files
    if low is None: low = flash_files[0]
    if high is None: high = flash_files[-1]
    # set requested value (time or sink mass) and get low and high value info
    if time is not None:
        req_val = time
        low_val = get_time(low)
        high_val = get_time(high)
    if sink_mass is not None:
        req_val = sink_mass
        low_val = read_sink_masses(low)
        if low_val is not None:
            low_val = sum(low_val)
        else:
            low_val = 0.0
        high_val = read_sink_masses(high)
        if high_val is not None:
            high_val = sum(high_val)
        else:
            high_val = 0.0
    # if there are only 1 or 2 files (left)
    if (len(flash_files) == 1):
        return flash_files[0], low_val
    if (len(flash_files) == 2):
        if abs(low_val-req_val) < abs(high_val-req_val):
            return flash_files[0], low_val
        else:
            return flash_files[1], high_val
    # get middle info
    mid_index = len(flash_files)//2
    mid = flash_files[mid_index]
    if time is not None:
        mid_val = get_time(mid)
    if sink_mass is not None:
        mid_val = read_sink_masses(mid)
        if mid_val is not None:
            mid_val = sum(mid_val)
        else:
            mid_val = 0.0
    if not quiet:
        if time is not None:
            print('--- Searching for time value: '+str(req_val)+' ---')
        if sink_mass is not None:
            print('--- Searching for sink mass value: '+str(req_val)+' ---')
        print('low  (dump, value): '+low+", "+str(low_val))
        print('mid  (dump, value): '+mid+", "+str(mid_val))
        print('high (dump, value): '+high+", "+str(high_val))
    # if element is smaller than mid, then it can only be present in left sub-array
    if req_val <= mid_val:
        flash_files = flash_files[:mid_index+1]
        if time is not None:
            return get_dump(flash_files, time=req_val, sink_mass=None, low=flash_files[0], high=flash_files[-1])
        if sink_mass is not None:
            return get_dump(flash_files, time=None, sink_mass=req_val, low=flash_files[0], high=flash_files[-1])
    # else the element can only be present in right sub-array
    else:
        flash_files = flash_files[mid_index:]
        if time is not None:
            return get_dump(flash_files, time=req_val, sink_mass=None, low=flash_files[0], high=flash_files[-1])
        if sink_mass is not None:
            return get_dump(flash_files, time=None, sink_mass=req_val, low=flash_files[0], high=flash_files[-1])


# ======================= get_sim_info =======================
def get_sim_info(filename):
    if not (is_plt_file(filename) or is_chk_file(filename)):
        print('Needs either plt or chk file; returning...')
        return
    # basic information
    print("--- Basic information ---", color="green")
    scalars = read_scalars(filename)
    rtparams = read_runtime_parameters(filename)
    print('time = ', cfp.eform(scalars['time']))
    redshift = scalars['redshift']
    a = 1 / (1 + redshift)
    if redshift > 0: print('redshift = ', cfp.eform(redshift))
    print('dt = ', cfp.eform(scalars['dt']))
    print('cfl = ', rtparams['cfl'])
    if 'interpol_order' in rtparams: print('interpol_order = ', rtparams['interpol_order'])
    # grid information
    gg = FlashGG(filename)
    print("--- Grid information ---", color="green")
    print("dimensionality of grid =", gg.Ndim)
    print("grid type =", gg.GridType)
    gg.GetRefinementInfo()
    print("number of blocks =", gg.NumBlocks)
    print("number of cells per block =", gg.NB)
    print("base grid resolution =", gg.NBaseGrid)
    print("maximum effective resolution =", gg.NMax)
    print('min/max cell size =', gg.D.min(), gg.D.max())
    if redshift > 0:
        print('min/max cell size (proper) =', gg.D.min()*a, gg.D.max()*a)
    print('domain bounds =')
    print(gg.domain_bounds.tolist())
    if redshift > 0:
        print('domain bounds (proper) =')
        print((gg.domain_bounds*a).tolist())
    # particle information
    part = Particles(filename)
    if part.n > 0:
        print("--- Particle information ---", color="green")
        part.print_info()
        sink_masses = read_sink_masses(filename)
        if sink_masses is not None:
            sink_masses = np.sort(sink_masses)
            print("Sink particle masses: total, min, max, median, mean, std =",
                cfp.round([np.sum(sink_masses), sink_masses.min(), sink_masses.max(), np.median(sink_masses), sink_masses.mean(), sink_masses.std()]))
# ==================== end: get_sim_info  =====================

# =============================================================
def main():
    exit()


# ===== the following applies in case we are running this in script mode =====
if __name__ == "__main__":

    # create parser
    parser = argparse.ArgumentParser(description='FLASH library.')
    subparsers = parser.add_subparsers(title='subcommands', dest='subcommand', description='valid subcommands', help='additional help', required=True)
    # sub parser for 'dumpfind' sub-command
    parser_dumpfind = subparsers.add_parser('dumpfind')
    parser_dumpfind.add_argument("inputfiles", nargs='+', type=argparse.FileType('r'), help="Input data file(s) to process")
    parser_dumpfind.add_argument("-t", "--time", dest='requested_time', type=float, help="Return filename based on requested time")
    parser_dumpfind.add_argument("-s", "--sink_mass", dest='requested_sink_mass', type=float, help="Return filename based on requested total sink mass")
    # sub parser for 'datfile' sub-command
    parser_datfile = subparsers.add_parser('datfile')
    parser_datfile.add_argument("inputfile", type=argparse.FileType('r'), help="Time evolution file to clean")
    parser_datfile.add_argument("-clean", action='store_true', default=False, help="Write a cleaned datfile (with old data after later restarts removed)")
    parser_datfile.add_argument("-moments_col", help="Get statistical moments of input column")
    parser_datfile.add_argument("-time_range", nargs=2, type=float, default=[-1e99, +1e99], help="limit to time range")
    parser_datfile.add_argument("-plot_col", help="Plot time evolution of input column")
    # sub parser for 'siminfo' sub-command
    parser_siminfo = subparsers.add_parser('siminfo')
    parser_siminfo.add_argument("inputfiles", nargs='+', type=argparse.FileType('r'), help="Input data file(s) to process")
    # now parse arguments
    args = parser.parse_args()

    # ===================== handle dumpfind case =====================
    if args.subcommand == 'dumpfind':
        # searching for time and sink mass at the same time is not allowed
        if (args.requested_time is not None) and (args.requested_sink_mass is not None):
            print('Error: must not request both time and sink_mass in one call. Exiting...')
            exit()
        # sort input files
        inputfiles = sorted([x.name for x in list(args.inputfiles)])
        # find dump file
        if debug: print('Searching through files: ', inputfiles)
        filename_found, value_found = get_dump(inputfiles, time=args.requested_time, sink_mass=args.requested_sink_mass)
        if filename_found is not None:
            if args.requested_time is not None:
                print("Found requested time "+str(args.requested_time)+" in file '"+filename_found+"' (exact time in file: "+str(value_found)+")")
            if args.requested_sink_mass is not None:
                print("Found requested total sink mass "+str(args.requested_sink_mass)+" in file '"+filename_found+"' (exact total sink mass in file: "+str(value_found)+")")
            # print number of sinks and sink masses
            print(filename_found+': number of sink particles = ', read_num_sinks(filename_found), '; mass of sink particles = ', read_sink_masses(filename_found))
        else:
            print('No matching file found for request. Exiting...')

    # ===================== handle datfile case =====================
    if args.subcommand == 'datfile':
        # create new datfile class instance
        datfile_obj = datfile(args.inputfile.name)
        # clean if requested
        if args.clean: datfile_obj.write_cleaned()
        # get moments for column
        if args.moments_col:
            moments = datfile_obj.get_moments(args.moments_col, xs=args.time_range[0], xe=args.time_range[1])
            print("statistical moments for column '"+args.moments_col+"': ", moments)
        # plot column
        if args.plot_col: datfile_obj.plot_column(args.plot_column)

    if args.subcommand == 'siminfo':
        # sort input files
        inputfiles = sorted([x.name for x in list(args.inputfiles)])
        for filename in inputfiles:
            get_sim_info(filename)

# =============================================================
