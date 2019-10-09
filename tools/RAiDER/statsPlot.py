#!/usr/bin/env python3
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Simran Sangha, Jeremy Maurer, & David Bekaert
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from shapely.geometry import Polygon, Point
import numpy as np
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt

def createParser():
    '''
        Make any of the following specified plot(s): scatterplot of station locations, total empirical and experimental variogram fits for aggregate residuals between GNSS zenith delay and ERA-5 estimated zenith delays in each each grid cell (and for each valid time-slice if -verbose specified),and gridded heatmaps of station distribution, aggregate residuals between GNSS zenith delay and ERA-5 estimated zenith delays, range and sill values associated with experimental variogram fits. The default is to generate all of these.
    '''
    import argparse
    parser = argparse.ArgumentParser(description='Function to generate various quality control and baseline figures of the spatial-temporal network of products.')
    parser.add_argument('-f', '--file', dest='fname', type=str,required=True, help='csv file')
    parser.add_argument('-c', '--column_name', dest='col_name', type=str,default='GNSS_minus_ERA5', help='Name of the input column to plot. Input assumed to be in units of meters')
    parser.add_argument('-fmt', '--plot_format', dest='plot_fmt', type=str,default='png', help='Plot format to use for saving figures')
    parser.add_argument('-cb', '--color_bounds', dest='cbounds', type=float,nargs=2,default=None, help='List of two floats to use as color axis bounds')
    parser.add_argument('-w', '--workdir', dest='workdir', default='./', help='Specify directory to deposit all outputs. Default is local directory where script is launched.')
    parser.add_argument('-b', '--bbox', dest='bbox', type=str, default=None, help="Provide either valid shapefile or Lat/Lon Bounding SNWE. -- Example : '19 20 -99.5 -98.5'")
    parser.add_argument('-sp', '--spacing', dest='spacing', type=float, default='1', help='Specify spacing of grid-cells for statistical analyses. By default 1 deg.')
    parser.add_argument('-dt', '--densitythreshold', dest='densitythreshold', type=int, default='10', help='A given grid-cell is only valid if it contains this specified threshold of stations. By default 10 stations.')
    parser.add_argument('-sg', '--stationsongrids', dest='stationsongrids', action='store_true', help='In gridded plots, superimpose your gridded array with a scatterplot of station locations.')
    parser.add_argument('-dg', '--drawgridlines', dest='drawgridlines', action='store_true', help='Draw gridlines on gridded plots.')
    parser.add_argument('-cp', '--colorpercentile', dest='colorpercentile', type=float, default=None,nargs=2, help='Set low and upper percentile for plot colorbars. By default 25%% and 95%%, respectively.')
    parser.add_argument('-ti', '--timeinterval', dest='timeinterval', type=str, default=None, help="Subset in time by specifying earliest YYYY-MM-DD date followed by latest date YYYY-MM-DD. -- Example : '2016-01-01 2019-01-01'.")
    parser.add_argument('-si', '--seasonalinterval', dest='seasonalinterval', type=str, default=None, help="Subset in by an specific interval for each year by specifying earliest MM-DD time followed by latest MM-DD time. -- Example : '03-21 06-21'.")
    parser.add_argument('-station_distribution', '--station_distribution', action='store_true', dest='station_distribution', help="Plot station distribution.")
    parser.add_argument('-station_delay_mean', '--station_delay_mean', action='store_true', dest='station_delay_mean', help="Plot station mean delay.")
    parser.add_argument('-station_delay_stdev', '--station_delay_stdev', action='store_true', dest='station_delay_stdev', help="Plot station delay stdev.")
    parser.add_argument('-grid_heatmap', '--grid_heatmap', action='store_true', dest='grid_heatmap', help="Plot gridded station heatmap.")
    parser.add_argument('-grid_delay_mean', '--grid_delay_mean', action='store_true', dest='grid_delay_mean', help="Plot gridded station mean delay.")
    parser.add_argument('-grid_delay_stdev', '--grid_delay_stdev', action='store_true', dest='grid_delay_stdev', help="Plot gridded station delay stdev.")
    parser.add_argument('-variogramplot', '--variogramplot', action='store_true', dest='variogramplot', help="Plot gridded station variogram.")
    parser.add_argument('-binnedvariogram', '--binnedvariogram', action='store_true', dest='binnedvariogram', help="Apply experimental variogram fit to total binned empirical variograms for each time slice. Default is to total unbinned empiricial variogram.")
    parser.add_argument('-plotall', '--plotall', action='store_true', dest='plotall', help="Generate all above plots.")
    parser.add_argument('-verbose', '--verbose', action='store_true', dest='verbose', help="Toggle verbose mode on. Must be specified to generate variogram plots per gridded station AND time-slice.")
    '''
        Example call to plot gridded station mean delay in a specfic time interval : raiderStats.py -f <filename> -grid_delay_mean -ti '2017-01-01 2018-01-01'
    '''
    return parser

def cmdLineParse(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)

class variogramAnalysis():
    '''
        Class which ingests dataframe output from 'raiderStats' class and performs variogram analysis.
    '''
    def __init__(self, filearg, gridpoints, col_name, workdir='./', seasonalinterval=None, densitythreshold=10, verbose=False, binnedvariogram=False):
        self.df = filearg
        self.col_name = col_name
        self.gridpoints = gridpoints
        self.col_name = col_name
        self.workdir = workdir
        self.seasonalinterval = seasonalinterval
        self.densitythreshold = densitythreshold
        self.verbose = verbose
        self.binnedvariogram = binnedvariogram

    def _getSamples(self, data, Nsamp=None): 
        '''
        pull samples from a 2D image for variogram analysis 
        '''
        import random
        import itertools
        if len(data)<self.densitythreshold:
            print('WARNING: Less than {} points for this gridcell'.format(self.densitythreshold))
            print('Will pass empty list')
            d=[]
            indpars=[]
        else:
            indpars = list(itertools.combinations(range(len(data)), 2))
            random.shuffle(indpars)
            #subsample
            Nsamp = int((len(data)*len(data))/2)
            #Only downsample if Nsamps>1000
            if Nsamp>1000:
                indpars=indpars[:Nsamp]
            d = np.array([[data[r[0]],data[r[1]]] for r in indpars])
            # oversample and remove NaNs if possible
            mask = ~np.isnan(d) 
            if False in mask:
                print('Warning: NaNs present')
                d = d[mask] 
                indpars = indpars[mask]

        return d, indpars

    def _getXY(self, x2d, y2d, indpars):
        '''
        Given a list of indices, return the x,y locations 
        from two matrices
        '''
        x = np.array([[x2d[r[0]],x2d[r[1]]] for r in indpars])
        y = np.array([[y2d[r[0]],y2d[r[1]]] for r in indpars])

        return x, y

    def _getDistances(self, XY, xy = None):
        '''
        Return the distances between each point in a list of points
        '''
        if xy is None:
            from scipy.spatial.distance import cdist
            return np.diag(cdist(XY[:,:,0], XY[:,:,1], metric = 'euclidean'))
        else:
            return 0.5*np.square(XY - xy)  #XY = 1st col xy= 2nd col

    def _empVario(self, x, y, data, Nsamp = 1e3):
        '''
        Compute empirical semivariance
        '''
        #deramp
        A=np.array([x,y,np.ones(len(x))]).T
        ramp = np.linalg.lstsq(A, data.T, rcond=None)[0]
        data=data-(np.matmul(A,ramp))

        samples, indpars = self._getSamples(data)
        x, y = self._getXY(x, y, indpars)
        dists = self._getDistances(np.array([[x[:,0], y[:,0]],[x[:,1], y[:,1]]]).T)
        vario = self._getDistances(samples[:,0],samples[:,1])

        return dists, vario

    def _binnedVario(self, hEff, rawVario, xBin = None):
        '''
        return a binned empirical variogram
        '''
        if xBin is None:
           xBin= np.linspace(0, np.nanmax(hEff)*.67, 20)

        nBins=len(xBin)-1;
        hExp, expVario = [], []
        
        for iBin in range(nBins):
           iBinMask = np.logical_and(xBin[iBin]<hEff, hEff<=xBin[iBin+1])
             
           try:
              hExp.append(np.nanmean(hEff[iBinMask]))
              expVario.append(np.nanmean(rawVario[iBinMask]))
           except:
              pass

        if False in ~np.isnan(hExp):
            #print('Warning: NaNs present in binned histogram')
            hExp = [x for x in hExp if str(x) != 'nan']
            expVario = [x for x in expVario if str(x) != 'nan']

        return np.array(hExp), np.array(expVario)

    def _fitVario(self, dists, vario, model=None, x0 = None, Nparm = None,  ub = None):
        '''
        Fit a variogram model to data
        '''
        from scipy.optimize import least_squares

        def resid(x, d, v, m):
            return (m(x, d) - v)

        if ub is None:
           ub = np.array([np.nanmax(dists)*0.8, np.nanmax(vario)*0.8, np.nanmax(vario)*0.8])

        if x0 is None and Nparm is None:
           raise RuntimeError('Must specify either x0 or the number of model parameters')
        if x0 is not None:
           lb = np.zeros(len(x0))
        if Nparm is not None:
           lb = np.zeros(Nparm)
           x0 = (ub-lb)/2
     
        bounds = (lb, ub)

        mask = np.isnan(dists) | np.isnan(vario)
        d = dists[~mask].copy()
        v = vario[~mask].copy()

        res_robust = least_squares(resid, x0, bounds = bounds,
                       loss='soft_l1', f_scale = 0.1, 
                       args = (d, v, model))

        d_test = np.linspace(0, np.nanmax(dists), 100)
        v_test = model(res_robust.x, d_test) # v_test is my y., # res_robust.x =a, b, c, where a = range, b = sill, and c = nugget model, d_test=x

        return res_robust, d_test, v_test

    #this would be expontential plus nugget
    def __exponential__(self, parms, h):
        '''
        returns a variogram model given a set of arguments and 
        key-word arguments
        '''
        # a = range, b = sill, c = nugget model
        a, b, c = parms
        return b*(1 - np.exp(-h/a)) + c

    #this would be gaussian plus nugget
    def __gaussian__(self, parms, h):
        '''
        returns a Gaussian variogram model
        '''
        a, b, c = parms
        return b*(1 - np.exp(-np.square(h)/(a**2))) + c

    def plotVariogram(self, gridID, timeslice, coords, workdir='./', d_test=None, v_test=None, res_robust=None, dists=None, vario=None, dists_binned=None, vario_binned=None, seasonalinterval=None):
        """ Make empirical and/or experimental variogram fit plots """

        # If specified workdir doesn't exist, create it
        if not os.path.exists(workdir):
            os.mkdir(workdir)

        # make plot title
        title_str=' \nLat:{:.2f} Lon:{:.2f}\nTime:{}'.format(coords[1],coords[0],str(timeslice))
        if seasonalinterval:
            title_str+=' Season(mm/dd): {}/{} – {}/{}'.format(int(timeslice[4:6]),int(timeslice[6:8]), int(timeslice[-4:-2]), int(timeslice[-2:]))

        if dists is not None and vario is not None:
            plt.scatter(dists, vario, s=1, facecolor='0.5', label='raw')
        if dists_binned is not None and vario_binned is not None:
            plt.plot(dists_binned, vario_binned, 'bo', label='binned')
        if res_robust is not None:
            plt.axhline(y=res_robust[1], color='g', linestyle='--', label='ɣ\u0332\u00b2(m\u00b2)')
            plt.axvline(x=res_robust[0], color='c', linestyle='--', label='h\u0332(°)')
        if d_test is not None and v_test is not None:
            plt.plot(d_test, v_test, 'r-', label='experimental fit')
        plt.xlabel('Distance (°)')
        plt.ylabel('Dissimilarity (m\u00b2)')
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        #Plot empirical variogram
        if d_test is None and v_test is None:
            plt.title('Empirical variogram'+title_str)
            plt.tight_layout()
            plt.savefig(os.path.join(workdir,'grid{}_timeslice{}_justEMPvariogram.eps'.format(gridID,timeslice)))
        #Plot just experimental variogram
        else:
            plt.title('Experimental variogram'+title_str)
            plt.tight_layout()
            plt.savefig(os.path.join(workdir,'grid{}_timeslice{}_justEXPvariogram.eps'.format(gridID,timeslice)))
        plt.close()

        return

    def createVariograms(self):
        #Iterate through stations and time slice to append empirical variogram
        #track data for plotting
        TOT_good_slices=[]
        TOT_res_robust_arr=[]
        TOT_tot_timetag=[]
        #track pass/rejected grids
        sparse_grids=[]
        good_slices=[]
        skipped_slices=[]
        # record grid-centers for lookup-table
        gridcenterlist=[]
        for i in sorted(list(set(self.df['gridnode']))):
            dists_arr=[]; vario_arr=[]
            dists_binned_arr=[]; vario_binned_arr=[]
            res_robust_arr=[]; d_test_arr=[]; v_test_arr=[]
            grid_subset=self.df[self.df['gridnode']==i]
            for j in sorted(list(set(grid_subset['Date']))):
                #If insufficient sample size, skip slice and record occurence
                if len(np.array(grid_subset[grid_subset['Date']==j][self.col_name]))<self.densitythreshold:
                    #Record skipped [gridnode, timeslice]
                    skipped_slices.append([i, j.strftime("%Y-%m-%d")])
                else:
                    gridcenterlist.append(['grid{} '.format(i)+'Lat:{} Lon:{}'.format(str(self.gridpoints[i][1]),str(self.gridpoints[i][0]))])
                    lonarr=np.array(grid_subset[grid_subset['Date']==j]['Lon'])
                    latarr=np.array(grid_subset[grid_subset['Date']==j]['Lat'])
                    delayarray=np.array(grid_subset[grid_subset['Date']==j][self.col_name])
                    #fit empirical variogram for each time AND grid
                    dists, vario=self._empVario(lonarr,latarr,delayarray)
                    dists_binned, vario_binned=self._binnedVario(dists, vario)
                    #fit experimental variogram for each time AND grid, model default is exponential
                    res_robust, d_test, v_test = self._fitVario(dists_binned, vario_binned, model=self.__exponential__, x0 = None, Nparm = 3)
                    #Plot empirical + experimental variogram for this gridnode and timeslice
                    if not os.path.exists(os.path.join(self.workdir,'variograms/grid{}'.format(i))):
                        os.makedirs(os.path.join(self.workdir,'variograms/grid{}'.format(i)))
                    #Make variogram plots for each time-slice if verbose mode specified.
                    if self.verbose:
                        #Plot empirical variogram for this gridnode and timeslice
                        self.plotVariogram(i, j.strftime("%Y%m%d"), [self.gridpoints[i][1],self.gridpoints[i][0]], workdir=os.path.join(self.workdir,'variograms/grid{}'.format(i)), dists=dists, vario=vario, dists_binned=dists_binned, vario_binned=vario_binned) #in verbose
                        #Plot experimental variogram for this gridnode and timeslice
                        self.plotVariogram(i, j.strftime("%Y%m%d"), [self.gridpoints[i][1],self.gridpoints[i][0]], workdir=os.path.join(self.workdir,'variograms/grid{}'.format(i)), d_test=d_test, v_test=v_test, res_robust=res_robust.x, dists_binned=dists_binned, vario_binned=vario_binned) #in verbose
                    #append for plotting
                    good_slices.append([i, j.strftime("%Y%m%d")])
                    dists_arr.append(dists); vario_arr.append(vario)
                    dists_binned_arr.append(dists_binned); vario_binned_arr.append(vario_binned)
                    res_robust_arr.append(res_robust.x); d_test_arr.append(d_test); v_test_arr.append(v_test)
            #fit experimental variogram for each grid
            if dists_binned_arr!=[]:
                #TODO: need to change this from accumulating binned data to raw data
                dists_arr=np.concatenate(dists_arr).ravel(); vario_arr=np.concatenate(vario_arr).ravel()
                #if specified, passed binned empirical variograms
                if self.binnedvariogram:
                    dists_binned_arr=np.concatenate(dists_binned_arr).ravel(); vario_binned_arr=np.concatenate(vario_binned_arr).ravel()
                else:
                    dists_binned_arr, vario_binned_arr=self._binnedVario(dists_arr, vario_arr)
                TOT_res_robust, TOT_d_test, TOT_v_test = self._fitVario(dists_binned_arr, vario_binned_arr, model=self.__exponential__, x0 = None, Nparm = 3)
                #Plot empirical variogram for this gridnode and timeslice
                tot_timetag=good_slices[0][1]+'–'+good_slices[-1][1]
                #Append TOT arrays
                TOT_good_slices.append([i, tot_timetag])
                TOT_res_robust_arr.append(TOT_res_robust.x)
                TOT_tot_timetag.append(tot_timetag)
                self.plotVariogram(i, tot_timetag, [self.gridpoints[i][1],self.gridpoints[i][0]], workdir=os.path.join(self.workdir,'variograms/grid{}'.format(i)), dists=dists_arr, vario=vario_arr, dists_binned=dists_binned_arr, vario_binned=vario_binned_arr, seasonalinterval=self.seasonalinterval)
                #Plot experimental variogram for this gridnode and timeslice
                self.plotVariogram(i, tot_timetag, [self.gridpoints[i][1],self.gridpoints[i][0]], workdir=os.path.join(self.workdir,'variograms/grid{}'.format(i)), d_test=TOT_d_test, v_test=TOT_v_test, res_robust=TOT_res_robust.x, seasonalinterval=self.seasonalinterval)
            # Record sparse grids which didn't have sufficient sample size of data through any of the timeslices
            else:
                sparse_grids.append(i)

        # save grid-center lookup table
        gridcenterlist=[list(i) for i in set(tuple(j) for j in gridcenterlist)]
        gridcenter= open((os.path.join(self.workdir,'variograms/gridlocation_lookup.txt')),"w")
        for element in gridcenterlist:
            gridcenter.writelines("\n".join(element))
            gridcenter.write("\n")
        gridcenter.close()

        TOT_grids=[i[0] for i in TOT_good_slices]

        return TOT_grids,TOT_res_robust_arr


class raiderStats(object):
    '''
        Class which loads standard weather model/GPS delay files and generates a series of user-requested statistics and graphics.
    '''

    # import dependencies
    import glob

    def __init__(self, filearg,col_name, workdir='./', bbox=None, spacing=1, timeinterval=None, seasonalinterval=None, stationsongrids=False, colorpercentile='25 95', verbose=False):
        self.fname = filearg
        self.col_name=col_name 
        self.workdir = workdir
        self.bbox = bbox
        self.spacing = spacing
        self.timeinterval = timeinterval
        self.seasonalinterval = seasonalinterval
        self.stationsongrids = stationsongrids
        self.colorpercentile = colorpercentile
        self.verbose = verbose

        # create workdir if it doesn't exist
        if not os.path.exists(self.workdir):
            os.mkdir(self.workdir)

        self.createDF()

    def _getExtent(self):#dataset, spacing=1, userbbox=None
        """ Get the bbox, spacing in deg (by default 1deg), optionally pass user-specified bbox. Output array in WESN degrees """
        extent = [np.floor(min(self.df['Lon'])-(self.spacing/2)),np.ceil(max(self.df['Lon'])+(self.spacing/2)),
                    np.floor(min(self.df['Lat'])-(self.spacing/2)),np.ceil(max(self.df['Lat'])+(self.spacing/2))]
        if self.bbox is not None:
            dfextents_poly=Polygon(np.column_stack((np.array([extent[0],extent[0],extent[1],extent[1],extent[0]]),
                np.array([extent[2],extent[3],extent[3],extent[2],extent[2]]))))
            userbbox_poly=Polygon(np.column_stack((np.array([self.bbox[2],self.bbox[3],self.bbox[3],self.bbox[2],self.bbox[2]]),
                np.array([self.bbox[0],self.bbox[0],self.bbox[1],self.bbox[1],self.bbox[0]]))))
            if userbbox_poly.intersects(dfextents_poly):
                extent = [np.floor(self.bbox[2]-(self.spacing/2)),np.ceil(self.bbox[-1]+(self.spacing/2)),
                    np.floor(self.bbox[0]-(self.spacing/2)),np.ceil(self.bbox[1]+(self.spacing/2))]
            else:
                print("WARNING: User-specified bounds do not overlap with dataset bounds, proceeding with the latter.")
            del dfextents_poly, userbbox_poly

        # Create corners of rectangle to be transformed to a grid
        nw = [extent[0]+(self.spacing/2),extent[-1]-(self.spacing/2)]
        se = [extent[1]-(self.spacing/2),extent[2]+(self.spacing/2)]

        #Store grid dimension [y,x]
        grid_dim=[int((extent[1]-extent[0])/self.spacing), int((extent[-1]-extent[-2])/self.spacing)]

        # Iterate over 2D area
        gridpoints = []
        y_shape=[]
        x_shape=[]
        x = se[0]
        while x >= nw[0]:
            y = se[1]
            while y <= nw[1]:
                y_shape.append(y)
                gridpoints.append([x,y])
                y += self.spacing
            x_shape.append(x)
            x -= self.spacing
        gridpoints.reverse()

        return extent, grid_dim, gridpoints

    def _reader(self):
        '''
        Read a input file
        '''
        data = pd.read_csv(self.fname)
        return data

    def createDF(self):
        '''
            Create dataframe.
        '''
        # Open file
        self.df = self._reader()

        ###Filter dataframe
        #drop all nans
        self.df.dropna(how='any',inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        #convert to datetime object
        self.df['Date']=pd.to_datetime(self.df['Date'])

        #time-interval filter
        if self.timeinterval:
            self.timeinterval = [dt.datetime.strptime(val, '%Y-%m-%d') for val in self.timeinterval.split()]
            self.df=self.df[(self.df['Date']>=self.timeinterval[0]) & (self.df['Date']<=self.timeinterval[-1])]

        #seasonal filter
        if self.seasonalinterval:
            #get day of year
            self.seasonalinterval = [str(val) for val in self.seasonalinterval.split()]
            self.seasonalinterval=[dt.datetime.strptime('2001-'+self.seasonalinterval[0], '%Y-%m-%d').timetuple().tm_yday, dt.datetime.strptime('2001-'+self.seasonalinterval[-1], '%Y-%m-%d').timetuple().tm_yday]
            #non leap-year
            filtered_self=self.df[(self.df['Date'].dt.is_leap_year==False) & (self.df['Date'].dt.dayofyear>=self.seasonalinterval[0]) & (self.df['Date'].dt.dayofyear<=self.seasonalinterval[-1])]
            #leap-year
            self.seasonalinterval=[i+1 if i>59 else i for i in self.seasonalinterval]
            self.df=filtered_self.append(self.df[(self.df['Date'].dt.is_leap_year==True) & (self.df['Date'].dt.dayofyear>=self.seasonalinterval[0]) & (self.df['Date'].dt.dayofyear<=self.seasonalinterval[-1])], ignore_index=True)
            del filtered_self

        ###Get bbox, buffered by grid spacing.
        # Check if bbox input is valid list.
        if self.bbox is not None:
            try:
                self.bbox = [float(val) for val in self.bbox.split()]
            except:
                raise Exception('Cannot understand the --bbox argument. String input is incorrect or path does not exist.')
        self.plotbbox, self.grid_dim, self.gridpoints= self._getExtent()

        # generate list of grid-polygons
        append_poly=[]
        for i in self.gridpoints:
            bbox=[i[1]-(self.spacing/2), i[1]+(self.spacing/2), i[0]-(self.spacing/2), i[0]+(self.spacing/2)]
            append_poly.append(Polygon(np.column_stack((np.array([bbox[2],bbox[3],bbox[3],bbox[2],bbox[2]]),
                    np.array([bbox[0],bbox[0],bbox[1],bbox[1],bbox[0]]))))) #Pass lons/lats to create polygon

        #check for grid cell intersection with each station (loop through each station).. fast, but assumes station locations don't change
        idtogrid_dict={}
        unique_points=self.df.groupby(['ID', 'Lon', 'Lat']).size()
        unique_points=[unique_points.index.get_level_values('ID').tolist(),unique_points.index.get_level_values('Lon').tolist(),unique_points.index.get_level_values('Lat').tolist()]
        for i in unique_points[0]:
            try:
                coord = Point((unique_points[1][unique_points[0].index(i)], unique_points[2][unique_points[0].index(i)]))
                idtogrid_dict[i]=[j.intersects(coord) for j in append_poly].index(True)
            except:
                idtogrid_dict[i]='NaN'
        #map gridnode dictionary to dataframe
        self.df['gridnode'] = self.df['ID'].map(idtogrid_dict)
        del unique_points, idtogrid_dict, append_poly
        #sort by grid and date
        self.df.sort_values(['gridnode','Date'])


        ###If specified, pass station locations to superimpose on gridplots
        if self.stationsongrids:
            unique_points=self.df.groupby(['Lon', 'Lat']).size()
            self.stationsongrids=[unique_points.index.get_level_values('Lon').tolist(),unique_points.index.get_level_values('Lat').tolist()]

        ###Pass color percentile and check for input error
        if self.colorpercentile is None:
           self.colorpercentile=[25, 95]
        if self.colorpercentile[0]>self.colorpercentile[1]:
            raise Exception('Input colorpercentile lower threshold {} higher than upper threshold {}'.format(self.colorpercentile[0],self.colorpercentile[1]))

    def __call__(self, gridarr, plottype, workdir='./', drawgridlines=False, colorbarfmt='%.3f', stationsongrids=None, cbounds = None, resValue = 5, plotFormat = 'pdf'):
        '''
            Visualize a suite of statistics w.r.t. stations. Pass either a list of points or a gridded array as the first argument. Alternatively, you may superimpose your gridded array with a supplementary list of points by passing the latter through the stationsongrids argument.
        '''

        import matplotlib as mpl
        # supress matplotlib postscript warnings
        mpl._log.setLevel('ERROR')
        from pandas.plotting import register_matplotlib_converters
        register_matplotlib_converters()
        import cartopy.io.img_tiles as cimgt
        import cartopy.crs as ccrs
        from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
        import matplotlib.ticker as mticker

        # If specified workdir doesn't exist, create it
        if not os.path.exists(workdir):
            os.mkdir(workdir)

        basemap   = cimgt.Stamen('terrain-background')
        #convert basemap to black-and-white mode
        basemap.desired_tile_form='L'
        fig, axes = plt.subplots(subplot_kw={'projection':basemap.crs})
        axes.set_extent(self.plotbbox, ccrs.Geodetic())
        axes.add_image(basemap, resValue, cmap='gray')
        axes.coastlines()
        cmap = plt.cm.hot_r
        cmap.set_bad('black', 0.)
        # extract all colors from the hot map
        cmaplist = [cmap(i) for i in range(cmap.N)]
        # create the new map
        cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
        axes.set_xlabel('longitude',weight='bold', zorder=2)
        axes.set_ylabel('latitude',weight='bold', zorder=2)

        # set ticks
        axes.set_xticks(np.linspace(self.plotbbox[0], self.plotbbox[1], 5), crs=ccrs.PlateCarree())
        axes.set_yticks(np.linspace(self.plotbbox[2], self.plotbbox[3], 5), crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter(number_format='.0f', degree_symbol='')
        lat_formatter = LatitudeFormatter(number_format='.0f', degree_symbol='')
        axes.xaxis.set_major_formatter(lon_formatter)
        axes.yaxis.set_major_formatter(lat_formatter)

        #If individual stations passed
        if isinstance(gridarr, list):
            #spatial distribution of stations
            if plottype=="station_distribution":
                axes.set_title(" ".join(plottype.split('_')), zorder=2)
                im   = axes.scatter(gridarr[0], gridarr[1], zorder=1, s=1, facecolor='0', transform=ccrs.PlateCarree())

            #passing 3rd column as z-value
            if len(gridarr)>2:
                zvalues=gridarr[2]
                # define the bins and normalize
                if cbounds is None:
                    cbounds = [np.nanpercentile(zvalues,self.colorpercentile[0]), np.nanpercentile(zvalues,self.colorpercentile[1])]
                colorbounds = np.linspace(cbounds[0], cbounds[1], 11)

                norm = mpl.colors.BoundaryNorm(colorbounds, cmap.N)
                zvalues=np.ma.masked_where(zvalues == 0, zvalues)

                #plot data and initiate colorbar
                im   = axes.scatter(gridarr[0], gridarr[1], c=zvalues, cmap=cmap, norm=norm, vmin=cbounds[0], vmax=cbounds[1], zorder=1, s=1, transform=ccrs.PlateCarree())
                # initiate colorbar
                cbar_ax=fig.colorbar(im, cmap=cmap, norm=norm, spacing='proportional', ticks=colorbounds, boundaries=colorbounds, format=colorbarfmt, pad=0.1)
                cbar_ax.set_label(" ".join(plottype.split('_')), rotation=-90, labelpad=10)

        #If gridded area passed
        else:
            # define the bins and normalize
            if cbounds is None:
                cbounds = [np.nanpercentile(gridarr,self.colorpercentile[0]), np.nanpercentile(gridarr,self.colorpercentile[1])]
            colorbounds = np.linspace(cbounds[0], cbounds[1], 11)
            norm = mpl.colors.BoundaryNorm(colorbounds, cmap.N)
            gridarr=np.ma.masked_where(gridarr == np.nan, gridarr)

            #plot data
            im   = axes.imshow(gridarr, cmap=cmap, norm=norm, extent=self.plotbbox, vmin=cbounds[0], vmax=cbounds[1], zorder=1, origin = 'upper', transform=ccrs.PlateCarree())
            # initiate colorbar
            cbar_ax=fig.colorbar(im, cmap=cmap, norm=norm, spacing='proportional', ticks=colorbounds, boundaries=colorbounds, format=colorbarfmt, pad=0.1)

            #superimpose your gridded array with a supplementary list of point, if specified
            if self.stationsongrids:
                axes.scatter(self.stationsongrids[0], self.stationsongrids[1], zorder=2, s=1, color='b', facecolor='0', transform=ccrs.PlateCarree())

            #draw gridlines, if specified
            if drawgridlines:
                gl=axes.gridlines(crs=ccrs.PlateCarree(),linewidth=2, color='black', alpha=0.5, linestyle='-', zorder=3)
                gl.xlocator=mticker.FixedLocator(np.arange(self.plotbbox[0], self.plotbbox[1]+self.spacing, self.spacing).tolist())
                gl.ylocator=mticker.FixedLocator(np.arange(self.plotbbox[2], self.plotbbox[3]+self.spacing, self.spacing).tolist())

            #experimental variogram fit range heatmap
            if plottype=="range_heatmap":
                cbar_ax.set_label(" ".join(plottype.split('_'))+' (°)', rotation=-90, labelpad=10)

            #experimental variogram fit sill heatmap
            elif plottype=="sill_heatmap":
                cbar_ax.set_label(" ".join(plottype.split('_'))+' (cm\u00b2)', rotation=-90, labelpad=10)

            else:
                cbar_ax.set_label(" ".join(plottype.split('_')), rotation=-90, labelpad=10)

        # save/close figure
        plt.savefig(os.path.join(workdir,self.col_name +'_'+ plottype+'.'+plotFormat),format=plotFormat,bbox_inches='tight')
        plt.close()

        return

def parseCMD(iargs=None):
    inps = cmdLineParse(iargs)
    print("***Stats Function:***")
    # prep dataframe object for plotting/variogram analysis based off of user specifications
    df_stats=raiderStats(inps.fname, inps.col_name, inps.workdir, inps.bbox, inps.spacing, inps.timeinterval, inps.seasonalinterval, inps.stationsongrids, inps.colorpercentile, inps.verbose)

    # If user requests to generate all plots.
    if inps.plotall:
        print('"-plotall"==True. All plots will be made.')
        inps.station_distribution=True
        inps.station_delay_mean=True
        inps.station_delay_stdev=True
        inps.grid_heatmap=True
        inps.grid_delay_mean=True
        inps.grid_delay_stdev=True
        inps.variogramplot=True

    ###Station plots
    #Plot each individual station
    if inps.station_distribution:
        print("- Plot spatial distribution of stations.")
        unique_points=df_stats.df.groupby(['Lon', 'Lat']).size()
        df_stats([unique_points.index.get_level_values('Lon').tolist(),unique_points.index.get_level_values('Lat').tolist()],'station_distribution', workdir=os.path.join(inps.workdir,'figures'),plotFormat = inps.plot_fmt, cbounds = inps.cbounds)
    #Plot mean delay per station
    if inps.station_delay_mean:
        print("- Plot mean delay for each station.")
        unique_points=df_stats.df.groupby(['Lon', 'Lat'])[inps.col_name].mean()
        unique_points.dropna(how='any',inplace=True)
        df_stats([unique_points.index.get_level_values('Lon').tolist(),unique_points.index.get_level_values('Lat').tolist(),unique_points.values],'station_delay_mean', workdir=os.path.join(inps.workdir,'figures'),plotFormat = inps.plot_fmt, cbounds = inps.cbounds)
    #Plot delay stdev per station
    if inps.station_delay_stdev:
        print("- Plot delay stdev for each station.")
        unique_points=df_stats.df.groupby(['Lon', 'Lat'])[inps.col_name].std()
        unique_points.dropna(how='any',inplace=True)
        df_stats([unique_points.index.get_level_values('Lon').tolist(),unique_points.index.get_level_values('Lat').tolist(),unique_points.values],'station_delay_stdev', workdir=os.path.join(inps.workdir,'figures'),plotFormat = inps.plot_fmt, cbounds = inps.cbounds)

    ###Gridded station plots
    #Plot density of stations for each gridcell
    if inps.grid_heatmap:
        print("- Plot density of stations per gridcell.")
        gridarr_heatmap=np.array([np.nan if i[0] not in df_stats.df['gridnode'].values[:] else float(len(np.unique(df_stats.df['ID'][df_stats.df['gridnode']==i[0]]))) for i in enumerate(df_stats.gridpoints)]).reshape(df_stats.grid_dim)
        df_stats(gridarr_heatmap.T,'grid_heatmap', workdir=os.path.join(inps.workdir,'figures'), drawgridlines=inps.drawgridlines, colorbarfmt='%1i', stationsongrids=inps.stationsongrids, cbounds = inps.cbounds, plotFormat = inps.plot_fmt)
    #Plot mean delay for each gridcell
    if inps.grid_delay_mean:
        print("- Plot mean delay per gridcell.")
        unique_points=df_stats.df.groupby(['gridnode'])[inps.col_name].mean()
        unique_points.dropna(how='any',inplace=True)
        gridarr_heatmap=np.array([np.nan if i[0] not in unique_points.index.get_level_values('gridnode').tolist() else unique_points[i[0]] for i in enumerate(df_stats.gridpoints)]).reshape(df_stats.grid_dim)
        df_stats(gridarr_heatmap.T,'grid_delay_mean', workdir=os.path.join(inps.workdir,'figures'), drawgridlines=inps.drawgridlines, stationsongrids=inps.stationsongrids,cbounds = inps.cbounds,plotFormat = inps.plot_fmt)
    #Plot mean delay for each gridcell
    if inps.grid_delay_stdev:
        print("- Plot delay stdev per gridcell.")
        unique_points=df_stats.df.groupby(['gridnode'])[inps.col_name].std()
        unique_points.dropna(how='any',inplace=True)
        gridarr_heatmap=np.array([np.nan if i[0] not in unique_points.index.get_level_values('gridnode').tolist() else unique_points[i[0]] for i in enumerate(df_stats.gridpoints)]).reshape(df_stats.grid_dim)
        df_stats(gridarr_heatmap.T,'grid_delay_stdev', workdir=os.path.join(inps.workdir,'figures'), drawgridlines=inps.drawgridlines, stationsongrids=inps.stationsongrids, cbounds = inps.cbounds,plotFormat = inps.plot_fmt)

    ###Perform variogram analysis
    if inps.variogramplot:
        print("***Variogram Analysis Function:***")
        make_variograms=variogramAnalysis(df_stats.df, df_stats.gridpoints, inps.col_name, inps.workdir, df_stats.seasonalinterval, inps.densitythreshold, inps.verbose, binnedvariogram = inps.binnedvariogram)
        TOT_grids,TOT_res_robust_arr=make_variograms.createVariograms()
        #plot range heatmap
        print("- Plot variogram range per gridcell.")
        gridarr_range=np.array([np.nan if i[0] not in TOT_grids else float(TOT_res_robust_arr[TOT_grids.index(i[0])][0]) for i in enumerate(df_stats.gridpoints)]).reshape(df_stats.grid_dim)
        df_stats(gridarr_range.T,'range_heatmap', workdir=os.path.join(inps.workdir,'figures'), drawgridlines=inps.drawgridlines, stationsongrids=inps.stationsongrids, cbounds = inps.cbounds,plotFormat = inps.plot_fmt)
        #plot sill heatmap
        print("- Plot variogram sill per gridcell.")
        gridarr_sill=np.array([np.nan if i[0] not in TOT_grids else float(TOT_res_robust_arr[TOT_grids.index(i[0])][1]) for i in enumerate(df_stats.gridpoints)]).reshape(df_stats.grid_dim)
        gridarr_sill=gridarr_sill*(10^4) #convert to cm
        df_stats(gridarr_sill.T,'sill_heatmap', workdir=os.path.join(inps.workdir,'figures'), drawgridlines=inps.drawgridlines, colorbarfmt='%.3e', stationsongrids=inps.stationsongrids, cbounds = inps.cbounds,plotFormat = inps.plot_fmt)
