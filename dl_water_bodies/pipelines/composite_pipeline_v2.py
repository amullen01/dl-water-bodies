# Composite pipeline
import os
import rasterio
import logging
import numpy as np
import numba as nb
import pandas as pd
from osgeo import osr
import rioxarray as rxr
from multiprocessing import Pool
from glob import glob
from dl_water_bodies.utils.gs_utils import write_geotiff
import re
from datetime import datetime
from pympler.tracker import SummaryTracker
import gc
import time
import subprocess
from dask.distributed import Client, LocalCluster, Lock
import xarray as xr

# -----------------------------------------------------------------------------
# class Composite
# -----------------------------------------------------------------------------
class Composite(object):

    def __init__(self, prediction_dir: str, mask_dir: str, years: str, months: str, 
                 output_dir: str, composite_name: str, nodata: int = 0,
                 logger: logging.Logger = None) -> None:
        
        self._prediction_dir = prediction_dir
        self._mask_dir = mask_dir
        
        self._years = years
        self._months = months
        
        self._output_dir = output_dir
        self._composite_name = composite_name

        self._nodata = nodata

        self._logger = logger
        
        self._prediction_paths = self.get_prediction_filepaths()
        self._mask_paths = self.get_corresponding_mask_filepaths()
        self._crs = None
    # -------------------------------------------------------------------------
    # build_composite
    # -------------------------------------------------------------------------
    def get_prediction_filepaths(self) -> list:
        """i
        Get filename from list of regexes
        """
        # get the paths/filenames of the regex
        filenames = []
        print(self._years)
        print(self._months)
        for im in os.listdir(self._prediction_dir):

            if im.endswith('.tif'):
                date=re.search(r'\d{8}', im)
                
                timestamp = datetime.strptime(date.group(), '%Y%m%d').date()

                if timestamp.month in self._months:
                    if timestamp.year in self._years:
                        
                        if (timestamp.month == 5) and (timestamp.day<15):
                            print(f'{timestamp} during snow period, skipping')
                            pass
                        elif (timestamp.month == 9) and (timestamp.day>15):
                            print(f'{timestamp} during snow period, skipping')
                            pass
                        else:
                            print(timestamp)    
                            filenames.append(os.path.join(self._prediction_dir, im))

        return sorted(filenames)

    def get_corresponding_mask_filepaths(self) -> list:
        """
        get masks corresponding to predictions, if no mask, will be None
        """
        #get all mask dates
        mask_dates = [re.search(r'\d{8}', d).group() for d in os.listdir(self._mask_dir)]
        
        #get full mask paths
        mask_paths = [os.path.join(self._mask_dir, m) for m in os.listdir(self._mask_dir)]
    
        #empty array to fill with mask paths corresponding to prediction paths
        corresponding_paths = []
    
        #iterate predictions
        for p in self._prediction_paths:
            date=re.search(r'\d{8}', p).group()
            
            if date in mask_dates:
                idx = mask_dates.index(date) 
                corresponding_paths.append(mask_paths[idx])
            
            else:
                corresponding_paths.append(None)
        
        return corresponding_paths      

    def mask_pl_im_with_UDM2(self, pl_pred_path, pl_mask_path):
    
        pl_pred = rxr.open_rasterio(pl_pred_path)

        pl_mask = rxr.open_rasterio(pl_mask_path)

        pl_pred = pl_pred.where((pl_mask[0]==1) & (pl_mask[1]==0) & (pl_mask[2]==0)
                          & (pl_mask[3]==0) & (pl_mask[4]==0) & (pl_mask[5]==0), 255)
        return pl_pred
    
    def mosaic_gdal(self, infiles, outfile):
    
        #infiles = ' '.join(infiles)

        if os.path.exists(outfile):
            os.remove(outfile)

        #gdal_command = ['gdal_merge.py', '-o', outfile, '-ot', 'UInt16', '-init', '255', '-n', '255', '-separate'] + infiles
        gdal_command = ['gdal_merge.py', '-o', outfile, '-ot', 'UInt16', '-init', '255', '-a_nodata', '255', '-separate', '-co', 'BIGTIFF=IF_SAFER'] + infiles
        print(gdal_command)

        result = subprocess.run(gdal_command, capture_output=True, text=True)

        if result.returncode == 0:
            print("Merging successful")
        else:
            print("Error occurred:", result.stderr)

    def prep_prediction_for_composite(self, pl_pred_path, pl_mask_path, out_path, reproject='ESRI:102001'):
        
        print('masking Planet with UDM2')
        pl_im_masked = self.mask_pl_im_with_UDM2(pl_pred_path, pl_mask_path)

        pl_im_masked = pl_im_masked.rio.write_nodata(255)

        if not reproject is None:
            pl_im_masked = pl_im_masked.rio.reproject(reproject, nodata=255)

        print(f'saving to: {out_path}')
        pl_im_masked.rio.to_raster(out_path, tiled=True, compress='deflate')
        
        del pl_im_masked
        
        return
    
    
    def composite_image_group(self, temp_dir='temp/'):
        t1_start = time.time()
        
        if not os.path.isdir(temp_dir):
            os.mkdir(temp_dir)
        
        if not os.path.isdir(self._output_dir):
            os.mkdir(self._output_dir)
        
        temp_output_paths = [os.path.join(temp_dir, p.split('/')[-1][:-4]+'_temp.tif') for p in self._prediction_paths]
        
        with Pool() as pool: 
            pool.starmap(self.prep_prediction_for_composite, zip(self._prediction_paths, self._mask_paths, temp_output_paths))
        
        print('mosaicking')
        self.mosaic_gdal(temp_output_paths, os.path.join(temp_dir, 'temp_mosaic.tif'))
        
        with LocalCluster(n_workers=10, threads_per_worker=1, memory_limit='64GB') as cluster, Client(cluster) as client:
            
            #mosaic = rxr.open_rasterio(os.path.join(temp_dir, 'temp_mosaic.tif'), masked=True, chunks={'x':1000, 'y': 1000}, lock=False)
            mosaic = rxr.open_rasterio(os.path.join(temp_dir, 'temp_mosaic.tif'), masked=True)
            
            print('compositing')
            mean = mosaic.mean(dim='band')#.round()
            mean = mean.fillna(255)
            mean = mean.where((mean<0.4) | (mean==255), 1)
            mean = mean.where(mean>=0.4, 0)
            
            mean.rio.write_crs(mosaic.rio.crs, inplace=True)
            mean.rio.write_transform(mosaic.rio.transform(), inplace=True)
            mean.rio.write_nodata(255, inplace=True)
            
            #mean.rio.to_raster(os.path.join(self._output_dir, self._composite_name), tiled=True, compress='deflate', dtype=np.uint8, lock=Lock("rio", client=client), compute=True)
            
            mean.rio.to_raster(os.path.join(self._output_dir, self._composite_name), tiled=True, compress='deflate', dtype=np.uint8)

            del mean
            del mosaic

        t1_stop = time.time()
        
        for f in glob(os.path.join(temp_dir,'*')):
            os.remove(f)
        
        print("Elapsed time (s):", t1_stop-t1_start)
    
