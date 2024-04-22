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

   
    def build_composites(self):

        min_x, max_x, min_y, max_y = self.get_bounding_box(
            self._prediction_paths
        )

        sum_water, num_predictions, water_prob = self.get_prediction_stack(
            min_x, max_x, min_y, max_y)

        np.save('nb_sum_water.npy', sum_water)
        np.save('nb_num_preds.npy', num_predictions)

        gt = [min_x, 3.0, 0, max_y, 0, -3.0]
        projection = osr.SpatialReference()
        projection.ImportFromEPSG(32603)
        projection = projection.ExportToWkt()

        waterprob_path = os.path.join(self._output_dir,
                                      f'{self._composite_name}_prob_composite.tif')

        self._logger.info(f'Writing water prob to: {waterprob_path}')

        write_geotiff(
            waterprob_path,
            np.shape(water_prob[0]), gt,
            projection, water_prob[0], 10)

        npred_path = os.path.join(self._output_dir,
                                  f'{self._composite_name}_num_obs_composite.tif')

        self._logger.info(f'Writing num predictions to: {npred_path}')

        write_geotiff(
            npred_path,
            np.shape(num_predictions[0]), gt,
            projection, num_predictions[0], 10)

        nwater_path = os.path.join(self._output_dir,
                                   f'{self._composite_name}_num_water_composite.tif')

        self._logger.info(f'Writing num water to: {nwater_path}')

        write_geotiff(
            nwater_path,
            np.shape(sum_water[0]), gt,
            projection, sum_water[0], 10)

        return sum_water, num_predictions, water_prob, min_x, max_y

    # -------------------------------------------------------------------------
    # get_bounding_box 
    # -------------------------------------------------------------------------
    def get_bounding_box(self, prediction_names):
        """
        Get UTM bounding box for a group of images (upper left corner of pixels)
        using multiprocessing.
        :param prediction_names: Image names to get bounding box from.
        :type prediction_names: list(str) only image names + .tif
        :return: (min_x, max_x, min_y, max_y) coordinates of upper left of
            pixels in bounding box
        :rtype: tuple(float, float, float, float)
        """

        min_x = 10000000
        max_x = 0
        min_y = 10000000
        max_y = 0

        def update_coordinates(result):
            nonlocal min_x, max_x, min_y, max_y
            shape, gt = result
            if gt[0] < min_x:
                min_x = gt[0]
            if gt[0] + gt[1] * shape[1] > max_x:
                max_x = gt[0] + gt[1] * shape[1]
            if gt[3] > max_y:
                max_y = gt[3]
            if gt[3] + gt[5] * shape[0] < min_y:
                min_y = gt[3] + gt[5] * shape[0]

        with Pool() as pool:
            results = pool.map(self.get_image_info, prediction_names)
            pool.close()
            pool.join()

        # Update coordinates based on the results
        for result in results:
            update_coordinates(result)

        self._logger.info(f'Bbox: [{min_x}, {max_x}, {min_y}, {max_y}]')

        return min_x, max_x, min_y, max_y

    # -------------------------------------------------------------------------
    # get_image_info 
    # -------------------------------------------------------------------------
    @staticmethod
    def get_image_info(prediction):
        """
        Get shape and gt from a single image.
        :param prediction: Image name to get information from.
        :type prediction: str
        :return: (shape, gt) tuple representing image shape and
            transform information
        :rtype: tuple
        """
        with rasterio.open(prediction) as src:
            image_data = src.read(1)
            shape = np.shape(image_data)
            gt = src.get_transform()
        print(shape, gt)
        return shape, gt

    # -------------------------------------------------------------------------
    # get_prediction_stack 
    # -------------------------------------------------------------------------
    def get_prediction_stack(self, min_x, max_x, min_y, max_y):
        """
            Get total predictions, total water predictions, and proportion
            water for each pixel for the entire domain
            :param min_x: minimum x-coordinate from get_bounding_box.
            :type min_x: `float`
            :param max_x: maximum x-coordinate from get_bounding_box.
            :type max_x: `float`
            :param min_y: minimum y-coordinate from get_bounding_box.
            :type min_y: `float`
            :param max_y: maximum y-coordinate from get_bounding_box.
            :type max_y: `float`
            
            :return: (sum_water, num_predictions, sum_water/num_predictions)
            :rtype: tuple(float, float, float)
        """
        
        #initialize empty arrays to build
        sum_water = np.array([])
        num_predictions = np.array([])

        for i in range(0,len(self._prediction_paths)):

            self._logger.info(f'Processing {self._prediction_paths[i]}')

            #open prediction, pad to bounding box extent
            xds_mask = rxr.open_rasterio(self._mask_paths[i], nodata=0)
            xds_mask.values = xds_mask.values.astype(np.uint8)
            xds_mask = xds_mask.rio.pad_box(
                minx=min_x,
                miny=min_y,
                maxx=max_x,
                maxy=max_y
            )
            #open prediction, pad to bounding box extent
            xds_pred =  rxr.open_rasterio(self._prediction_paths[i], nodata=0)
            xds_pred = xds_pred.rio.pad_box(
                minx=min_x,
                miny=min_y,
                maxx=max_x,
                maxy=max_y
            )

            #get values
            mask_data = np.array(xds_mask.values)
            xds_im = None

            pred_data = np.array(xds_pred.values)
            xds_pred = None

            if i == 0:
                # First iteration
                sum_water = np.zeros(pred_data.shape)
                num_predictions = np.zeros(pred_data.shape) 

            sum_water, num_predictions = self.composite_kernel(
                i, mask_data, pred_data, sum_water, num_predictions)

            mask_data = None
            pred_data = None

        return sum_water, num_predictions, sum_water/num_predictions

    # -------------------------------------------------------------------------
    # composite_kernel 
    # -------------------------------------------------------------------------
    @staticmethod
    @nb.njit(parallel=True, fastmath=False)
    def composite_kernel(iteration, mask_data, pred_data, sum_water,
                        num_predictions):
        """
        Compositing kernel that performs the thresholding and
        gets the total water and number of predictions for each pixel.
        :param iteration: Iteration index
        :param image_data: Image array to get masking from
        :param pred_data: Prediction array
        :param sum_water: Initialized sum of water array we calculate values
            for
        :param num_predictions: Initialized number of predictions array
        """
        for y in nb.prange(pred_data.shape[1]):
            for x in nb.prange(pred_data.shape[2]):

                pred = pred_data[0, y, x]
              
                #catch any nans or otherwise unexpected values
                if pred != 1:
                    pred=0
                
                #pred = np.uint32(pred)

                pixel_valid = 1
                
                #valid mask pixel: clear = 1 & snow = 0 & shadow = 0 & haze_light = 0 & haze_heavy = 0 & cloud = 0 
                #if invalid pixel
                #if mask_data[0, y, x] != 1 or np.any(mask_data[1:6, y, x]):
                if mask_data[0, y, x] != 1 or mask_data[1, y, x] != 0 or mask_data[4, y, x] != 0 or mask_data[5, y, x] != 0: 
                    pixel_valid = 0
                    pred = np.nan
                    
                   
                sum_water[0, y, x] = np.nansum(np.array([sum_water[0, y, x], pred]))
                num_predictions[0, y, x] = np.sum(np.array([num_predictions[0, y, x], pixel_valid]))

        return sum_water, num_predictions
