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

from dl_water_bodies.utils.gs_utils import write_geotiff


# -----------------------------------------------------------------------------
# class Composite
# -----------------------------------------------------------------------------
class Composite(object):

    def __init__(self, input_dataframe_path: str,
                 output_dir: str, month: str, nodata: int = 0,
                 logger: logging.Logger = None) -> None:

        self._dataframe = pd.read_csv(input_dataframe_path)

        self._output_dir = output_dir

        self._month = month

        self._nodata = nodata

        self._image_paths = self._dataframe['associated_image'].to_list()
        self._prediction_paths = self._dataframe['prediction_name'].to_list()

        self._logger = logger

    # -------------------------------------------------------------------------
    # build_composite
    # -------------------------------------------------------------------------
    def build_composites(self):

        min_x, max_x, min_y, max_y = self.get_bounding_box(
            self._prediction_paths
        )

        sum_water, num_predictions, water_prob = self.get_prediction_stack(
            min_x, max_x, min_y, max_y)

        np.save('nb_sum_water.npy', sum_water)
        np.save('nb_num_preds.npy', num_predictions)

        gt = [min_x, 3.125, 0, max_y, 0, -3.125]
        projection = osr.SpatialReference()
        projection.ImportFromEPSG(32606)
        projection = projection.ExportToWkt()

        waterprob_path = os.path.join(self._output_dir,
                                      f'{self._month}prob_composite.tif')

        self._logger.info(f'Writing water prob to: {waterprob_path}')

        write_geotiff(
            waterprob_path,
            np.shape(water_prob[0]), gt,
            projection, water_prob[0], 0)

        npred_path = os.path.join(self._output_dir,
                                  f'{self._month}num_obs_composite.tif')

        self._logger.info(f'Writing num predictions to: {npred_path}')

        write_geotiff(
            npred_path,
            np.shape(num_predictions[0]), gt,
            projection, num_predictions[0], 0)

        nwater_path = os.path.join(self._output_dir,
                                   f'{self._month}num_water_composite.tif')

        self._logger.info(f'Writing num water to: {nwater_path}')

        write_geotiff(
            nwater_path,
            np.shape(sum_water[0]), gt,
            projection, sum_water[0], 0)

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

            #open image, pad to bounding box extent
            xds_im = rxr.open_rasterio(self._image_paths[i], nodata=0)
            xds_im.values = xds_im.values.astype(np.uint8)
            xds_im = xds_im.rio.pad_box(
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
            image_data = np.array(xds_im.values)
            xds_im = None

            pred_data = np.array(xds_pred.values)
            xds_pred = None

            if i == 0:
                # First iteration
                sum_water = np.zeros(pred_data.shape)
                num_predictions = np.zeros(pred_data.shape) 

            sum_water, num_predictions = self.composite_kernel(
                i, image_data, pred_data, sum_water, num_predictions,
                nodata=self._nodata)

    
        return sum_water, num_predictions, sum_water/num_predictions

    # -------------------------------------------------------------------------
    # composite_kernel 
    # -------------------------------------------------------------------------
    @staticmethod
    @nb.njit(parallel=True, fastmath=False)
    def composite_kernel(iteration, image_data, pred_data, sum_water,
                        num_predictions, nodata=10):
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

                if pred == nodata:
                    pred = np.nan

                if pred != 1:
                    if image_data[0, y, x] != 0: 
                        pred = 0

                pred = np.float32(pred)

                if iteration == 0:

                    sum_water[0, y, x] = pred

                    if pred >= 0:
                        pred = 1
                    if np.isnan(pred):
                        pred = 0

                    pred = np.uint32(pred)
                    num_predictions[0, y, x] = pred
                
                else:

                    sum_water[0, y, x] = np.nansum(
                        np.array([sum_water[0, y, x], pred])
                    )

                    if pred >= 0:
                        pred = 1

                    if pred != 1:
                        pred = 0

                    pred = np.uint32(pred)

                    num_predictions[0, y, x] = np.nansum(
                        np.array([num_predictions[0, y, x], pred])
                    )

        return sum_water, num_predictions
