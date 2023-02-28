import sys
sys.path.append("..")
sys.path
import os
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from matplotlib import pyplot as plt
import json
from utils.gs_utils import write_geotiff
from osgeo import gdal, osr
import rioxarray as rxr

#grouping = 'monthly'

predictions_dir = ''
images_dir = ''
image_descriptor = '_masked.tif'
prediction_descriptor = '_unet_pred_pp.tif'
water_prob_image_dir = ''

def get_bounding_box(prediction_names):
    """
        Get UTM bounding box for group of images (upper left corner of pixels)
        :param prediction_names: prediction names to get bounding box from.
        :type prediction_names: `list(str)` only image names + .tif
        
        :return: (min_x, max_x, min_y, max_y) coordinates of upper left of pixels in bounding box
        :rtype: tuple(float, float, float, float)
    """
    #initialize minimum and maximum coordinates to extremes
    min_x = 10000000
    max_x = 0
    min_y = 10000000
    max_y = 0

    #loop through predictions
    for prediction in prediction_names:

        with rasterio.open(os.path.join(predictions_dir, prediction)) as src:  

            image_data = src.read(1).astype(np.float32) # rescale bands for "analytic_sr" asset

            shape = np.shape(image_data)

            gt = src.get_transform()

            if gt[0] < min_x: #gt[0] is the min x of the image
                min_x = gt[0]

            if gt[0] + gt[1] * shape[1] > max_x: #gt[0] + gt[1] * shape[1] is the x-coordinate of the left side of the rightmost pixel
                max_x = gt[0] + gt[1] * shape[1]

            if gt[3] > max_y: #gt[3] is the max y of the image
                max_y = gt[3]

            if gt[3] + gt[5] * shape[0] < min_y: #gt[3] + gt[5] * shape[0] is the y coordinate of the top of the bottommost pixel

                min_y = gt[3] + gt[5] * shape[0]
                
    return min_x, max_x, min_y, max_y

def get_prediction_stack(min_x, max_x, min_y, max_y, prediction_names, image_names):
    """
        Get total predictions, total water predictions, and proportion water for each pixel for the entire domain
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

    for i in range(0,len(prediction_names)):
        
        #open image, pad to bounding box extent
        xds_im = rxr.open_rasterio(os.path.join(images_dir, image_names[i]), nodata=0)
        xds_im.values = xds_im.values.astype(np.uint8)
        xds_im = xds_im.rio.pad_box(
            minx=min_x,
            miny=min_y,
            maxx=max_x,
            maxy=max_y
        )
        
        #open prediction, pad to bounding box extent
        xds_pred =  rxr.open_rasterio(os.path.join(predictions_dir, prediction_names[i]), nodata=0)
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
        
        #threshold prediction so p>0.5 = water, otherwise nan
        pred_data[pred_data>=0.5] = 1
        pred_data[pred_data<0.5] = np.nan
        
        #create mask for pixels that were predicted to be not water, not including imge nodata areas
        data_mask = np.where((pred_data!=1) & (image_data[0]!=0))
        
        #change values from nan to 0 for these pixels
        pred_data[data_mask] = 0
        data_mask = None
        
        pred_data = pred_data.astype(np.float16)
        
        if sum_water.size == 0: #initialize arrays
            
            sum_water = np.copy(pred_data)
            pred_data[pred_data>=0]=1
            pred_data[np.isnan(pred_data)]=0
            pred_data = pred_data.astype(np.uint16)
            num_predictions = pred_data
            
        else: #add to arrays
            
            sum_water = np.nansum(np.array([sum_water,pred_data]), axis=0, dtype=np.float16)
            pred_data[pred_data>=0]=1
            pred_data[pred_data!=1]=0
            pred_data = pred_data.astype(np.uint16)
            num_predictions = np.nansum(np.array([num_predictions, pred_data]), axis=0, dtype=np.uint16)
            
    return sum_water, num_predictions, sum_water/num_predictions

def build_composites(prediction_names, image_names):
    
    print('getting bounding box')
    min_x, max_x, min_y, max_y = get_bounding_box(prediction_names)

    print('building prediction stack')
    sum_water, num_predictions, water_prob = get_prediction_stack(min_x, max_x, min_y, max_y, prediction_names, image_names)
    
    return sum_water, num_predictions, water_prob, min_x, max_y

def main():
    
    if len(sys.argv) > 1:
        print('argument read')
    
        df = pd.read_csv(sys.argv[1])
        month = sys.argv[1].split('/')[1].split('_composite_list.csv')[0]
        
        image_names = df['associated_image'].to_list()
        prediction_names = df['prediction_name'].to_list()

        sum_water, num_predictions, water_prob, min_x, max_y = build_composites(prediction_names, image_names)

        gt = [min_x, 3.125, 0, max_y, 0, -3.125]
        projection = osr.SpatialReference()
        projection.ImportFromEPSG(32606)
        projection = projection.ExportToWkt()

        write_geotiff(water_prob_image_dir + month + 'prob_composite.tif', np.shape(water_prob[0]), gt, projection, water_prob[0], 0)
        write_geotiff(water_prob_image_dir + month + 'num_obs_composite.tif', np.shape(num_predictions[0]), gt, projection, num_predictions[0], 0)
        write_geotiff(water_prob_image_dir +  month + 'num_water_composite.tif', np.shape(sum_water[0]), gt, projection, sum_water[0], 0)
        
if __name__ == "__main__":
    
    main()