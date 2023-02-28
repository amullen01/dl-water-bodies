# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 18:55:07 2022

@author: amullen
"""
import sys
sys.path.append("..")
sys.path
from utils.gs_utils import get_shapes
import os
import rioxarray as rxr
from rasterio import features
import geopandas as gpd
import pandas as pd
import shapely
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from rasterio.mask import mask
import rasterio as rio
import utils
import ast

# This script takes the u-net output (single-band raster with values ranging from 0-1), binarizes it, filters water body objects using a slope threshold and nodata filter, and outputs a shapefile containing the resulting water body objects.

#path to image directory that predictions were generated from
images_dir = ''

#path_to_predictions
predictions_dir = ''

#path to output postprocessed shapefiles
out_shapefiles_dir = ''
out_geotiffs_dir = ''

#image descriptor
image_descriptor = '_masked.tif'

#path to slope raster to use for slope filtering
path_to_slope = ''

NODATA_FILTER = True
SLOPE_FILTER = False
SLOPE_THRESHOLD = 7


def get_gradients(gdf, buffer_sizes):

    gdf['geometry'] = gdf['geometry'].buffer(0)

    gdf['ndwi_water'] = np.zeros(len(gdf))
    gdf['ndvi_water'] = np.zeros(len(gdf))
    
    distance_counter = 0
    sum_buffer = 0
    
    for i in range(0, len(buffer_sizes)):
        
        distance_counter+=buffer_sizes[i]
        
        gdf['ndwi_{}m'.format(distance_counter)] = np.zeros(len(gdf))
        gdf['ndvi_{}m'.format(distance_counter)] = np.zeros(len(gdf))

        buffered_gdf = gdf.buffer(distance_counter)
        buffered_area = buffered_gdf.difference(gdf.buffer(sum_buffer))
        gdf['{}m_geometry'.format(distance_counter)] = buffered_area
        
        sum_buffer+=buffer_sizes[i]

    return gdf

def calc_buffer_stats(image, gdf, buffer_field, stats_field):
    
    for index, row in gdf.iterrows():
        
        masked_raster, masked_raster_transform = mask(image, [row[buffer_field]], invert=False, nodata = 0)
        masked_raster = masked_raster.astype(np.float32)
        masked_raster[masked_raster==0] = np.nan
        
        ndvi = (masked_raster[3]-masked_raster[2]) / (masked_raster[3]+masked_raster[2])
        ndwi = (masked_raster[1]-masked_raster[3]) / (masked_raster[1]+masked_raster[3])

        if 'ndwi' in stats_field:
            gdf.at[index,stats_field] = np.nanmean(ndwi)
            
        if 'ndvi' in stats_field:
            gdf.at[index,stats_field] = np.nanmean(ndvi)
        
    return gdf
    
def run_postprocessing(path_to_image, path_to_prediction, path_to_out_shapefile, path_to_output_geotiff, slope = None):
    """
        Run postprocessing procedure
        :param path_to_image: path to .h5 model file.
        :type shape: `str` representing the path to .h5 model file.
        :param path_to_prediction: path to .h5 model file.
        :type shape: `str` representing the path to .h5 model file.
        :param path_to_out_shapefile: path to .h5 model file.
        :type shape: `str` representing the path to .h5 model file.
        :param slope: path to .h5 model file.
        :type shape: `str` representing the path to .h5 model file.
        
        :return: u-net model
        :rtype: 'tf.keras.Model'
    """
    #get shapes from prediction
    with rio.open(path_to_prediction) as src_pred:
        
        #read prediction band
        prediction = src_pred.read(1)
        
        #set every pixel >= 0.5 to water, everything < 0.5 to not water
        prediction[prediction<0.5] = 0
        prediction[prediction>=0.5] = 1
        
        #get geotransform and crs
        t_prediction = src_pred.transform
        proj_prediction = src_pred.crs
        
        #get shapes from raster
        gdf_pred = get_shapes(prediction, t_prediction, proj_prediction)
        #open image in "read + mode"
        
    
        if SLOPE_FILTER == True:
            #reproject/resample slope raster to match extent, resolution, and projection of image
            image_match = rxr.open_rasterio(path_to_prediction)
            slope_res = slope.rio.reproject_match(image_match)
            slope_res = slope_res.assign_coords({
                "x": image_match.x,
                "y": image_match.y,
            }) 
        
        with rio.open(path_to_image, 'r+') as src:
            
            #iterate through predicted water body shapes
            for index, row in gdf_pred.iterrows():

                #optional nodata filter
                if NODATA_FILTER == True:

                        #set image nodata to 255 (this only changes metadata value, does not affect raster values)
                        src.nodata = 65535

                        masked_water_body = rio.mask.mask(src, [row['geometry'].buffer(10)], crop = True)[0]

                        #Planet imagery comes with nodata value of 0, so if a water body contains 0 in the corresponding image, remove it
                        if 0 in masked_water_body:

                            masked_water_body = rio.mask.mask(src, [row['geometry'].buffer(10)], crop = False)[0]
                            gdf_pred = gdf_pred.drop(index)
                            prediction[masked_water_body[0]!=65535] = 0

                            src.nodata = 0 #set metadata nodata value back to zero

                            continue

                        else:

                            src.nodata = 0 #set metadata nodata value back to zero
                        
            #optional slope filter
            if SLOPE_FILTER == True:
                
                src.nodata = 65535
                #clip slope to the exent of the water body and calculate mean slope
                masked_slope = slope_res.rio.clip([row['geometry']], gdf_pred.crs, drop=True)[0]
            
                if np.nanmean(masked_slope)>SLOPE_THRESHOLD:
                    masked_slope = slope_res.rio.clip([row['geometry']], gdf_pred.crs, drop=False)
                    #remove water body if slope is above threshold
                    gdf_pred = gdf_pred.drop(index)
                    prediction[masked_slope!=65535] = 0
                    #print('dropped water body above slope threshold')
                    
                src.nodata = 0
            
        if len(gdf_pred)>0:
            print('writing to {}'.format(path_to_out_shapefile))
            print(gdf_pred.dtypes)
            gdf_pred.to_file(path_to_out_shapefile)
            
            #write postprocessed mask to tif
        out_meta = src_pred.meta.copy()
        out_meta.update({'driver':'GTiff',
                         'width':src_pred.shape[1],
                         'height':src_pred.shape[0],
                         'count':1,
                         'dtype':'float64',
                         'crs':src_pred.crs, 
                         'transform':src_pred.transform,
                         'nodata':0})
                            
        with rio.open(fp=path_to_output_geotiff, mode='w',**out_meta) as dst:
            dst.write(prediction, 1)
        
def main():
    
    #check if script is being called with arguments from SLURM
    if len(sys.argv) > 1:
        print('argument read')
        path_df = pd.read_csv(sys.argv[1])
        image_paths = path_df['image_paths'].to_list()
        prediction_paths = path_df['prediction_paths'].to_list()
        out_shapefiles = path_df['out_shapefiles'].to_list()
        out_geotiffs = path_df['out_geotiffs'].to_list()
    
    else:

        #build arrays of images, predictions, and output shapefile paths
        image_paths = [os.path.join(images_dir, image) for image in os.listdir(images_dir) if image.endswith(image_descriptor)]
        prediction_paths = [os.path.join(predictions_dir, image.split(image_descriptor)[0] + '_unet_pred.tif') for image in os.listdir(images_dir) if image.endswith(image_descriptor)]
        out_shapefiles = [os.path.join(out_shapefiles_dir, image.split(image_descriptor)[0] + '_unet_pred.shp') for image in os.listdir(images_dir) if image.endswith(image_descriptor)]
        out_geotiffs = [os.path.join(out_geotiffs_dir, image.split(image_descriptor)[0] + '_pp.tif') for image in os.listdir(images_dir) if image.endswith(image_descriptor)]
    
    #open slope raster if slope filtering on
    if SLOPE_FILTER == True:
        
        xds_slope = rxr.open_rasterio(path_to_slope, masked=True)
    
    #loop through images and predictions
    for i in range(0,len(image_paths)):
        print(prediction_paths[i][:-4]+'_pp.tif')
        if os.path.exists(prediction_paths[i][:-4]+'_pp.tif'):
            
            continue
        
        print('opening {}, associated with image: {}'.format(prediction_paths[i], image_paths[i]))
        
        if SLOPE_FILTER == True:
            
            run_postprocessing(image_paths[i], prediction_paths[i], out_shapefiles[i], out_geotiffs[i], slope = xds_slope)
        
        else:
            
            run_postprocessing(image_paths[i], prediction_paths[i], out_shapefiles[i], out_geotiffs[i])

    
if __name__ == "__main__":
    
    main()
    
