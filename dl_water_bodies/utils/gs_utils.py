# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 11:58:48 2021

@author: amullen
"""
import os
from osgeo import gdal, osr, gdalconst
import numpy as np
from skimage import filters
import rasterio
from rasterio import features
import geopandas as gpd

def gaussian_smoothing(image_path, output_path, sigma, input_ndv = None):
    
    with rasterio.open(image_path) as src:
            
            img_data = src.read(np.arange(1,src.count+1).tolist()).astype(float) # rescale bands for "analytic_sr" asset
            
            ndv = src.nodata
            
            if input_ndv != None:
                ndv = input_ndv
                img_data[img_data==input_ndv] = np.nan
                
            else:
                img_data[img_data==ndv] = np.nan    
            
            for band_num in range(0,src.count):
                
                img_data[band_num] = filters.gaussian(img_data[band_num], sigma)
    
    with rasterio.open(output_path,
                       'w',
                       driver='GTiff',
                       height=src.height,
                       width=src.width,
                       count=src.count,
                       dtype=rasterio.uint16,
                       crs=src.crs,
                       transform=src.transform,
                       compression = 'lzw',
                       nodata = ndv) as output:
        output.write(img_data[0].astype(np.uint16), 1)
        output.write(img_data[1].astype(np.uint16), 2)
        output.write(img_data[2].astype(np.uint16), 3)
        output.write(img_data[3].astype(np.uint16), 4)
        
    return

def resample(original_source, desired_source, destination_path, nodata_value = -9999):

    original_proj = original_source.GetProjection()
    original_gt = original_source.GetGeoTransform()
    
    desired_gt = desired_source.GetGeoTransform()
    desired_proj = desired_source.GetProjection()
    
    if original_proj != desired_proj or original_gt != desired_gt:
        
        desired_cols = desired_source.RasterXSize
        desired_rows = desired_source.RasterYSize
        desired_band = desired_source.GetRasterBand(1)
        
        
        minX = desired_gt[0]
        maxX = desired_gt[0] + (desired_cols * desired_gt[1])
        
        minY = desired_gt[3] + (desired_rows * desired_gt[5])
        maxY = desired_gt[3]
    
        warp_options = gdal.WarpOptions(format = 'GTiff', 
                                    outputBounds = [minX, minY, maxX, maxY], 
                                    width = desired_cols, height = desired_rows,
                                    srcSRS = original_proj, dstSRS = desired_proj, 
                                    resampleAlg = gdalconst.GRA_Bilinear, outputType = gdalconst.GDT_Float32, creationOptions = ['COMPRESS=LZW'], 
                                    dstNodata=nodata_value)
        
        out = gdal.Warp(destination_path, original_source, options = warp_options)
        #out.GetRasterBand(1).SetNoDataValue(nodata_value)
        
        desired_band = None
        return out
    
    return original_source


def write_geotiff(new_image_name, image_shape, geotransform, projection, dataset, ndv):
    
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(new_image_name, image_shape[1], image_shape[0], 1, 
                            gdalconst.GDT_Float32, ['COMPRESS=LZW'])
    
    outdata.SetGeoTransform(geotransform)   ##sets same geotransform as input
    srs = osr.SpatialReference()            # establish encoding
    #srs.ImportFromEPSG(projection)                # WGS84 lat/long
    srs.ImportFromWkt(projection)
    outdata.SetProjection(srs.ExportToWkt())
    outdata.GetRasterBand(1).SetNoDataValue(ndv)
    outdata.GetRasterBand(1).WriteArray(dataset)
    
    #outdata.GetRasterBand(1).SetNoDataValue(10000)##if you want these values transparent
    outdata.FlushCache() ##saves to disk!!
    outdata = None

def stack_singleband_raster(raster, existing_stack):
        
    
    raster_band = raster.GetRasterBand(1)
    raster_array = raster_band.ReadAsArray().astype(np.float16)
    raster_array[np.isinf(raster_array)] = np.nan
    stackedBands = np.append(existing_stack, [raster_array], axis=0)
        
    raster = None
    raster_band = None
    raster_array = None
        
    return stackedBands

def get_shapes(mask_data, gt, proj):
    
    feature_collection = {
        'type': 'FeatureCollection',
        'features': []
    }

    for shape, value in features.shapes(mask_data, connectivity=4, transform=gt):
        feature_collection["features"].append({"type": "Feature",
                                               "id": value,
                                               "properties": {"value": value},
                                               "geometry": shape})
        
    gdf = gpd.GeoDataFrame.from_features(
        feature_collection, crs=proj)

    gdf['geometry'] = gdf['geometry'].buffer(0)
    gdf = gdf.loc[gdf['value']==1]
    gdf['area'] = gdf['geometry'].area


    return gdf
