import os
import tensorflow.keras as keras
import numpy as np
import rasterio as rio
import segmentation_models as sm
from tensorflow.keras import models
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.models import Model
from MightyMosaic import MightyMosaic
import rioxarray as rxr
import sys
import pandas as pd
import argparse

#################################################################### variable definitions ##################################################################
# set backbone and backbone-specific preprocessing
BACKBONE = 'efficientnetb7'
preprocess_input = sm.get_preprocessing(BACKBONE)

#set tensorflow data options
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = \
        tf.data.experimental.AutoShardPolicy.OFF

###########################################################################################################################################################

# compile cnn from .h5 model file
def compile_cnn(model_file):
    """
        Compile u-net from .h5 model file
        :param model_file: path to .h5 model file.
        :type shape: `str` representing the path to .h5 model file.
        
        :return: u-net model
        :rtype: 'tf.keras.Model'
    """
    model = keras.models.load_model(model_file, 
                custom_objects={'binary_focal_loss_plus_jaccard_loss': sm.losses.binary_focal_jaccard_loss,
                                'precision':sm.metrics.Precision,
                                'recall':sm.metrics.Recall,
                                'f1-score':sm.metrics.FScore,
                                'iou_score': sm.metrics.IOUScore})

    return model

def compile_cnn_from_weights(weights_file):
    """
        Compile u-net from .h5 model file that includes weights only
        
        :param model_file: path to .h5 model weights file.
        :type shape: `str` representing the path to .h5 model weights file.
        
        :return: u-net model
        :rtype: tf.keras.Model
    """
    
    #set backbone and get associated preprocessing
    BACKBONE = 'efficientnetb7'
    preprocess_input = sm.get_preprocessing(BACKBONE)
    
    #define model params
    BATCH_SIZE = 32
    LR = 0.001
    EPOCHS = 220
    activation = 'softmax'
    
    #ensure that segmentation models is using the keras framework in channels_last mode
    sm.set_framework('tf.keras')
    keras.backend.set_image_data_format('channels_last')
    
    # define optimizer
    optimizer = keras.optimizers.Adam(LR)
    # enabling mixed precision to avoid underflow
    optimizer = keras.mixed_precision.LossScaleOptimizer(optimizer)
    # define loss
    total_loss = keras.losses.CategoricalCrossentropy()
    # define metrics to track
    metrics = [sm.metrics.Precision(threshold=0.5), 
               sm.metrics.Recall(threshold=0.5), 
               sm.metrics.FScore(threshold=0.5), 
               sm.metrics.IOUScore(threshold=0.5)]
    
    #build model, compile, and load weights
    model = sm.Unet(BACKBONE, classes = 1, activation=activation, encoder_weights=None, input_shape=(None, None, 4))
    model.compile(optimizer, total_loss, metrics)
    model.load_weights(weights_file)
    
    return model
        
def create_moving_window_data(image_dir, image, preprocess_input, slope = None):
    """
        Format input data for u-net prediction
        
        :param image_dir: path to directory containing images to be predicted.
        :type image_dir: `str` representing path to directory containing images to be predicted.
        :param image: name of image to predict.
        :type image: `str` representing name of image, must end in .tif.
        :param slope: slope raster.
        :type slope: `xarray.core.dataarray.DataArray` of slope raster.
        :param preprocess_input: u-net preprocessing function specific to model backbone (efficientnet-b7).
        :type preprocess_input: 'function'.
        
        :return: full_im
        :rtype: 'np.array' of preprocessed image input data
        :return: crs (projection)
        :rtype: 'str' wkt representation of image projection
        :return: gt (geotransform)
        :rtype: 'tuple' array of image geotransform in gdal format
        :return: shape
        :rtype: 'tuple' dimensions of input image in (y,x)
    """
    #open image
    xds_full_im = rxr.open_rasterio(os.path.join(image_dir, image))
    
    #get projection information (crs, geotransform, and raster shape)
    crs = xds_full_im.rio.crs.wkt
    gt = xds_full_im.rio.transform()
    shape = np.shape(xds_full_im[0])
    
    #get values as array
    full_im = xds_full_im.values
    
    #scale image values
    full_im = (full_im/10000.0)*255 #PlanetScope specific normalization
    
    #set areas where image=0 to 0.5
    image_zero_mask = full_im==0
    full_im[image_zero_mask==True] = 0.5
    
    #stack on optional slope raster
    if slope is not None:
        
        #reproject to match image extent, resolution, and crs
        slope_data = slope.rio.reproject_match(xds_full_im)
        slope_data = slope_data.assign_coords({
            "x": xds_full_im.x,
            "y": xds_full_im.y,
        })
        
        slope_data = slope_data.values[0]
        #set nodata areas in image to 0.5 in the slope raster
        slope_data[image_zero_mask[0]==True] = 0.5
        
        #add slope to stack
        full_im = np.append(full_im, [slope_data], axis=0)
    
    #reshape to bands_last format and preprocess
    full_im = full_im.transpose(2, 0, 1)
    full_im = full_im.transpose(2, 0, 1)
    full_im = preprocess_input(full_im)
    
    return full_im, crs, gt, shape

def write_geotiff(out_image_path, image_shape, transform, crs, dataset, nodata):
    
    with rio.open(out_image_path,
                       'w',
                       driver='GTiff',
                       height=image_shape[0],
                       width=image_shape[1],
                       count=1,
                       dtype=np.float64,
                       crs=crs,
                       transform=transform,
                       nodata=nodata,
                       compress='lzw') as output:
        output.write(dataset, 1)

def run_prediction(args):
    
    print('compiling model')
    model = compile_cnn(args.model_file)
        
    images = [i for i in os.listdir(args.image_dir) if i.endswith('.tif')]
    
    #open slope raster once
    if args.slope_model:
        xds_slope = rxr.open_rasterio(args.slope_file)

    for image in images: 
            
        print('formatting input data')
        if args.slope_model:
            full_im, crs, gt, image_shape = create_moving_window_data(args.image_dir, image, preprocess_input, slope=xds_slope)
        else:
            full_im, crs, gt, image_shape = create_moving_window_data(args.image_dir, image, preprocess_input)
        print('generating Mighty Mosaic')
        mosaic = MightyMosaic.from_array(full_im, (256,256), overlap_factor=args.tile_overlap, fill_mode='constant')
            
        print('predicting for input data')
        prediction_mosaic = mosaic.apply(model.predict, progress_bar=True, batch_size=args.tile_batch)
        prediction_mosaic = prediction_mosaic.get_fusion()
            
        write_geotiff(args.output_dir + image.split('.')[0] + '_unet_pred.tif', 
                    image_shape, gt, crs, prediction_mosaic[:,:,0], 0)
            
        print('')

    slope = None
    
    return

def main():
    
    parser=argparse.ArgumentParser()
    parser.add_argument("--image_dir", help="image directory to predict, expects geotiff format", required = False, default = 'examples/data/images/')
    parser.add_argument("--output_dir", help="directory to save output water masks", required = False, default = 'examples/data/predictions/')
    parser.add_argument("--model_file", help="path to the .h5 model file", required = False, default = 'res/single_class_best_model.h5')
    parser.add_argument("--slope_model", help="whether or not to use the model that considers slope angle", required = False, default = False)
    parser.add_argument("--slope_file", help="path to slope raster", required = False, default = 'examples/data/alaskaned_ex.tif')
    parser.add_argument("--tile_overlap", help="#overlap factor for moving window prediction. Higher values increasingly mitigate image tiling artifacts but can potentially introduce noise.", required = False, default = 2)
    parser.add_argument("--tile_batch", help="#batch size for model prediction. Number of tiles in the image must be divisible by this number", required = False, default = 4)
    args=parser.parse_args()
    
    run_prediction(args)
    
    
if __name__ == "__main__":
    main()
        
    
