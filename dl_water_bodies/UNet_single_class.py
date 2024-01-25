# -*- coding: utf-8 -*-
"""A module-level docstring

Notice the comment above the docstring specifying the encoding.
Docstrings do appear in the bytecode, so you can access this through
the ``__doc__`` attribute. This is also what you'll see if you call
help() on a module or any other Python object.
"""
import os
from rasterio.plot import show
import cv2
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import segmentation_models as sm
from tensorflow.keras import models
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.models import Model
from osgeo import gdal, gdalconst
from skimage.color import rgb2hsv
from rasterio.features import sieve, shapes
import albumentations as A
from functools import partial
import glob       # for arrays modifications
import rioxarray as rxr
import random
from numpy.random import default_rng
import pandas as pd
import seaborn as sns

#paths to image directories
x_train_dirs = ['']

#paths to mask directories
y_train_dirs = ['']

#paths to slope directories
slope_dirs = ['']

#paths to twi directories
twi_dirs = ['']

#path to npz directories
npz_dirs = ['']

#tile ids to be used during training
path_to_tile_ids = ''

#directory to save training history, validation stats, plots, and models
model_save_dir = ''

#name of training run to associate with saved data
model_name = ''

BATCH_SIZE = 32
LR = 0.0001
EPOCHS = 200

generate_npz = True #must be set to True if .npz data does not exist. If it does exist, set to false to save time on training
BACKBONE = 'efficientnetb7'

#albumentations transformations
transforms = A.Compose([
    A.ShiftScaleRotate(p=0.5),
    A.HorizontalFlip(p=0.125),
    A.VerticalFlip(p=0.125)
    ])

def gen_npz(ids, image_dirs, slope_dirs, masks_dirs, twi_dirs, preprocessing):
    """
    Generate .npz files to store preprocessed training data to feed directly into model.
    Args:
    
        ids (list(str)): list of image tile names to generate training data for, example:['']
        image_dirs (list(str)): list of directories containing image tiles
        slope_dirs (list(str)): list of directories containing slope tiles
        masks_dirs (list(str)): list of directories containing mask tiles
        twi_dirs (list(str)): object id from npz file to get labels from
        preprocessing (obj): segmentation models preprocessing object based on backbone specified
        
    Return:
        get image and label datasets
    ----------
    Example
    ----------
        get_tensorslices(data_dir='images', img_id='x', label_id='y')
    """
    
    for image in ids:
        
        image_dir = ''
        masks_dir = ''
        slope_dir = ''
        twi_dir = ''
        
        #loop through directories for different AOIs to make sure we are looking in the proper directory for each imag
        for i in image_dirs:
            if os.path.exists(os.path.join(i, image)):
                image_dir = i
                
        for m in masks_dirs:
            if os.path.exists(os.path.join(m, image)):
                masks_dir = m
                
        for s in slope_dirs:
            if os.path.exists(os.path.join(s, image)):
                slope_dir = s
                
        for t in twi_dirs:
            if os.path.exists(os.path.join(t, image)):
                twi_dir = t
                
        save_dir = masks_dir[:-5] + 'npz/'
        
        
        # read data
        with rasterio.open(os.path.join(image_dir, image)) as src:  
            
            #get array from image
            image_data = (src.read(np.arange(1,src.count+1).tolist()).astype(np.float16)/10000.0)*255 #PlanetScope specific rescaling
            
            #get mask nodata
            image_zero_mask = image_data==0
            
            #set to different value
            image_data[image_zero_mask==True] = 0.5
            
            #open slope and append to data stack
            '''
            with rasterio.open(os.path.join(slope_dir, image)) as slope_src:  
            
                slope_data = slope_src.read(1).astype(np.float16)

                slope_data[image_zero_mask[0]==True] = 0.5
                slope_data[slope_data<=-1000] = 1
                
                image_data = np.append(image_data, [slope_data], axis=0)
            '''  
            '''
            #open twi and append to data stack
            with rasterio.open(os.path.join(twi_dir, image)) as twi_src:  
            
                twi_data = twi_src.read(1).astype(np.float16)

                twi_data[image_zero_mask[0]==True] = 0.5
                twi_data[twi_data<=-1000] = 0
                
                image_data = np.append(image_data, [twi_data], axis=0)
            '''    
            #reshape to 'bands last' format
            image_data = image_data.transpose(2, 0, 1)
            image_data = image_data.transpose(2, 0, 1)
            
            #apply preprocessing function
            image_data = preprocessing(image_data)
            
        with rasterio.open(os.path.join(masks_dir, image)) as src_mask:  
            
            #open mask data as np array
            mask = src_mask.read(np.arange(1,src_mask.count+1).tolist()) 
            
            #set all non-water to 0, water should = 1
            mask[np.expand_dims(image_zero_mask[0],axis=0)==True] = 0
            mask[mask<0.0] = 0
            mask[mask>1.0] = 0
            
            mask = mask.astype(np.float16)
                
            #transpose array to "bands-last" format
            mask = mask.transpose(2,0,1)
            mask = mask.transpose(2,0,1)
        
        #save as npz
        np.savez(f'{save_dir}{image[:-4]}.npz', x=image_data, y=mask)

def get_tensorslices(data_dirs=[], img_id='x', label_id='y', test_proportion = 0.2, validation_proportion = 0.1):
    """
    Getting tensor slices from .npz files
    Args:
        data_dirs (list): list of data directories where .npz data resides
        img_id (str): object id from npz file to get data from
        label_id (str): object id from npz file to get labels from
        test_proportion (float): proportion of training data to use as test data
        validation_proportion (float): proportion of training data to use as a holdout validation set
    Return:
        (tuple)
        train_ids (list): list of image ids used for training
        train_images (numpy array): array of shape (n_train_images, 256, 256, n_bands)
        train_labels (numpy array): array of shape (n_train_labels, 256, 256, 1)
        test_ids (list): list of image ids used for testing
        test_images (numpy array): array of shape (n_test_images, 256, 256, n_bands)
        test_labels (numpy array): array of shape (n_test_labels, 256, 256, 1)
        validation_ids (list): list of image ids used for validation
        validation_images (numpy array): array of shape (n_validation_images, 256, 256, n_bands)
        validation_labels datasets (numpy array): array of shape (n_validation_labels, 256, 256, 1)
    ----------
    Example
    ----------
        get_tensorslices(data_dirs=['training_data/AOI/A/npz/', 'training_data/AOI/B/npz/'], img_id='x', label_id='y')
    """
    # open files and generate training dataset
    images = np.array([])
    labels = np.array([])
    ids = np.array([])
    
    for data_dir in data_dirs:
        print('reading in {}'.format(data_dir))
        # read all data files from disk
        for f in glob.glob(f'{data_dir}/*'):
            
            with np.load(f) as data:
                # vstack image batches into memory
                if images.size:  # if images has elements, vstack new batch
                    images = np.vstack([images, [data[img_id]]])
                else:  # if images empty, images equals new batch
                    images = np.array([data[img_id]])
                # vstack label batches into memory
                if labels.size:  # if labels has elements, vstack new batch
                    labels = np.vstack([labels, [data[label_id]]])
                else:  # if labels empty, images equals new batch
                    labels = np.array([data[label_id]])
                if ids.size:  # if ids has elements, vstack new batch
                    ids = np.vstack([ids, [f[:-4] + '.tif']])
                else:  # if ids empty, images equals new batch
                    ids = np.array([f[:-4] + '.tif'])
                    
    # now we will randomly separate the train and test data based on specified proportion
    num_test_samples = int(len(images) * test_proportion)
    num_validation_samples = int(len(images) * validation_proportion)
    
    rng = default_rng(seed=4)
    
    random_indices = rng.choice(len(images), size=num_test_samples+num_validation_samples, replace=False)
    
    random.seed(4)
    test_indices = np.array(random.sample(list(random_indices), num_test_samples))
    validation_indices = random_indices[~np.isin(random_indices,test_indices)]
    
    # create test set
    test_images = images[test_indices]
    test_labels = labels[test_indices]
    test_ids = ids[test_indices]
    
    # create validation set
    validation_images = images[validation_indices]
    validation_labels = labels[validation_indices]
    validation_ids = ids[validation_indices]
    
    # delete test and validation images from training stacks
    train_images = np.delete(images, random_indices, axis=0)
    train_labels = np.delete(labels, random_indices, axis=0) 
    train_ids = np.delete(ids, random_indices, axis=0)
    
    return train_ids, train_images, train_labels, test_ids, test_images, test_labels, validation_ids, validation_images, validation_labels

#albumentations function
def set_shapes(img, label, img_shape=(256,256,4), mask_shape=(256,256,1)):
    
    img.set_shape(img_shape)
    label.set_shape(mask_shape)

    return img, label

#albumentations function
def aug_fn(image, label, img_size):
   
    data = {"image":image, "masks": [label[:,:,0]]} #uncomment for single class
    
    aug_data = transforms(**data)
    
    aug_img = aug_data["image"]
    
    aug_img = tf.image.resize(aug_img, size=[img_size, img_size])
    
    aug_masks = aug_data["masks"]
    
    aug_masks = tf.transpose(aug_masks, perm = [2,0,1])
    aug_masks = tf.transpose(aug_masks, perm = [2,0,1])

    return aug_img, aug_masks

#albumentations function
def process_data(image, label, img_size):
    
    aug_data = tf.numpy_function(func=aug_fn, inp=[image, label, img_size], Tout=tuple([tf.float32, tf.float32]))
    
    return aug_data

def get_training_dataset(dataset, batch_size = 8, drop_remainder=True):
    
    """
    Return training dataset to feed tf.fit.
    Args:
        dataset (tf.dataset): tensorflow dataset
        batch_size (int): training batch size
        drop_remainder (bool): drop remaineder when value does not match batch
    Return:
        tf dataset for training
    ----------
    Example
    ----------
        get_tensorslices(data_dir='images', img_id='x', label_id='y')
    """
    
    # create dataset
    dataset = dataset.map(partial(process_data, img_size=256)) #use albumentations to augment data
    dataset = dataset.map(set_shapes) #ensure input data has correct shape
    dataset = dataset.shuffle(3000, reshuffle_each_iteration=True) #fill shuffle buffer
    dataset = dataset.repeat() #repeat an undefined amount of times
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
   
    # prefetch next batch while training (autotune prefetch buffer size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def water_bodies_by_size(validation_labels, predictions):
    """
    Calculate accuracy metrics by water body size
    Args:
        validation_labels (numpy array): array of shape (n_validation_labels, 256, 256, 1)
        predictions (numpy array): array of shape (n_predictions, 256, 256, 1)
    Return:
    tuple
        IoU_scores (list(float))
        Precision_scores (list(float))
        Recall_scores (list(float))
        predicted_num_shapes (list(float)) 
        predicted_sieved (numpy array): array of shape (n_predictions, n_sieve_levels, 256, 256, 1)
        pixel_cutoffs (list(float)): lower area boundary (in pixels)
    ----------
    Example
    ----------
        iou, precision, recall, num_shapes, sieved_predictions, pixel_cutoffs = water_bodies_by_size(validation_labels, predictions)
    """

    #define pixel cutoffs
    pixel_cutoffs = [1000, 500, 250, 150, 60, 1]

    true_sieved = []
    predicted_sieved = []
    true_num_shapes = np.zeros(np.shape(pixel_cutoffs))
    predicted_num_shapes = np.zeros(np.shape(pixel_cutoffs))

    for x in range(0,len(predictions)):

        true_sieved_levels = []
        predicted_sieved_levels = []
        
        #get individual label and prediction
        true = validation_labels[x, :, :, 0]
        predicted = predictions[x, :, :, 0]

        true = true.astype(np.int16)
        #binarize prediction
        predicted[predicted<0.5] = 0
        predicted[predicted>=0.5] = 1
        predicted = predicted.astype(np.int16)

        i=0
        
        for pixel_cutoff in pixel_cutoffs:
            
            #sieve the true mask and prediction based on water body size, starting with largest water bodies
            ts = rasterio.features.sieve(true, pixel_cutoff, out=np.zeros(np.shape(true), np.int16))
            ps = rasterio.features.sieve(predicted, pixel_cutoff, out=np.zeros(np.shape(predicted), np.int16))
            
            #append size group to list
            true_sieved_levels.append(ts)
            predicted_sieved_levels.append(ps)

            #add to shape count for water body sieve level
            for x in range(0, len(list(rasterio.features.shapes(ts)))):
                if list(rasterio.features.shapes(ts))[x][1]==1:
                    true_num_shapes[i] += 1

            for y in range(0, len(list(rasterio.features.shapes(ps)))):
                if list(rasterio.features.shapes(ps))[y][1]==1:
                    predicted_num_shapes[i] += 1

            #set values of true and predicted mask from current sieve level to 0
            true[ts!=0] = 0
            predicted[ps!=0] = 0

            i+=1

        true_sieved.append(true_sieved_levels)
        predicted_sieved.append(predicted_sieved_levels)  
        
    #calculate scores for every sieve level and append results to respective lists
    IoU_scores = []
    for sieve_level in range(0,len(pixel_cutoffs)):
    
        m = tf.keras.metrics.MeanIoU(2)
        m.update_state(np.array(true_sieved)[:,sieve_level,:,:], np.array(predicted_sieved)[:,sieve_level,:,:])
        IoU_scores.append(m.result().numpy())
        
    Precision_scores = []
    for sieve_level in range(0,len(pixel_cutoffs)):
        
        m = tf.keras.metrics.Precision()
        m.update_state(np.array(true_sieved)[:,sieve_level,:,:], np.array(predicted_sieved)[:,sieve_level,:,:])
        Precision_scores.append(m.result().numpy())
        
    Recall_scores = []
    for sieve_level in range(0,len(pixel_cutoffs)):

        m = tf.keras.metrics.Recall()
        m.update_state(np.array(true_sieved)[:,sieve_level,:,:], np.array(predicted_sieved)[:,sieve_level,:,:])
        Recall_scores.append(m.result().numpy())
        
    return IoU_scores, Precision_scores, Recall_scores, predicted_num_shapes, predicted_sieved, pixel_cutoffs

def save_history_plot(history, out_img):
    """
    Save plots for training loss and accuracy by epoch
    Args:
        history (tensorflow training history): output of model.fit
        out_img (string): path to save output plot (must end in .jpg, .png, etc.)
    Return:
        None: saves plot to specified filepath
    ----------
    Example
    ----------
        save_history_plot(history, out_img)
    """
    
    # Plot training & validation iou_score values
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.plot(history.history['iou_score'])
    plt.plot(history.history['val_iou_score'])
    plt.title('Model IOU-score')
    plt.ylabel('IOU-Score')
    plt.xlabel('Epoch')
    plt.ylim(0, 1)
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    plt.savefig(out_img, dpi=300)

def main():
    
    preprocess_input = sm.get_preprocessing(BACKBONE)
    
    df_ids = pd.read_csv(path_to_tile_ids)
    ids = df_ids['ids'].to_numpy()
    
    if generate_npz == True:
        print('generating npz')
        gen_npz(ids, x_train_dirs, slope_dirs, y_train_dirs, twi_dirs, preprocess_input)
    
    #get data and convert to tf.Dataset object
    train_ids, train_images, train_labels, test_ids, test_images, test_labels, validation_ids, validation_images, validation_labels= get_tensorslices(npz_dirs)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images.astype(np.float32), train_labels.astype(np.float32)))
    valid_dataset = tf.data.Dataset.from_tensor_slices((test_images.astype(np.float32), test_labels.astype(np.float32))).batch(8)
    
    #define model params
    activation = 'sigmoid'
    sm.set_framework('tf.keras')
    keras.backend.set_image_data_format('channels_last')
    optimizer = keras.optimizers.Adam(LR)
    optimizer = keras.mixed_precision.LossScaleOptimizer(optimizer) # enabling mixed precision to avoid underflow
    total_loss = sm.losses.binary_focal_jaccard_loss # define loss
    # define metrics to track
    metrics = [sm.metrics.Precision(threshold=0.5), 
               sm.metrics.Recall(threshold=0.5), 
               sm.metrics.FScore(threshold=0.5), 
               sm.metrics.IOUScore(threshold=0.5)]
    
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    
    callbacks = [
    keras.callbacks.ModelCheckpoint(model_save_dir+model_name+'_best_model.h5', save_weights_only=False, save_best_only=True, mode='min', monitor = 'val_loss'),
    keras.callbacks.ReduceLROnPlateau(monitor = 'loss', mode = 'min', verbose = 1),
    keras.callbacks.CSVLogger(model_save_dir + model_name+'_train.csv')
    ]
    
    strategy = tf.distribute.MirroredStrategy(devices= ["/GPU:0","/GPU:1", "/GPU:2", "/GPU:3"],cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # Open a strategy scope.
    with strategy.scope():

      # Everything that creates variables should be under the strategy scope.
      # In general this is only model construction & `compile()`.
      # create model

        #model = sm.Unet(BACKBONE, classes = 1, activation=activation, encoder_weights=None, input_shape=(None, None, 5))
        model = sm.Unet(BACKBONE, classes = 1, activation=activation, encoder_weights=None, input_shape=(None, None, 4))
        model.compile(optimizer, total_loss, metrics)
        
    # Train the model on all available devices.
    # train model
    history = model.fit(
        get_training_dataset(train_dataset, batch_size = BATCH_SIZE).with_options(options),
        steps_per_epoch=int(np.ceil(len(train_ids)/BATCH_SIZE)), 
        epochs=EPOCHS,
        callbacks=callbacks, 
        validation_data=valid_dataset.with_options(options),
        validation_steps=None,
        use_multiprocessing = False
    )
    
    #save training history figure
    hist_plot_path=model_save_dir + model_name+ '_history_plot.jpg'
    save_history_plot(history, hist_plot_path)
    
    #get validation scores and save to csv
    scores = model.evaluate(validation_images, validation_labels)
    df_scores=pd.DataFrame(scores)
    df_scores.to_csv(model_save_dir + model_name+ '_val.csv')
    
    #get validation predictions as tensor and get size-based stats
    predictions = model.predict(validation_images)
    IoU_scores, Precision_scores, Recall_scores, predicted_num_shapes, predicted_sieved, pixel_cutoffs = water_bodies_by_size(validation_labels, predictions)
    
    #save size-based stats to image
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=pixel_cutoffs, y = IoU_scores, label = 'IoU')
    sns.lineplot(x=pixel_cutoffs, y = IoU_scores, alpha = 0.3)
    sns.scatterplot(x=pixel_cutoffs, y = Precision_scores, label = 'Precision')
    sns.lineplot(x=pixel_cutoffs, y = Precision_scores, alpha = 0.3)
    sns.scatterplot(x=pixel_cutoffs, y = Recall_scores, label = 'Recall')
    sns.lineplot(x=pixel_cutoffs, y = Recall_scores, alpha = 0.3)
    plt.ylabel('Score')
    plt.xlabel('Water Body Area (pixels)')
    plt.savefig(model_save_dir + model_name+ '_size_stats.jpg', dpi=300)
    
if __name__ == "__main__":
    main()