import os
import re
import sys
import time
import logging
import omegaconf
import rasterio
import numpy as np
import xarray as xr
import rioxarray as rxr
import segmentation_models as sm

from glob import glob
from pathlib import Path
from omegaconf import OmegaConf
from multiprocessing import Pool, cpu_count
from omegaconf.listconfig import ListConfig

# from tensorflow_caney.model.config.cnn_config import Config
from dl_water_bodies.config.dlwater_config \
    import WaterMaskConfig as Config
from tensorflow_caney.utils.system import seed_everything, \
    set_gpu_strategy, set_mixed_precision, set_xla, array_module
from tensorflow_caney.utils.data import read_dataset_csv, \
    gen_random_tiles, modify_bands, normalize_image, rescale_image, \
    modify_label_classes, get_dataset_filenames, get_mean_std_dataset, \
    get_mean_std_metadata, read_metadata
from tensorflow_caney.utils import indices
from tensorflow_caney.model.dataloaders.segmentation import \
    SegmentationDataLoader
from tensorflow_caney.utils.losses import get_loss
from tensorflow_caney.utils.model import load_model, get_model
from tensorflow_caney.utils.optimizers import get_optimizer
from tensorflow_caney.utils.metrics import get_metrics
from tensorflow_caney.utils.callbacks import get_callbacks
from tensorflow_caney.inference import inference

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

CHUNKS = {'band': 'auto', 'x': 'auto', 'y': 'auto'}
xp = array_module()

__status__ = "Production"
__all__ = ["cp"]


# -----------------------------------------------------------------------------
# class WaterMaskPipeline
# -----------------------------------------------------------------------------
class WaterMaskPipeline(object):

    # -------------------------------------------------------------------------
    # __init__
    # -------------------------------------------------------------------------
    def __init__(
                self,
                config_filename: str = None,
                data_csv: str = None,
                model_filename: str = None,
                output_dir: str = None,
                inference_regex_list: str = None,
                mask_regex_list: str = None,
                force_delete: bool = False,
                default_config: str = 'templates/watermask_default.yaml',
                logger=None
            ):

        # TODO:
        # slurm filename in output dir

        # Set logger
        self.logger = logger if logger is not None else self._set_logger()

        # Configuration file intialization
        if config_filename is None:
            config_filename = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                default_config)
            logging.info(f'Loading default config: {config_filename}')

        # load configuration into object
        self.conf = self._read_config(config_filename, Config)

        # rewrite model filename option if given from CLI
        if model_filename is not None:
            assert os.path.exists(model_filename), \
                f'{model_filename} does not exist.'
            self.conf.model_filename = model_filename

        # rewrite output directory if given from CLI
        if output_dir is not None:
            self.conf.inference_save_dir = output_dir
            os.makedirs(self.conf.inference_save_dir, exist_ok=True)

        # rewrite inference regex list
        if inference_regex_list is not None:
            self.conf.inference_regex_list = inference_regex_list

        # rewrite mask regex list
        if mask_regex_list is not None:
            self.conf.mask_regex_list = mask_regex_list

        # Set Data CSV
        self.data_csv = data_csv

        # Force deletion
        self.force_delete = force_delete

        # Set output directories and locations
        self.images_dir = os.path.join(self.conf.data_dir, 'images')
        os.makedirs(self.images_dir, exist_ok=True)
        self.logger.info(f'Images dir: {self.images_dir}')

        self.labels_dir = os.path.join(self.conf.data_dir, 'labels')
        os.makedirs(self.labels_dir, exist_ok=True)
        self.logger.info(f'Images dir: {self.labels_dir}')

        self.model_dir = self.conf.model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        # save configuration into the model directory
        try:
            OmegaConf.save(
                self.conf, os.path.join(self.model_dir, 'config.yaml'))
        except PermissionError:
            logging.info('No permissions to save config, skipping step.')

        # Seed everything
        seed_everything(self.conf.seed)

    # -------------------------------------------------------------------------
    # _read_config
    # -------------------------------------------------------------------------
    def _read_config(self, filename: str, config_class=Config):
        """
        Read configuration filename and initiate objects
        """
        # Configuration file initialization
        schema = omegaconf.OmegaConf.structured(config_class)
        conf = omegaconf.OmegaConf.load(filename)
        try:
            conf = omegaconf.OmegaConf.merge(schema, conf)
        except BaseException as err:
            sys.exit(f"ERROR: {err}")
        return conf

    def _set_logger(self):
        """
        Set logger configuration.
        """
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s; %(levelname)s; %(message)s", "%Y-%m-%d %H:%M:%S"
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger

    # -------------------------------------------------------------------------
    # Getters
    # -------------------------------------------------------------------------
    def get_filenames(self, data_regex: str) -> list:
        """
        Get filename from list of regexes
        """
        # get the paths/filenames of the regex
        filenames = []
        if isinstance(data_regex, list) or isinstance(data_regex, ListConfig):
            for regex in data_regex:
                filenames.extend(glob(regex))
        else:
            filenames = glob(data_regex)
        assert len(filenames) > 0, f'No files under {data_regex}'
        return sorted(filenames)

    # -------------------------------------------------------------------------
    # Utils
    # -------------------------------------------------------------------------
    def delete_filename(self, filename):
        try:
            os.remove(filename)
        except FileNotFoundError:
            logging.info(f'Lock file not found {filename}')
        return

    # -------------------------------------------------------------------------
    # predict
    # -------------------------------------------------------------------------
    def predict(self) -> None:

        logging.info('Starting prediction stage')

        # Load model for inference
        model = load_model(
            model_filename=self.conf.model_filename,
            model_dir=self.model_dir,
            conf=self.conf,
            custom_objects={
                'binary_focal_loss_plus_jaccard_loss':
                sm.losses.binary_focal_jaccard_loss,
                'precision': sm.metrics.Precision,
                'recall': sm.metrics.Recall,
                'f1-score': sm.metrics.FScore,
                'iou_score': sm.metrics.IOUScore
            }
        )

        # Retrieve preprocessing step
        # TODO: make this an option of the config filename
        BACKBONE = 'efficientnetb7'
        preprocess_input = sm.get_preprocessing(BACKBONE)

        # Gather filenames to predict
        if len(self.conf.inference_regex_list) > 0:
            data_filenames = sorted(
                self.get_filenames(self.conf.inference_regex_list))
        else:
            data_filenames = sorted(
                self.get_filenames(self.conf.inference_regex))
        logging.info(f'{len(data_filenames)} files to predict')

        # Gather filenames to mask predictions
        if len(self.conf.mask_regex_list) > 0:
            mask_filenames = sorted(
                self.get_filenames(self.conf.mask_regex_list))
        else:
            mask_filenames = sorted(self.get_filenames(self.conf.mask_regex))
        logging.info(f'{len(mask_filenames)} masks for prediction')

        # iterate files, create lock file to avoid predicting the same file
        for filename, mask_filename in zip(data_filenames, mask_filenames):

            # start timer
            start_time = time.time()

            # set output directory
            basename = os.path.basename(os.path.dirname(filename))
            output_directory = os.path.join(
                self.conf.inference_save_dir, basename)
            os.makedirs(output_directory, exist_ok=True)

            # set prediction output filename
            output_filename = os.path.join(
                output_directory,
                f'{Path(filename).stem}-{self.conf.experiment_type}.tif')

            # lock file for multi-node, multi-processing
            lock_filename = f'{output_filename}.lock'

            # if the prediction file exists, or the lock file
            # is present, skip the prediction and move on to
            # the next filename
            if os.path.isfile(output_filename) or \
                    os.path.isfile(lock_filename):
                logging.info(f'{output_filename} already predicted.')
                continue

            # okay, let's try to get some predictions done
            try:
                logging.info(f'Starting to predict {filename}')

                # create lock file
                open(lock_filename, 'w').close()

                # open data filename
                image = rxr.open_rasterio(filename)
                logging.info(f'Prediction shape: {image.shape}')

                # open mask filename
                mask = rxr.open_rasterio(mask_filename)
                logging.info(f'Mask shape: {mask.shape}')

                # check bands in imagery, do not proceed if one band
                if image.shape[0] == 1:
                    logging.info(
                        'Skipping file because of non sufficient bands')
                    continue

            except rasterio.errors.RasterioIOError:
                logging.info(f'Skipped {filename}, probably corrupted.')
                if self.force_delete:
                    self.delete_filename(lock_filename)
                continue

            # Transpose the image for channel last format
            image = image.transpose("y", "x", "band")

            # scale image values, PlanetScope specific normalization
            image = (image / 10000.0) * 255
            temporary_tif = preprocess_input(image.values)

            # Sliding window prediction
            prediction, probability = \
                inference.sliding_window_tiler_multiclass(
                    xraster=temporary_tif,
                    model=model,
                    n_classes=self.conf.n_classes,
                    overlap=self.conf.inference_overlap,
                    batch_size=self.conf.pred_batch_size,
                    threshold=self.conf.inference_treshold,
                    standardization=self.conf.standardization,
                    mean=None,
                    std=None,
                    normalize=1.0,
                    rescale=self.conf.rescale,
                    window=self.conf.window_algorithm,
                    probability_map=self.conf.probability_map
                )

            # Set nodata values on mask
            # Planet imagery does not have the burned no-data.
            # The default is 0 which we are adding here.
            nodata = 0  # prediction.rio.nodata

            # Mask out using the Planet quota
            mask = np.squeeze(mask[0, :, :].values)
            prediction[mask == 0] = nodata

            # TODO: Fix no-data from corners to 255
            # nodata_mask = np.squeeze(image[:, :, 0].values)
            # prediction[nodata_mask == 0] = np.nan#255

            # Drop image band to allow for a merge of mask
            image = image.drop(
                dim="band",
                labels=image.coords["band"].values[1:],
            )

            # Get metadata to save raster prediction
            prediction = xr.DataArray(
                np.expand_dims(prediction, axis=-1),
                name=self.conf.experiment_type,
                coords=image.coords,
                dims=image.dims,
                attrs=image.attrs
            )

            # Add metadata to raster attributes
            prediction.attrs['long_name'] = (self.conf.experiment_type)
            prediction.attrs['model_name'] = (self.conf.model_filename)
            prediction = prediction.transpose("band", "y", "x")

            # do not need this if it comes from numpy
            prediction = prediction.where(image != nodata)

            # prediction.rio.write_nodata(
            #    self.conf.prediction_nodata, encoded=True, inplace=True)
            prediction.rio.write_nodata(nodata, encoded=True, inplace=True)

            # Save output raster file to disk
            prediction.rio.to_raster(
                output_filename,
                BIGTIFF="IF_SAFER",
                compress=self.conf.prediction_compress,
                driver=self.conf.prediction_driver,
                dtype=self.conf.prediction_dtype
            )
            del prediction

            # delete lock file after the prediction is done
            self.delete_filename(lock_filename)

            logging.info(f'Finished processing {output_filename}')
            logging.info(f"{(time.time() - start_time)/60} min")

        return
