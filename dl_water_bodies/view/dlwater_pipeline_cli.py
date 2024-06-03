import sys
import time
import logging
import argparse
from dl_water_bodies.pipelines.dlwater_pipeline import WaterMaskPipeline

# -----------------------------------------------------------------------------
# main
#
# python dlwater_pipeline_cli.py -c config.yaml -d config.csv -s preprocess
# -----------------------------------------------------------------------------      
def main():

    desc = 'Use this application to perform CNN segmentation.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-c',
                        '--config-file',
                        type=str,
                        required=False,
                        default=None,
                        dest='config_file',
                        help='Path to the configuration file')

    parser.add_argument('-s',
                        '--step',
                        type=str,
                        nargs='*',
                        required=True,
                        dest='pipeline_step',
                        help='Pipeline step to perform',
                        default=['preprocess', 'train', 'predict'],
                        choices=['preprocess', 'train', 'predict'])

    parser.add_argument('-m',
                        '--model-filename',
                        type=str,
                        required=False,
                        default=None,
                        dest='model_filename',
                        help='Path to model file')

    parser.add_argument('-o',
                        '--output-dir',
                        type=str,
                        default=None,
                        required=False,
                        dest='output_dir',
                        help='Path to output directory')

    parser.add_argument('-r',
                        '--image-dir',
                        type=str,
                        nargs='*',
                        required=False,
                        dest='image_dir',
                        help='directory with images to predict',
                        default=['*.tif'])

    parser.add_argument('-mr',
                        '--mask-dir',
                        type=str,
                        nargs='*',
                        required=False,
                        dest='mask_dir',
                        help='directory with corresponding image masks',
                        default=['*.tif'])

    parser.add_argument('-f',
                        '--force-delete',
                        default=False,
                        required=False,
                        dest='force_delete',
                        action='store_true',
                        help='Force the deletion of lock files')
    args = parser.parse_args()

    # Setup timer to monitor script execution time
    timer = time.time()

    # Initialize pipeline object
    pipeline = WaterMaskPipeline(
        args.config_file,
        args.model_filename,
        args.output_dir,
        args.image_dir,
        args.mask_dir,
        args.force_delete
    )

    # Regression CHM pipeline steps
    # if "preprocess" in args.pipeline_step:
    #    pipeline.preprocess()
    # if "train" in args.pipeline_step:
    #    pipeline.train()
    if "predict" in args.pipeline_step:
        pipeline.predict()

    logging.info('Took {} min.'.format((time.time()-timer)/60.0))

    return


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
