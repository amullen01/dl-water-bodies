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

    # Process command-line args.
    desc = 'Use this application to perform CNN segmentation.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-c',
                        '--config-file',
                        type=str,
                        required=False,
                        default=None,
                        dest='config_file',
                        help='Path to the configuration file')

    parser.add_argument('-d',
                        '--data-csv',
                        type=str,
                        required=False,
                        default=None,
                        dest='data_csv',
                        help='Path to the data configuration file')

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
                        '--regex-list',
                        type=str,
                        nargs='*',
                        required=False,
                        dest='inference_regex_list',
                        help='Inference regex list',
                        default=['*.tif'])

    parser.add_argument('-mr',
                        '--mask-regex-list',
                        type=str,
                        nargs='*',
                        required=False,
                        dest='mask_regex_list',
                        help='Mask regex list',
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
        args.data_csv,
        args.model_filename,
        args.output_dir,
        args.inference_regex_list,
        args.mask_regex_list,
        args.force_delete
    )

    # Regression CHM pipeline steps
    # if "preprocess" in args.pipeline_step:
    #    pipeline.preprocess()
    # if "train" in args.pipeline_step:
    #    pipeline.train()
    if "predict" in args.pipeline_step:
        pipeline.predict()

    logging.info(f'Took {(time.time()-timer)/60.0:.2f} min.')

    return


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
