# CLI
import argparse
import logging
import sys
import os

from dl_water_bodies.pipelines.composite_pipeline import Composite

# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():

    # Process command-line args.
    desc = 'Use this application to generate ' + \
        'composites of planet water body predictions.'

    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-df',
                        required=True,
                        type=str,
                        help='Path to the dataframe containing' +
                            ' image and prediction paths')

    parser.add_argument('-o',
                        default='.',
                        help='Path to output directory')

    parser.add_argument('-nodata',
                        default=0,
                        type=int,
                        help='nodata value')

    args = parser.parse_args()

    fmt = '[%(asctime)s %(name)s] ' + \
        '(%(filename)s %(lineno)d): ' + \
        '%(levelname)s %(message)s'

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(ch)

    month = os.path.basename(args.df).split('_composite_list.csv')[0]

    logger.info(f'Month/identifier: {month}')

    compositePipeline = Composite(input_dataframe_path=args.df,
                                  output_dir=args.o,
                                  month=month,
                                  nodata=args.nodata,
                                  logger=logger)
    
    compositePipeline.build_composites()


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())