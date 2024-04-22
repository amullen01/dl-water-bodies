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

    parser.add_argument('-prediction_dir',
                        required=True,
                        type=str,
                        help='Path to image predictions')
    
    parser.add_argument('-mask_dir',
                        required=True,
                        type=str,
                        help='Path to image predictions')
    
    parser.add_argument('-years',
                        required=True,
                        type=int,
                        nargs='*',
                        help='years to composite')
    
    parser.add_argument('-months',
                        required=True,
                        type=int,
                        nargs='*',
                        help='months to composite')

    parser.add_argument('-o',
                        default='.',
                        help='Path to output directory')
    
    parser.add_argument('-n',
                        default='composite',
                        help='composite name')

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

    compositePipeline = Composite(prediction_dir=args.prediction_dir,
                                  mask_dir=args.mask_dir,
                                  years=args.years,
                                  months=args.months,
                                  output_dir=args.o,
                                  composite_name= args.n,
                                  nodata=args.nodata,
                                  logger=logger)
    
    compositePipeline.build_composites()


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())