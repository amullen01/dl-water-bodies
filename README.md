# dl-water-bodies
This repository contains code for detecting lakes and ponds in PlanetScope imagery, in addition to notebooks used for analysis supporting the manuscript "Using High-resolution Satellite Imagery and Deep Learning to Track Dynamic Seasonality in Small Water Bodies", in review at Geophysical Research Letters.

## Predictor.py
This python code handles lake and pond detection in PlanetScope imagery. To run the predictor you will first need to download the .h5 model files from https://doi.org/10.5281/zenodo.7682754. The script is setup to run on a directory of PlanetScope images and output predictions to another directory.

### example usage
```python Predictor.py --image_dir='path_to_input_images' --output_dir='path_to_output_predictions' --model_file='path_to_h5_model_file' --slope_model=True --slope_file='path_to_slope_file'```

The prediction_example.ipynb notebook allows for visualization of an example image and predicted water body mask.

### input requirements
There are two model files that can be used with the predictor, "single_class_best_model.h5" and "single_class_slope_best_model.h5". Both models expect 4-band (RGBNIR) PlanetScope imagery in geotiff format. The "single_class_slope_best_model.h5" model also needs a slope angle geotiff that overlaps spatially with the imagery. The slope file does not need to be provided in the same crs or resolution as the PlanetScope imagery, but should have a native resolution of ~30 m.
