# dl-water-bodies

This repository contains code for detecting lakes and ponds in PlanetScope imagery,
in addition to notebooks used for analysis supporting the manuscript
"Using High-resolution Satellite Imagery and Deep Learning to Track Dynamic Seasonality
in Small Water Bodies", in review at Geophysical Research Letters.

## Getting Started

The main recommended avenue for using dl-water-bodies is through the publicly available set of containers
provided via this repository. If containers are not an option for your setup, follow the installation
instructions via PIP.

### Downloading the Container

All Python and GPU depenencies are installed in an OCI compliant Docker image. You can
download this image into a Singularity format to use in HPC systems.

```bash
singularity pull docker://nasanccs/dl-water-bodies:latest
```

In some cases, HPC systems require Singularity containers to be built as sandbox environments because
of uid issues (this is the case of NCCS Explore). For that case you can build a sandbox using the following
command. Depending the filesystem, this can take between 5 minutes to an hour.

```bash
singularity build --sandbox dl-water-bodies docker://nasanccs/dl-water-bodies:latest
```

If you have done this step, you can skip the Installation step since the containers already
come with all dependencies installed.

### Installation

dl-water-bodies can be installed by itself, but instructions for installing the full environments
are listed under the requirements directory so projects, examples, and notebooks can be run.

Note: PIP installations do not include CUDA libraries for GPU support. Make sure
NVIDIA libraries are installed locally in the system if not using conda.

dl-water-bodies is available on [PyPI](https://pypi.org/project/dl-water-bodies/).
To install dl-water-bodies, run this command in your terminal or from inside a container:

```bash
pip install dl-water-bodies
```

If you have installed dl-water-bodies before and want to upgrade to the latest version,
you can run the following command in your terminal:

```bash
pip install -U dl-water-bodies
```

### Running Inference of Water Masks

Use the following command if you need to perform inference using a regex that points
to the necessary files and by leveraging the default global model. The following is
a singularity exec command with options from both Singularity and the water masking
application.

Singularity options:
- '-B': mounts a filesystem from the host into the container
- '--nv': mount container binaries/devices

dlwater_cli options:
- '-r': list of regex strings to find geotiff files to predict from
- '-o': output directory to store cloud masks
- '-s': pipeline step, to generate masks only we want to predict

```bash
singularity exec --nv -B $NOBACKUP,/explore/nobackup/people,/explore/nobackup/projects \
  /explore/nobackup/projects/ilab/containers/dl-water-bodies.sif dlwater-cli \
  -o '/explore/nobackup/projects/ilab/test/dlwater-test' \
  -r '/explore/nobackup/people/almullen/smallsat_augmentation/data/planet/YKD/Ch009v024/Ch009v024_20231026_composite.tif' '/explore/nobackup/people/almullen/smallsat_augmentation/data/planet/YKD/Ch009v024/Ch009v024_20231025_composite.tif' \
  -s predict
```

To predict via slurm for a large set of files, use the following script which will start a large number
of jobs (up to your processing limit), and process the remaining files.

```bash
for i in {0..64}; do sbatch --mem-per-cpu=10240 -G1 -c10 -t05-00:00:00 -J water --wrap="singularity exec --nv -B $NOBACKUP,/explore/nobackup/people,/explore/nobackup/projects /explore/nobackup/projects/ilab/containers/dl-water-bodies.sif dlwater-cli -o '/explore/nobackup/projects/ilab/test/dlwater-test' -r '/explore/nobackup/people/almullen/smallsat_augmentation/data/planet/YKD/Ch009v024/Ch009v024_20231026_composite.tif' '/explore/nobackup/people/almullen/smallsat_augmentation/data/planet/YKD/Ch009v024/Ch009v024_20231025_composite.tif' -s predict"; done
```

For development from gpulogin1 on PRISM:

```bash
salloc -t05-00:00:00 -c8 -G1 -q ilab -J water
singularity exec --nv --env PYTHONPATH="$NOBACKUP/development/dl-water-bodies" -B $NOBACKUP,/explore/nobackup/people,/explore/nobackup/projects /explore/nobackup/projects/ilab/containers/vhr-cloudmask.sif python /explore/nobackup/people/jacaraba/development/dl-water-bodies/dl_water_bodies/view/dlwater_pipeline_cli.py -o '/explore/nobackup/projects/ilab/test/dlwater-test' -r '/explore/nobackup/people/almullen/smallsat_augmentation/data/planet/YKD/Ch009v024/Ch009v024_20231026_composite.tif' '/explore/nobackup/people/almullen/smallsat_augmentation/data/planet/YKD/Ch009v024/Ch009v024_20231025_composite.tif' '/explore/nobackup/people/almullen/smallsat_augmentation/data/planet/YKD/Ch009v024/Ch009v024_20210715_composite.tif' -s predict
```

Latest test:

```bash
singularity exec --nv --env PYTHONPATH="$NOBACKUP/development/dl-water-bodies" -B $NOBACKUP,/explore/nobackup/people,/explore/nobackup/projects /explore/nobackup/projects/ilab/containers/vhr-cloudmask.sif python /explore/nobackup/people/jacaraba/development/dl-water-bodies/dl_water_bodies/view/dlwater_pipeline_cli.py -o '/explore/nobackup/projects/ilab/test/dlwater-test' -r '/explore/nobackup/people/almullen/smallsat_augmentation/data/planet/YKD/Ch009v024/Ch009v024_20210715_composite.tif' -s predict
```

## Predictor.py

This python code handles lake and pond detection in PlanetScope imagery. To run the predictor you will first need to download the .h5 model files from https://doi.org/10.5281/zenodo.7682754. The script is setup to run on a directory of PlanetScope images and output predictions to another directory.

### example usage

```bash
python Predictor.py --image_dir='path_to_input_images' --output_dir='path_to_output_predictions' --model_file='path_to_h5_model_file' --slope_model=True --slope_file='path_to_slope_file'
```

The prediction_example.ipynb notebook allows for visualization of an example image and predicted water body mask.

### input requirements
There are two model files that can be used with the predictor, "single_class_best_model.h5" and "single_class_slope_best_model.h5". Both models expect 4-band (RGBNIR) PlanetScope imagery in geotiff format. The "single_class_slope_best_model.h5" model also needs a slope angle geotiff that overlaps spatially with the imagery. The slope file does not need to be provided in the same crs or resolution as the PlanetScope imagery, but should have a native resolution of ~30 m.
