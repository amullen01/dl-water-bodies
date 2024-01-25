from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class WaterMaskConfig:
    """
    CNN data configuration class (embedded with OmegaConf).
    """

    # Directory to store all data files
    data_dir: str

    # Directory to store model artifacts
    model_dir: Optional[str]

    # String with model function (e.g. tensorflow object)
    model: str

    # Directory to store inference output files
    inference_save_dir: str = "results"

    # Experiment name to track
    experiment_name: str = "unet-cnn"

    # Experiment type to track (normally embedded in the inference output)
    experiment_type: str = "landcover"

    # seed to control the randomization
    seed: Optional[int] = 24

    # GPU devices to utilize
    gpu_devices: str = "0,1,2,3"

    # Bool to enable mixed_precision
    mixed_precision: Optional[bool] = True

    # Bool to enable linear acceleration
    xla: Optional[bool] = False

    # Input bands from the incoming dataset
    input_bands: List[str] = field(
        default_factory=lambda: [
            "Blue", "Green", "Red", "NIR1", "HOM1", "HOM2"]
    )

    # Output bands that will be used to train and predict from
    output_bands: List[str] = field(
        default_factory=lambda: ["Blue", "Green", "Red", "NIR1"]
    )

    # List of strings to support the modification of labels
    modify_labels: Optional[List[str]] = None

    # Subtract 1 from labels
    substract_labels: Optional[bool] = False

    # Expand dimensions of training data (useful for binary class)
    expand_dims: bool = True

    # Tile size or chips to predict from
    tile_size: int = 256

    # Force algorithm to include more than one class per tile
    include_classes: bool = False

    # Perform data augmentation on the fly
    augment: bool = True

    # Center crop option on the fly
    center_crop: bool = False

    # Specify no-data value from labels
    no_data: Optional[int] = 0

    # Specify wether or not we accept tiles with no-data values
    nodata_fractional: bool = False

    # Specify the fraction on no-data values allowed
    nodata_fractional_tolerance: float = 0.75

    # Directory to store json metadata for experiments reproducibility
    json_tiles_dir: Optional[str] = None

    # Specify wether we upload the dataset from the json file
    dataset_from_json: bool = False

    # Value to normalize data with
    normalize: Optional[float] = 1.0
    normalize_label: Optional[float] = 1.0

    # Determine if we rescale the data
    rescale: Optional[str] = None

    # Specify standardization mechanism
    standardization: Optional[str] = None

    # Specify CNN batch size for training
    batch_size: int = 32

    # Specify number of classes to work with
    n_classes: int = 1

    # Specify test size ratio
    test_size: float = 0.20

    # List to store mean values per band
    mean: List[float] = field(default_factory=lambda: [])

    # List to store std values per band
    std: List[float] = field(default_factory=lambda: [])

    # Loss function expression, expects a loss function
    loss: str = "tf.keras.losses.CategoricalCrossentropy"

    # Optimizer function expression, expects an optimizer function
    optimizer: str = "tf.keras.optimizers.Adam"

    # Metrics function expression, expects list of metrics
    metrics: List[str] = field(
        default_factory=lambda: ["tf.keras.metrics.Recall"])

    # Callbacks function expression, expects list of metrics
    callbacks: List[str] = field(
        default_factory=lambda: ["tf.keras.callbacks.ModelCheckpoint"]
    )

    # Transfer learning components, options: feature-extraction, fine-tuning
    transfer_learning: Optional[str] = None

    # Specify weights file for transfer learning
    transfer_learning_weights: Optional[str] = None

    # Specify step to fine-tune model on
    transfer_learning_fine_tune_at: Optional[int] = None

    # Model learning rate
    learning_rate: Optional[float] = 0.0001

    # Maximum number of epochs to run the model
    max_epochs: Optional[int] = 6000

    # Number of epochs to wait for the model to improve
    patience: int = 7

    # Model filename for inference
    model_filename: Optional[str] = None

    # String regex to find rasters to predict
    inference_regex: Optional[str] = "*.tif"

    # List regex to find rasters to predict (multiple locations)
    inference_regex_list: Optional[List[str]] = field(
        default_factory=lambda: [])

    # Window size for sliding window operations
    window_size: Optional[int] = 8120

    # Overlap between the tiles to avoid artifacts
    inference_overlap: Optional[float] = 0.5

    # Threshold used for binary classification
    inference_treshold: Optional[float] = 0.5

    # Inference padding value to replace no-data boundaries
    inference_pad_value: Optional[int] = 1000

    # Window algorithm for smoothing prediction
    window_algorithm: Optional[str] = 'triang'

    # Batch size for inference
    pred_batch_size: Optional[int] = 128

    # Output probability map with prediction
    probability_map: Optional[bool] = False

    # Specify dtype of prediction
    prediction_dtype: Optional[str] = 'uint8'

    # Specify no-data value for prediction
    prediction_nodata: Optional[int] = 255

    # Specify compression for prediction
    prediction_compress: Optional[str] = 'LZW'

    # Specify driver for prediction (COG and ZARR included)
    prediction_driver: Optional[str] = 'GTiff'

    # Regex to find model metadata
    metadata_regex: Optional[str] = None

    # Validation database for validation of predictions
    validation_database: Optional[str] = None

    # Test classes
    test_classes: Optional[List[str]] = field(
        default_factory=lambda: [])

    # Test colors for vis, must match test_classes size
    test_colors: Optional[List[str]] = field(
        default_factory=lambda: [])

    # Test truth regex where labels are stored to compare with
    test_truth_regex:  Optional[str] = None

"""
@dataclass
class CloudMaskConfig(Config):

    # model parameters from Huggingface
    hf_repo_id: str = 'nasa-cisto-data-science-group/vhr-cloudmask'
    hf_model_filename: str = 'cloudmask-vietnam-senegal-46-0.04.hdf5'
"""