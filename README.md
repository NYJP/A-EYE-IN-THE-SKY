# POLYGENCE Experiments

This repository contains the Python version of the experimental code used for the paper.
- `dataloader.py`: dataset construction and preprocessing
- `model_suite.py`: model construction, training utilities, validation, and testing


## Experiments

The code keeps the experiment differences behind configuration variables.

1. Normalization method
   Controlled in `TrainingConfig.normalization` in `model_suite.py`
2. Presence of Data Transformation and Selection
   Controlled in `DatasetBuildConfig` in `dataloader.py`
3. Addition of Time Context
   Controlled by `DatasetBuildConfig.include_time_context` in `dataloader.py`
4. Presence of Generous Labelling
   Controlled by `DatasetBuildConfig.label_strategy` and timing tolerances in `dataloader.py`
5. Model Architecture
   Controlled by `ModelConfig.backbone_name` in `model_suite.py`
6. Pretrained Condition
   Controlled by `ModelConfig.pretrained` and `ModelConfig.freeze_backbone` in `model_suite.py`
7. Class-Balanced Training
   Controlled by `TrainingConfig.use_class_weights` in `model_suite.py`
8. Hidden Layer
   Controlled by `ModelConfig.hidden_dim` in `model_suite.py`
9. Dropout Rate
   Controlled by `ModelConfig.dropout` in `model_suite.py`
10. Learning Rate Scheduler
   Controlled by `TrainingConfig.scheduler_name` in `model_suite.py`

## Data Pipeline

`dataloader.py` is one configurable data pipeline.

Included options:

- frame cropping
- frame resizing
- grayscale conversion
- contrast adjustment
- negative-frame sampling
- original, generous, and one-sided labeling
- temporal context using previous/current/next frames

Experiment presets are available in `DATASET_PRESETS`.


## Model Pipeline

`model_suite.py` is one configurable module.

Included options:
- ResNet, MobileNet, EfficientNet, and ConvNeXt backbones
- feed-forward MLP model for temporal frame input
- pretrained and non-pretrained variants
- grayscale or temporal-context input
- optional hidden layer
- configurable dropout
- optional class-weighted loss
- selectable normalization
- step or plateau learning-rate scheduling
- optional staged backbone unfreezing
- train, validation, and test utilities

Experiment presets are available in `MODEL_PRESETS`.


## Reproducing Experiments

The table below shows how each original variant maps to the configuration system.

### Data Variants

| Variant | Main experiment focus |
| --- | --- |
| `DATA B&W OG` | Baseline labeling and preprocessing |
| `DATA B&W GENEROUS` | Experiment 4: generous labeling |
| `DATA B&W ONE SIDED` | Experiment 4: one-sided labeling window |
| `DATA B&W w MEMORY` | Experiment 3: temporal context |

### Model Variants

| Variant | Main experiment focus |
| --- | --- |
| `resnet not pretrained` | Experiment 5, 6 |
| `pretrained resnet` | Experiment 6 |
| `pretrained resnet with time context` | Experiment 3 |
| `not pretrained resnet with time context` | Experiment 3 |
| `mobilenet not pretrained` | Experiment 5 |
| `efficientnet not pretrained` | Experiment 5 |
| `convnet not pretrained` | Experiment 5 |
| `feed-forward model` | Experiment 5 |
| `min-max normalization` | Experiment 1 |
| `z normalization` | Experiment 1 |
| `weighted training` | Experiment 7 |
| `dropout of 0` | Experiment 9 |
| `step based time schedular` | Experiment 10 |
| `fine tuned resnet with freezing (pretrained)` | Experiment 6 |

### Typical Workflow

1. Choose a dataset preset from `DATASET_PRESETS`.
2. Build and save the dataset splits with `build_dataset_splits(...)` and `save_dataset_splits(...)`.
3. Choose a model preset from `MODEL_PRESETS`.
4. Build the model, loss, optimizer, and scheduler from the selected configuration.
5. Run training with `train_one_epoch(...)` or `run_training_epoch(...)`.
6. Run validation with `validate_model(...)`.
7. Run final evaluation with `test_model(...)`.

## Notes

- The Python files are the main reference point for the project.
- Some plotting, intermediate checks, and one-off inspection code from the raw source files was not carried into the final modules.
