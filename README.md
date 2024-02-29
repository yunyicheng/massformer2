# MassFormer2: Mass Spectrometry Transformer

MassFormer is a transformer-based model specifically designed for analyzing mass spectrometry data. It leverages the power of Graphormer configurations and integrates custom data handling, modeling, and training scripts to predict mass spectrometry outputs accurately.

## Features

- Utilizes `Graphormer` for deep learning on mass spectrometry data.
- Customizable model configurations through `configuration_massformer.py`.
- Efficient data collation and preprocessing with `collating_massformer.py`.
- Model training and evaluation script in `running_massformer.py`.
- Includes a demonstration notebook `model_demo.ipynb` for quick start and examples.

## Installation

1. Clone this repository.
2. Install required libraries: `torch`, `numpy`, `pandas`, `tqdm`, `matplotlib`.

## Usage

- Configure your model settings in `configuration_massformer.py`.
- Prepare your dataset and adjust preprocessing steps in `collating_massformer.py`.
- Train and evaluate the model using `running_massformer.py`.
- For a detailed example, refer to `model_demo.ipynb`.

## Contribution

Contributions to MassFormer are welcome. Please ensure to follow the coding standards and submit pull requests for any enhancements.

## License

MassFormer is open-sourced under the MIT license.