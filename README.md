# log well prediction

Welcome to the **log well prediction** repository! This project focuses on predicting specific targets using machine learning models. Below, you'll find a comprehensive guide to get started with the project, including data preparation, model training, and result evaluation.

## Table of Contents

1. [Introduction](#introduction)
2. [Setup](#setup)
3. [Data Preparation](#data-preparation)
4. [Model Training](#model-training)
5. [Results](#results)
6. [Contributing](#contributing)
7. [License](#license)

## Introduction

This project leverages Transformer-KAN techniques to predict log well from a given dataset. The main workflow involves data preparation, model training, and evaluating the model's performance.

## Setup

Before you begin, ensure you have the necessary dependencies installed. You can do this by running:

```bash
pip install -r requirements.txt
```

## Data Preparation

First, you need to prepare your data by running the `data_pre.py` script. This script processes your dataset and saves it in the specified directory. If you need to change the prediction target, you can modify the following code in the `tool_for_pre` module:

```python
parser.add_argument('--input_directory', type=str, default=r'data_save\54口井的数据集', help='输入地址')
```
Please go to for complete data https://www.kaggle.com/datasets/charzhang/well-log-data/data download

Your data should be stored in the `data_save` directory. For testing purposes, you can use the dataset provided in the `数据读取的案例数据` file.

To run the data preparation script, use the following command:

```bash
python data_pre.py
```

## Model Training

Next, train your model using the `train.py` script. You can execute the training process from the command line with customizable parameters. Here is an example command:

```bash
python train.py --model_name Transformer_KAN --hidden_size 32 --num_layers 4 --num_heads 4 --num_epochs 200 --learning_rate 0.001 --input_directory data_save\数据读取的案例数据 --input_size 5 --batch_size 32 --sequence_length 20 --predict_target "DEN"
```

### Parameters:

- `--model_name`: The name of the model to be used (e.g., Transformer_KAN).
- `--hidden_size`: Size of the hidden layers.
- `--num_layers`: Number of layers in the model.
- `--num_heads`: Number of attention heads.
- `--num_epochs`: Number of training epochs.
- `--learning_rate`: Learning rate for the optimizer.
- `--input_directory`: Directory where the input data is stored.
- `--input_size`: Size of the input features.
- `--batch_size`: Batch size for training.
- `--sequence_length`: Length of the input sequences.
- `--predict_target`: The target variable to predict (e.g., "DEN").

## Results

After training, the model outputs the results, which are stored in the `model_save` directory. You can check the prediction results in this directory to evaluate the performance of your model.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue to improve the project.

## License

This project is licensed under the EIAS License. See the [LICENSE](LICENSE) file for more details.

---

Thank you for using **log well prediction**! If you have any questions or need further assistance, please feel free to contact us.
