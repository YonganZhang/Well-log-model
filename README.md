## Project Workflow
Project Workflow
Data Preprocessing:

Run data_pre.py to preprocess the data.
Modify prediction targets in tool_for_pre if needed.
Model Training:

Execute train.py with the specified parameters to train the model.
Model Output
The model's output results will be saved in the model_save directory. You can check the predicted results in this directory.

## Installation

To install the required dependencies, run:

```bash pip install -r requirements.txt

## Train
Usage
1. Data Preprocessing
First, run the data_pre.py script. If you need to change the prediction target, you can do so in the tool_for_pre directory within the parser.add_argument function. By default, the data is stored in the data_save directory. You can test the code with the file named "数据读取的案例数据".
bash python data_pre.py
2. Training the Model
Next, train the model by running the train.py script. You can use the following command in the terminal:

bash python train.py --model_name Transformer_KAN --hidden_size 32 --num_layers 4 --num_heads 4 --num_epochs 200 --learning_rate 0.001 --input_directory data_save\数据读取的案例数据 --input_siz 5 --batch_siz 32 --sequence_length 20 --predict_target “DEN”
