import argparse
import os
import pickle
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset

from tool_for_test import print_log


def load_excel_files(directory):
    """
    读取指定目录下的所有xlsx文件并合并成一个DataFrame
    """
    print("开始读取Excel文件...")
    files = [f for f in os.listdir(directory) if f.endswith('.xlsx')]
    total_files = len(files)
    data_frames = []
    for i, file in enumerate(files):
        progress = (i + 1) / total_files * 100
        print(f"正在读取文件: {file} (全文件读取进度：{progress:.2f}%)")
        df = pd.read_excel(os.path.join(directory, file))
        data_frames.append(df)
    print("Excel文件读取完成。")
    return pd.concat(data_frames, ignore_index=True)


# def create_time_series_nomalized(data, target_column, sequence_length):
#     """
#     将DataFrame转换为时序数据并分别对X和y进行归一化
#     """
#     print("开始转换为时序数据...")
#
#     # 创建保存归一化器的目录
#     save_dir = "data_save/本次数据读取的缓存"
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#
#     # 初始化归一化器
#     scaler_X = MinMaxScaler()
#     scaler_y = MinMaxScaler()
#
#     # 归一化特征数据（X）
#     X_data = data.drop(columns=[target_column])
#     X_scaled = pd.DataFrame(scaler_X.fit_transform(X_data), columns=X_data.columns)
#
#     # 归一化目标数据（y）
#     y_data = data[[target_column]]
#     y_scaled = pd.DataFrame(scaler_y.fit_transform(y_data), columns=[target_column])
#
#     # 保存归一化器
#     joblib.dump(scaler_X, os.path.join(save_dir, 'scaler_X.pkl'))
#     joblib.dump(scaler_y, os.path.join(save_dir, 'scaler_y.pkl'))
#
#     X, y = [], []
#     total_sequences = len(data) - sequence_length
#     for i in range(total_sequences):
#         if i % (round(total_sequences / 200)) == 0:
#             progress = (i + 1) / total_sequences * 100
#             print(f"正在处理第{i + 1}个序列 (全序列处理进度：{progress:.2f}%)")
#         X.append(X_scaled.iloc[i:i + sequence_length].values)
#         y.append(y_scaled.iloc[i + sequence_length - 1].values)
#
#     print("时序数据转换完成。")
#
#     return np.array(X), np.array(y)


def create_time_series(data, target_column, sequence_length):
    """
    将DataFrame转换为时序数据
    """
    print("开始转换为时序数据...")
    X, y = [], []
    total_sequences = len(data) - sequence_length
    for i in range(total_sequences):
        X.append(data.iloc[i:i + sequence_length].drop(target_column, axis=1).values)
        y.append(data.iloc[i + sequence_length - 1][target_column])
    print("时序数据转换完成。")
    return np.array(X), np.array(y)


def split_data(X, y, test_size=0.2, val_size=0.2):
    """
    将数据分为训练集、验证集和测试集
    """
    print("开始划分数据集...")
    # X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size + val_size, random_state=42)
    # X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size / (test_size + val_size), random_state=42)
    # print("数据集划分完成。")
    # return X_train, X_val, X_test, y_train, y_val, y_test

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size + val_size, random_state=42)
    print("数据集划分完成。")
    return X_train, X_val, y_train, y_val


def create_data_loaders(X_train, X_val, y_train, y_val, batch_size=32):
    """
    创建数据加载器
    """
    print("开始创建数据加载器...")
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("数据加载器创建完成。")

    return train_loader, val_loader


def main(directory, target_column, sequence_length, batch_size=32):
    print("开始数据处理流程...")

    # 读取所有xlsx文件并合并
    data_train_val = load_excel_files(os.path.join(directory, "训练集和验证集"))
    # 将数据转换为时序数据
    X_train_val, y_train_val = create_time_series(data_train_val, target_column, sequence_length)

    # 划分训练集、验证集和测试集
    X_train, X_val, y_train, y_val = split_data(X_train_val, y_train_val)

    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(X_train, X_val, y_train, y_val, batch_size)

    # 打印数据集的形状
    print(f'注意：以下为输入情况：')
    print(f'训练集: X={X_train.shape}, y={y_train.shape}')
    print(f'验证集: X={X_val.shape}, y={y_val.shape}')

    print("数据预处理流程完成")

    return train_loader, val_loader


def save_data_loaders(train_loader, val_loader, save_directory="data_save/本次数据读取的缓存"):
    """
    保存数据加载器到指定目录并存储目录路径
    """
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    with open(os.path.join(save_directory, 'train_loader.pkl'), 'wb') as f:
        pickle.dump(train_loader, f)
    with open(os.path.join(save_directory, 'val_loader.pkl'), 'wb') as f:
        pickle.dump(val_loader, f)



def load_data_loaders(args):
    """
    从存储的路径中加载数据加载器
    """
    # 读取存储的目录路径
    save_directory = 'data_save/本次数据读取的缓存'
    with open(os.path.join(save_directory, 'train_loader.pkl'), 'rb') as f:
        train_loader = pickle.load(f)
    with open(os.path.join(save_directory, 'val_loader.pkl'), 'rb') as f:
        val_loader = pickle.load(f)
    with open(os.path.join(save_directory, 'test_loader.pkl'), 'rb') as f:
        test_loader = pickle.load(f)

    print("数据加载器已从目录加载: ", save_directory)
    print_log(f"数据加载器已从目录加载: {save_directory}", args)
    return train_loader, val_loader, test_loader


def parse_int_list(arg):
    return [int(x) for x in arg.split(',')]


def get_parameters(modelname="LSTM"):
    parser = argparse.ArgumentParser(description='训练模型的脚本')
    ## model
    parser.add_argument('--model_name', type=str, default=modelname, help='选择一个：LSTM,TCN,Transformer,Transformer_KAN,BiLSTM,GRU')
    parser.add_argument('--hidden_size', type=int, default=32, help='隐藏层的神经元数量')
    parser.add_argument('--num_layers', type=int, default=4, help='层的数量')
    parser.add_argument('--dropout', type=float, default=0.2, help='丢失概率')

    ## kan
    parser.add_argument('--grid_size', type=int, default=200, help='grid')

    ## TCN
    parser.add_argument('--num_channels', type=parse_int_list, default=[25, 50, 25])
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--args.dropout')

    ## transformer
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--hidden_space', type=int, default=32)

    # training
    parser.add_argument('--num_epochs', type=int, default=50, help='训练的轮数')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='学习率')

    # data
    parser.add_argument('--input_directory', type=str, default=r'data_save\54口井的数据集', help='输入地址')
    parser.add_argument('--predict_target', type=str, default='DEN', help='预测目标')
    parser.add_argument('--input_size', type=int, default=5, help='输入特征的维度')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--output_size', type=int, default=1, help='输出特征的维度')
    parser.add_argument('--sequence_length', type=int, default=20, help='时序数据的长度')
    args = parser.parse_args()

    return args
