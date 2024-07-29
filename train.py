import torch
import torch.nn as nn
import torch.optim as optim
from data_pre import data_pre_process
from model_BiLSTM import BiLSTM
from model_GRU import GRU
from model_LSTM import LSTM
from model_TCN import TemporalConvNet
from model_Trans_KAN import TimeSeriesTransformer_ekan
from model_Transformer import TransformerModel
from tool_for_test import print_log
from tool_for_pre import get_parameters, load_data_loaders
from tool_for_train import train_model


def train(args):
    # 定义模型
    if args.model_name == 'GRU':
        model = GRU(input_dim=args.input_size, hidden_dim=args.hidden_size, num_layers=args.num_layers, output_dim=args.output_size)
    elif args.model_name == 'LSTM':
        model = LSTM(args.input_size, args.hidden_size, args.num_layers, args.output_size)
    elif args.model_name == 'BiLSTM':
        model = BiLSTM(input_dim=args.input_size, hidden_dim=args.hidden_size, num_layers=args.num_layers, output_dim=args.output_size)
    elif args.model_name == 'TCN':
        model = TemporalConvNet(num_inputs=args.input_size, num_outputs=args.output_size, num_channels=args.num_channels, kernel_size=args.kernel_size, dropout=args.dropout)
    elif args.model_name == 'Transformer':
        model = TransformerModel(args.input_size, args.hidden_size, args.num_layers, args.output_size)
    elif args.model_name == 'Transformer_KAN':
        model = TimeSeriesTransformer_ekan(input_dim=args.input_size, num_heads=args.num_heads, num_layers=args.num_layers, num_outputs=args.output_size, hidden_space=args.hidden_space, dropout_rate=args.dropout)
    else:
        print('please choose correct model name')
        model = LSTM(args.input_size, args.hidden_size, args.num_layers, args.output_size)

    # 使用GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_log(f"The device being used is: {device}",args)
    model = model.to(device)
    # 读取数据
    train_loader, val_loader, _ = load_data_loaders(args)
    # 定义损失函数和优化器
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 训练模型
    model_file_path = train_model(model, train_loader, val_loader, criterion, optimizer, args.num_epochs, device, args)

    return model_file_path


if __name__ == "__main__":
    # 获取参数
    args = get_parameters("Transformer_KAN")

    # 数据预处理（初次运行即可，运行后结果保存到data_save文件夹内）
    # data_pre_process()

    # 训练模型（运行后保存到model_save文件夹内）
    model_file_path = train(args)

    # 测试模型
    # test(args, model_file_path)
    # cd /mycode && python train.py --model_name Transformer_KAN
