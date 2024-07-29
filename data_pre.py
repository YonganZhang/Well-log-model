from tool_for_pre import save_data_loaders, main, get_parameters


def data_pre_process():
    args = get_parameters()
    directory = args.input_directory # 替换为你的目录路径
    target_column = args.predict_target  # 替换为你的目标列名称
    sequence_length = args.sequence_length  # 替换为你的时序数据长度
    batch_size = args.batch_size  # 替换为你的批次大小

    train_loader, val_loader = main(directory, target_column, sequence_length, batch_size)

    save_data_loaders(train_loader, val_loader)

if __name__ == "__main__":
    # 示例用法
    data_pre_process()
