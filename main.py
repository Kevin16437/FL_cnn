import argparse
from models.fc_network import FCNetwork
from models.cnn_network import CNNNetwork
from utils.data_loader import MNISTLoader
from utils.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description='Train neural network on MNIST dataset')
    parser.add_argument('--model', type=str, default='cnn', choices=['fc', 'cnn'],
                      help='Model type: fc (fully connected) or cnn (convolutional)')
    parser.add_argument('--epochs', type=int, default=5,  # 减少默认epoch数5
                      help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,  # 增加默认batch size
                      help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.01,  # 调整学习率
                      help='Learning rate')
    return parser.parse_args()

def main():
    try:
        # 解析命令行参数
        args = parse_args()
        
        # 初始化数据加载器
        data_loader = MNISTLoader()
        
        # 根据模型类型设置数据格式
        if args.model == 'cnn':
            data_loader.set_mode(is_cnn=True)
            model = CNNNetwork()
        else:
            data_loader.set_mode(is_cnn=False)
            model = FCNetwork()
        
        # 初始化训练器
        trainer = Trainer(
            model, 
            data_loader,
            learning_rate=args.learning_rate,
            num_epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        # 开始训练
        trainer.train()
        
    except Exception as e:
        print(f"程序执行错误: {str(e)}")
        raise

if __name__ == '__main__':
    main()
    