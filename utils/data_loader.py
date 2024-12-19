import numpy as np
import os
import gzip
import urllib.request
from configs.config import Config

class MNISTLoader:
    def __init__(self, data_path='./data'):
        """初始化数据加载器，自动下载MNIST数据集"""
        self.data_path = data_path
        os.makedirs(data_path, exist_ok=True)
        
        # MNIST数据集URL
        self.urls = {
            'train_images': 'https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
            'train_labels': 'https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz',
            'test_images': 'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz',
            'test_labels': 'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz'
        }
        
        self.is_cnn = False  # 添加标志来区分CNN和FC网络
        
        # 下载并加载数据
        try:
            self._download_data()
            self._load_data()
            print(f"数据集加载完成：")
            print(f"训练集大小: {len(self.X_train)}")
            print(f"测试集大小: {len(self.X_test)}")
        except Exception as e:
            print(f"数据加载错误: {str(e)}")
            raise

    def _download_data(self):
        """下载MNIST数据集"""
        for name, url in self.urls.items():
            filename = os.path.join(self.data_path, f"{name}.gz")
            if not os.path.exists(filename):
                print(f"下载 {name} 数据...")
                urllib.request.urlretrieve(url, filename)
                
    def _load_data(self):
        """加载MNIST数据"""
        # 加载训练图像
        with gzip.open(os.path.join(self.data_path, 'train_images.gz'), 'rb') as f:
            self.X_train = np.frombuffer(f.read(), np.uint8, offset=16)
            self.X_train = self.X_train.reshape(-1, 784).astype(np.float32) / 255.0
        
        # 加载训练标签
        with gzip.open(os.path.join(self.data_path, 'train_labels.gz'), 'rb') as f:
            self.y_train = np.frombuffer(f.read(), np.uint8, offset=8)
            
        # 加载测试图像
        with gzip.open(os.path.join(self.data_path, 'test_images.gz'), 'rb') as f:
            self.X_test = np.frombuffer(f.read(), np.uint8, offset=16)
            self.X_test = self.X_test.reshape(-1, 784).astype(np.float32) / 255.0
            
        # 加载测试标签
        with gzip.open(os.path.join(self.data_path, 'test_labels.gz'), 'rb') as f:
            self.y_test = np.frombuffer(f.read(), np.uint8, offset=8)

    def set_mode(self, is_cnn=False):
        """设置数据加载模式"""
        self.is_cnn = is_cnn
        if is_cnn:
            # 重塑数据为CNN格式 (samples, channels, height, width)
            self.X_train = self.X_train.reshape(-1, 1, 28, 28)
            self.X_test = self.X_test.reshape(-1, 1, 28, 28)
        else:
            # 重塑数据为FC格式 (samples, features)
            self.X_train = self.X_train.reshape(-1, 784)
            self.X_test = self.X_test.reshape(-1, 784)
            
    def get_batch(self, batch_size):
        """获取随机批次数据"""
        try:
            indices = np.random.randint(0, len(self.X_train), batch_size)
            return self.X_train[indices], self.y_train[indices]
        except Exception as e:
            print(f"批次数据生成错误: {str(e)}")
            raise
            
    def get_test_data(self):
        """获取测试数据"""
        return self.X_test, self.y_test