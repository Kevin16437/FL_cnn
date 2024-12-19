import numpy as np
from models.layers import FCLayer, ReLU, Softmax

class FCNetwork:
    def __init__(self):
        # 网络架构定义
        self.fc1 = FCLayer(784, 128)
        self.relu1 = ReLU()
        self.fc2 = FCLayer(128, 10)
        self.softmax = Softmax()
        
    def forward(self, input_data):
        try:
            # 前向传播
            self.fc1_output = self.fc1.forward(input_data)
            self.relu1_output = self.relu1.forward(self.fc1_output)
            self.fc2_output = self.fc2.forward(self.relu1_output)
            output = self.softmax.forward(self.fc2_output)
            
            return output
        except Exception as e:
            print(f"FC网络前向传播错误: {str(e)}")
            raise
            
    def backward(self, grad_output, learning_rate):
        try:
            # 反向传播
            grad = self.softmax.backward(grad_output)
            grad = self.fc2.backward(grad, learning_rate)
            grad = self.relu1.backward(grad)
            grad = self.fc1.backward(grad, learning_rate)
            
            return grad
        except Exception as e:
            print(f"FC网络反向传播错误: {str(e)}")
            raise