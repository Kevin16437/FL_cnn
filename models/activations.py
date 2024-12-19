import numpy as np

class Activations:
    @staticmethod
    def relu(x):
        """ReLU激活函数"""
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        """ReLU导数"""
        return np.where(x > 0, 1, 0)
    
    @staticmethod
    def softmax(x):
        """Softmax激活函数"""
        try:
            # 数值稳定性处理
            x_max = np.max(x, axis=1, keepdims=True)
            exp_x = np.exp(x - x_max)
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        except Exception as e:
            print(f"Softmax计算错误: {str(e)}")
            raise