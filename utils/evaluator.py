import numpy as np
from configs.config import Config

class Evaluator:
    def __init__(self, model, data_loader):
        """初始化评估器"""
        self.model = model
        self.data_loader = data_loader
        
    def evaluate(self):
        """评估模型"""
        try:
            X_test, y_test = self.data_loader.get_test_data()
            
            # 前向传播
            output = self.model.forward(X_test)
            
            # 计算准确率
            predictions = np.argmax(output, axis=1)
            accuracy = np.mean(predictions == y_test)
            
            # 计算损失
            loss = -np.mean(np.log(output[range(len(y_test)), y_test.astype(int)] + 1e-7))
            
            return {
                'accuracy': accuracy,
                'loss': loss,
                'predictions': predictions
            }
            
        except Exception as e:
            print(f"评估过程错误: {str(e)}")
            raise