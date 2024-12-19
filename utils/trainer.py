import numpy as np
from utils.visualizer import Visualizer
import time

class Trainer:
    def __init__(self, model, data_loader, learning_rate=0.01, num_epochs=10, batch_size=32):
        self.model = model
        self.data_loader = data_loader
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.visualizer = Visualizer()
        
    def compute_loss(self, output, target):
        batch_size = output.shape[0]
        target_one_hot = np.zeros((batch_size, 10))
        target_one_hot[np.arange(batch_size), target] = 1
        loss = -np.sum(target_one_hot * np.log(output + 1e-7)) / batch_size
        return loss, target_one_hot
            
    def train(self):
        print("开始训练...")
        start_time = time.time()
        
        try:
            for epoch in range(self.num_epochs):
                epoch_start_time = time.time()
                total_loss = 0
                num_batches = 0
                
                # 计算每个epoch需要的批次数
                total_batches = len(range(0, 60000, self.batch_size))
                
                for i in range(0, 60000, self.batch_size):
                    batch_start_time = time.time()
                    
                    # 打印批次进度
                    if num_batches % 10 == 0:
                        print(f"\rEpoch {epoch+1}/{self.num_epochs}, "
                              f"Batch {num_batches}/{total_batches} "
                              f"({(num_batches/total_batches)*100:.1f}%)", end="")
                    
                    # 获取批次数据
                    X_batch, y_batch = self.data_loader.get_batch(self.batch_size)
                    
                    # 前向传播
                    output = self.model.forward(X_batch)
                    
                    # 计算损失
                    loss, target_one_hot = self.compute_loss(output, y_batch)
                    total_loss += loss
                    
                    # 反向传播
                    grad_output = output - target_one_hot
                    self.model.backward(grad_output, self.learning_rate)
                    
                    # 每50个批次显示详细信息
                    if num_batches % 50 == 0:
                        batch_time = time.time() - batch_start_time
                        print(f"\nBatch {num_batches} - Loss: {loss:.4f} "
                              f"Time: {batch_time:.2f}s")
                    
                    num_batches += 1
                
                # 计算并显示epoch统计信息
                epoch_time = time.time() - epoch_start_time
                avg_loss = total_loss / num_batches
                print(f"\nEpoch {epoch+1}/{self.num_epochs} 完成 - "
                      f"平均损失: {avg_loss:.4f} - "
                      f"用时: {epoch_time:.2f}s")
                
                # 评估模型
                if epoch % 1 == 0:  # 每个epoch评估一次
                    accuracy = self.evaluate()
                    print(f"测试集准确率: {accuracy:.4f}")
                
        except KeyboardInterrupt:
            print("\n训练被手动中断")
        except Exception as e:
            print(f"\n训练过程发生错误: {str(e)}")
            raise
        finally:
            total_time = time.time() - start_time
            print(f"\n总训练时间: {total_time:.2f}s")
            
    def evaluate(self, verbose=True):
        """在测试集上评估模型"""
        try:
            print("\n正在评估模型...")
            X_test, y_test = self.data_loader.get_test_data()
            
            # 前向传播
            output = self.model.forward(X_test)
            
            # 计算准确率
            predictions = np.argmax(output, axis=1)
            accuracy = np.mean(predictions == y_test)
            
            if verbose:
                print(f"测试集准确率: {accuracy:.4f}")
            
            return accuracy
            
        except Exception as e:
            print(f"评估过程错误: {str(e)}")
            raise

    # 在train()的最后添加保存模型代码
    try:
        # 训练完成后的模型保存
        self.model.save_model("cnn_model_params.npy")
        print("模型已成功保存为 cnn_model_params.npy")
    except Exception as e:
        print(f"保存模型时发生错误: {str(e)}")
