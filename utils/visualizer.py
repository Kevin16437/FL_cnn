import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
import time

class Visualizer:
    def __init__(self):
        self.loss_history = []
        self.accuracy_history = []
        self.batch_loss_history = []
        plt.ion()  # 开启交互模式
        
    def plot_sample(self, image, prediction=None, label=None, reshape=True):
        """显示单个MNIST图像样本"""
        if reshape:
            if len(image.shape) == 3:  # CNN格式 (1, 28, 28)
                image = image[0]
            else:  # FC格式 (784,)
                image = image.reshape(28, 28)
                
        plt.figure(figsize=(4, 4))
        plt.imshow(image, cmap='gray')
        if prediction is not None and label is not None:
            plt.title(f'Prediction: {prediction}\nTrue Label: {label}')
        plt.axis('off')
        plt.show()
        
    def plot_training_progress(self, epoch, batch, loss, accuracy=None):
        """实时显示训练进度"""
        self.batch_loss_history.append(loss)
        
        plt.figure(figsize=(12, 4))
        
        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.batch_loss_history, 'b-')
        plt.title(f'Training Loss (Epoch {epoch+1}, Batch {batch})')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        
        # 准确率曲线（如果提供）
        if accuracy is not None:
            plt.subplot(1, 2, 2)
            self.accuracy_history.append(accuracy)
            plt.plot(self.accuracy_history, 'g-')
            plt.title('Test Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
        
        plt.tight_layout()
        plt.show()
        plt.pause(0.1)
        
    def plot_confusion_matrix(self, true_labels, predictions, classes=range(10)):
        """绘制混淆矩阵"""
        cm = np.zeros((10, 10), dtype=int)
        for t, p in zip(true_labels, predictions):
            cm[t][p] += 1
            
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # 在每个格子中显示数值
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
                        
        plt.tight_layout()
        plt.show()
        
    def plot_example_predictions(self, model, X_test, y_test, num_samples=5):
        """显示一些预测示例"""
        # 随机选择样本
        indices = np.random.choice(len(X_test), num_samples, replace=False)
        
        plt.figure(figsize=(2*num_samples, 2))
        for i, idx in enumerate(indices):
            X = X_test[idx:idx+1]
            y_true = y_test[idx]
            
            # 获取预测
            output = model.forward(X)
            y_pred = np.argmax(output)
            
            # 显示图像
            plt.subplot(1, num_samples, i+1)
            if len(X.shape) == 4:  # CNN format
                img = X[0, 0]
            else:  # FC format
                img = X[0].reshape(28, 28)
                
            plt.imshow(img, cmap='gray')
            color = 'green' if y_pred == y_true else 'red'
            plt.title(f'Pred: {y_pred}\nTrue: {y_true}', color=color)
            plt.axis('off')
            
        plt.tight_layout()
        plt.show()