import numpy as np

class CNNNetwork:
    def __init__(self):
        # 卷积层1参数
        self.conv1_filters = np.random.randn(8, 1, 3, 3) * 0.1
        self.conv1_bias = np.zeros(8)
        
        # 卷积层2参数
        self.conv2_filters = np.random.randn(16, 8, 3, 3) * 0.1
        self.conv2_bias = np.zeros(16)
        
        # 全连接层参数
        self.fc1_weights = np.random.randn(16 * 5 * 5, 128) * 0.1
        self.fc1_bias = np.zeros(128)
        self.fc2_weights = np.random.randn(128, 10) * 0.1
        self.fc2_bias = np.zeros(10)
        
        print("CNN网络初始化完成")

    def convolution(self, input_data, filters, bias):
        """执行卷积操作"""
        batch_size, channels, height, width = input_data.shape
        num_filters, _, filter_height, filter_width = filters.shape
        
        # 计算输出维度
        output_height = height - filter_height + 1
        output_width = width - filter_width + 1
        
        # 初始化输出
        output = np.zeros((batch_size, num_filters, output_height, output_width))
        
        # 对每个样本执行卷积
        for b in range(batch_size):
            for f in range(num_filters):
                for i in range(output_height):
                    for j in range(output_width):
                        # 提取当前窗口
                        window = input_data[b, :, i:i+filter_height, j:j+filter_width]
                        # 计算卷积
                        output[b, f, i, j] = np.sum(window * filters[f]) + bias[f]
        
        return output

    def max_pooling(self, input_data, pool_size=2):
        """执行最大池化操作"""
        batch_size, channels, height, width = input_data.shape
        output_height = height // pool_size
        output_width = width // pool_size
        
        output = np.zeros((batch_size, channels, output_height, output_width))
        self.pool_masks = {}  # 使用字典存储池化掩码
    
        for b in range(batch_size):
            for c in range(channels):
                for i in range(output_height):
                    for j in range(output_width):
                        h_start = i * pool_size
                        w_start = j * pool_size
                        h_end = min(h_start + pool_size, height)
                        w_end = min(w_start + pool_size, width)
                        
                        window = input_data[b, c, h_start:h_end, w_start:w_end]
                        if window.size > 0:
                            max_val = np.max(window)
                            output[b, c, i, j] = max_val
                            
                            # 存储最大值位置信息
                            max_idx = np.unravel_index(np.argmax(window), window.shape)
                            key = (b, c, i, j)
                            self.pool_masks[key] = (h_start + max_idx[0], w_start + max_idx[1])
        
        return output

    def relu(self, x):
        """ReLU激活函数"""
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """ReLU导数"""
        return (x > 0).astype(float)

    def softmax(self, x):
        """Softmax激活函数"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        
    def forward(self, X):
        try:
            # 保存中间结果用于反向传播
            self.input = X
            
            # 第一个卷积层
            self.conv1_output = self.convolution(X, self.conv1_filters, self.conv1_bias)
            self.conv1_activated = self.relu(self.conv1_output)
            self.pool1_output = self.max_pooling(self.conv1_activated)
            
            # 第二个卷积层
            self.conv2_output = self.convolution(self.pool1_output, self.conv2_filters, self.conv2_bias)
            self.conv2_activated = self.relu(self.conv2_output)
            self.pool2_output = self.max_pooling(self.conv2_activated)
            
            # 展平
            batch_size = X.shape[0]
            self.flattened = self.pool2_output.reshape(batch_size, -1)
            
            # 全连接层
            self.fc1_output = np.dot(self.flattened, self.fc1_weights) + self.fc1_bias
            self.fc1_activated = self.relu(self.fc1_output)
            self.fc2_output = np.dot(self.fc1_activated, self.fc2_weights) + self.fc2_bias
            
            # Softmax
            self.output = self.softmax(self.fc2_output)
            
            return self.output
            
        except Exception as e:
            print(f"前向传播错误: {str(e)}")
            raise

    def convolution_backward(self, input_data, grad_output, filters):
        """计算卷积层的梯度"""
        batch_size, channels, height, width = input_data.shape
        num_filters, _, filter_height, filter_width = filters.shape
        
        grad_filters = np.zeros_like(filters)
        grad_bias = np.zeros(num_filters)
        
        for b in range(batch_size):
            for f in range(num_filters):
                grad_bias[f] += np.sum(grad_output[b, f])
                for i in range(grad_output.shape[2]):
                    for j in range(grad_output.shape[3]):
                        window = input_data[b, :, i:i+filter_height, j:j+filter_width]
                        grad_filters[f] += window * grad_output[b, f, i, j]
        
        return grad_filters, grad_bias

    def convolution_backward_to_input(self, grad_output, filters, input_shape):
        """计算卷积层对输入的梯度"""
        if isinstance(input_shape, np.ndarray):
            batch_size, channels, height, width = input_shape.shape
        else:
            batch_size, channels, height, width = input_shape
            
        # 初始化输入梯度
        grad_input = np.zeros((batch_size, channels, height, width))
        
        # 获取卷积核的尺寸
        num_filters, _, filter_height, filter_width = filters.shape
        
        # 对每个样本进行处理
        for b in range(batch_size):
            for c in range(channels):
                for h in range(height):
                    for w in range(width):
                        for f in range(num_filters):
                            # 计算该位置对应的输出区域
                            h_start = max(0, h - filter_height + 1)
                            h_end = min(height - filter_height + 1, h + 1)
                            w_start = max(0, w - filter_width + 1)
                            w_end = min(width - filter_width + 1, w + 1)
                            
                            # 计算该位置的梯度贡献
                            for i in range(h_start, h_end):
                                for j in range(w_start, w_end):
                                    h_offset = h - i
                                    w_offset = w - j
                                    grad_input[b, c, h, w] += (
                                        grad_output[b, f, i, j] * 
                                        filters[f, c, h_offset, w_offset]
                                    )
        
        return grad_input

    def max_pooling_backward(self, grad_output, original_input, pool_size=2):
    #"""计算最大池化层的梯度"""
        batch_size, channels, height, width = original_input.shape
        grad_input = np.zeros_like(original_input)
        
        output_height = grad_output.shape[2]
        output_width = grad_output.shape[3]
    
        for b in range(batch_size):
            for c in range(channels):
                for i in range(output_height):
                    for j in range(output_width):
                        h_start = i * pool_size
                        w_start = j * pool_size
                        h_end = min(h_start + pool_size, height)
                        w_end = min(w_start + pool_size, width)
                    
                    window = original_input[b, c, h_start:h_end, w_start:w_end]
                    if window.size > 0:  # 确保窗口非空
                        max_idx = np.unravel_index(np.argmax(window), window.shape)
                        grad_input[b, c, h_start + max_idx[0], w_start + max_idx[1]] = \
                            grad_output[b, c, i, j]
        
        return grad_input

    def backward(self, grad_output, learning_rate):
        try:
            batch_size = grad_output.shape[0]
            
            # FC2层的梯度
            grad_fc2_weights = np.dot(self.fc1_activated.T, grad_output)
            grad_fc2_bias = np.sum(grad_output, axis=0)
            grad_fc1_activated = np.dot(grad_output, self.fc2_weights.T)
            
            # FC1层的梯度
            grad_fc1_output = grad_fc1_activated * self.relu_derivative(self.fc1_output)
            grad_fc1_weights = np.dot(self.flattened.T, grad_fc1_output)
            grad_fc1_bias = np.sum(grad_fc1_output, axis=0)
            grad_flattened = np.dot(grad_fc1_output, self.fc1_weights.T)
            
            # 重塑梯度以匹配池化层输出
            grad_pool2 = grad_flattened.reshape(self.pool2_output.shape)
            
            # 池化层和卷积层2的梯度
            grad_conv2_activated = self.max_pooling_backward(grad_pool2, self.conv2_activated)
            grad_conv2_output = grad_conv2_activated * self.relu_derivative(self.conv2_output)
            grad_conv2_filters, grad_conv2_bias = self.convolution_backward(
                self.pool1_output, grad_conv2_output, self.conv2_filters)
            
            # 池化层和卷积层1的梯度
            grad_pool1 = self.convolution_backward_to_input(
                grad_conv2_output, self.conv2_filters, self.pool1_output)
            grad_conv1_activated = self.max_pooling_backward(grad_pool1, self.conv1_activated)
            grad_conv1_output = grad_conv1_activated * self.relu_derivative(self.conv1_output)
            grad_conv1_filters, grad_conv1_bias = self.convolution_backward(
                self.input, grad_conv1_output, self.conv1_filters)
            
            # 更新参数
            self.conv1_filters -= learning_rate * grad_conv1_filters
            self.conv1_bias -= learning_rate * grad_conv1_bias
            self.conv2_filters -= learning_rate * grad_conv2_filters
            self.conv2_bias -= learning_rate * grad_conv2_bias
            self.fc1_weights -= learning_rate * grad_fc1_weights
            self.fc1_bias -= learning_rate * grad_fc1_bias
            self.fc2_weights -= learning_rate * grad_fc2_weights
            self.fc2_bias -= learning_rate * grad_fc2_bias
            
        except Exception as e:
            print(f"反向传播错误: {str(e)}")
            raise

    def save_model(self, filepath):
        """保存模型参数"""
        model_params = {
            'conv1_filters': self.conv1_filters,
            'conv1_bias': self.conv1_bias,
            'conv2_filters': self.conv2_filters,
            'conv2_bias': self.conv2_bias,
            'fc1_weights': self.fc1_weights,
            'fc1_bias': self.fc1_bias,
            'fc2_weights': self.fc2_weights,
            'fc2_bias': self.fc2_bias
        }
        np.save(filepath, model_params)
        print(f"模型已保存到 {filepath}")

    def load_model(self, filepath):
        """加载模型参数"""
        try:
            model_params = np.load(filepath, allow_pickle=True).item()
            self.conv1_filters = model_params['conv1_filters']
            self.conv1_bias = model_params['conv1_bias']
            self.conv2_filters = model_params['conv2_filters']
            self.conv2_bias = model_params['conv2_bias']
            self.fc1_weights = model_params['fc1_weights']
            self.fc1_bias = model_params['fc1_bias']
            self.fc2_weights = model_params['fc2_weights']
            self.fc2_bias = model_params['fc2_bias']
            print(f"模型已从 {filepath} 加载")
        except Exception as e:
            print(f"加载模型时出错: {str(e)}")
            raise