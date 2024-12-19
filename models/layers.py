import numpy as np

class FCLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros(output_size)
        
    def forward(self, input_data):
        try:
            self.input_data = input_data
            return np.dot(input_data, self.weights) + self.bias
        except Exception as e:
            print(f"全连接层前向传播错误: {str(e)}")
            raise
            
    def backward(self, grad_output, learning_rate):
        try:
            grad_weights = np.dot(self.input_data.T, grad_output)
            grad_bias = np.sum(grad_output, axis=0)
            grad_input = np.dot(grad_output, self.weights.T)
            
            self.weights -= learning_rate * grad_weights
            self.bias -= learning_rate * grad_bias
            
            return grad_input
        except Exception as e:
            print(f"全连接层反向传播错误: {str(e)}")
            raise

class ConvLayer:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # 初始化权重和偏置
        self.weights = np.random.randn(
            out_channels, 
            in_channels, 
            kernel_size, 
            kernel_size
        ) * 0.01
        
        self.bias = np.zeros(out_channels)
        
    def forward(self, input_data):
        try:
            batch_size, channels, height, width = input_data.shape
            
            # 添加padding
            if self.padding > 0:
                input_padded = np.pad(
                    input_data,
                    ((0,0), (0,0), (self.padding,self.padding), (self.padding,self.padding)),
                    mode='constant'
                )
            else:
                input_padded = input_data
                
            # 计算输出维度
            out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
            out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
            
            # 初始化输出
            output = np.zeros((batch_size, self.out_channels, out_height, out_width))
            
            # 执行卷积操作
            for b in range(batch_size):
                for c_out in range(self.out_channels):
                    for h in range(out_height):
                        for w in range(out_width):
                            h_start = h * self.stride
                            h_end = h_start + self.kernel_size
                            w_start = w * self.stride
                            w_end = w_start + self.kernel_size
                            
                            input_slice = input_padded[b, :, h_start:h_end, w_start:w_end]
                            output[b, c_out, h, w] = np.sum(
                                input_slice * self.weights[c_out]
                            ) + self.bias[c_out]
                            
            self.input_data = input_data
            self.input_padded = input_padded
            return output
            
        except Exception as e:
            print(f"卷积层前向传播错误: {str(e)}")
            raise
            
    def backward(self, grad_output, learning_rate):
        try:
            batch_size, _, out_height, out_width = grad_output.shape
            _, _, in_height, in_width = self.input_data.shape
            
            # 初始化梯度
            grad_input = np.zeros_like(self.input_padded)
            grad_weights = np.zeros_like(self.weights)
            grad_bias = np.sum(grad_output, axis=(0,2,3))
            
            # 计算梯度
            for b in range(batch_size):
                for c_out in range(self.out_channels):
                    for h in range(out_height):
                        for w in range(out_width):
                            h_start = h * self.stride
                            h_end = h_start + self.kernel_size
                            w_start = w * self.stride
                            w_end = w_start + self.kernel_size
                            
                            grad_input[b, :, h_start:h_end, w_start:w_end] += \
                                self.weights[c_out] * grad_output[b, c_out, h, w]
                                
                            grad_weights[c_out] += \
                                self.input_padded[b, :, h_start:h_end, w_start:w_end] * \
                                grad_output[b, c_out, h, w]
                                
            # 移除padding的梯度
            if self.padding > 0:
                grad_input = grad_input[:, :, 
                                     self.padding:-self.padding,
                                     self.padding:-self.padding]
            
            # 更新参数
            self.weights -= learning_rate * grad_weights
            self.bias -= learning_rate * grad_bias
            
            return grad_input
            
        except Exception as e:
            print(f"卷积层反向传播错误: {str(e)}")
            raise

class ReLU:
    def forward(self, input_data):
        try:
            self.input_data = input_data
            return np.maximum(0, input_data)
        except Exception as e:
            print(f"ReLU前向传播错误: {str(e)}")
            raise
            
    def backward(self, grad_output, learning_rate=None):
        try:
            grad_input = grad_output.copy()
            grad_input[self.input_data <= 0] = 0
            return grad_input
        except Exception as e:
            print(f"ReLU反向传播错误: {str(e)}")
            raise

class Softmax:
    def forward(self, input_data):
        try:
            exp_scores = np.exp(input_data - np.max(input_data, axis=1, keepdims=True))
            self.output = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            return self.output
        except Exception as e:
            print(f"Softmax前向传播错误: {str(e)}")
            raise
            
    def backward(self, grad_output, learning_rate=None):
        try:
            return grad_output * self.output * (1 - self.output)
        except Exception as e:
            print(f"Softmax反向传播错误: {str(e)}")
            raise

class Flatten:
    def forward(self, input_data):
        try:
            self.input_shape = input_data.shape
            return input_data.reshape(input_data.shape[0], -1)
        except Exception as e:
            print(f"Flatten前向传播错误: {str(e)}")
            raise
            
    def backward(self, grad_output, learning_rate=None):
        try:
            return grad_output.reshape(self.input_shape)
        except Exception as e:
            print(f"Flatten反向传播错误: {str(e)}")
            raise