class Config:
    # 数据集配置
    DATA_PATH = './data'
    
    # 训练配置
    LEARNING_RATE = 0.01
    NUM_EPOCHS = 10
    BATCH_SIZE = 32
    
    # 模型配置
    INPUT_SIZE = 784  # 28x28
    HIDDEN_SIZE = 128
    NUM_CLASSES = 10