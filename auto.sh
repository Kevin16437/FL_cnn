#!/bin/bash

# 创建项目主目录
mkdir -p handwriting_recognition
cd handwriting_recognition

# 创建项目结构
mkdir -p {data,models,utils,configs,results,tests}

# 创建源代码文件
touch models/{__init__.py,fc_network.py,cnn_network.py,layers.py,activations.py}
touch utils/{__init__.py,data_loader.py,trainer.py,evaluator.py}
touch configs/{__init__.py,config.py}
touch main.py
touch requirements.txt
touch README.md

# 将这些命令保存为 auto.sh

