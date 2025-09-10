#!/bin/bash
# export PATH=$PATH:/home/xiaoy/OneDrive/Desktop/QHESSO/only_train_once_private_research/

cd /c/Users/xiaoy/OneDrive/Desktop/QHESSO/only_train_once_private_research
# sys.path.append('..')
# sys.path.append(/home/xiaoy/OneDrive/Desktop/QHESSO/only_train_once_private_research)
export PYTHONPATH="C:/Users/xiaoy/OneDrive/Desktop/QHESSO/only_train_once_private_research"

# Run Python script1.py in the background
python ./tutorials/test_resnet56.py --sparsity 0.1
python ./tutorials/test_resnet56.py --sparsity 0.3
python ./tutorials/test_resnet56.py --sparsity 0.5
python ./tutorials/test_resnet56.py --sparsity 0.7

echo "All Python scripts have finished running."