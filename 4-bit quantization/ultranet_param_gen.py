from qnn_param_reader import QNNParamReader
from qnn_mem_process import QNNLayerMemProcess
import numpy as np
import json
import os
import sys

# 逐层处理UltraNet模型的参数，将量化神经网络参数转换为适合硬件加速器(如FPGA)的内存布局格式
# 并生成两个头文件：param.h和config.h
# param.h: 包含每一层的权重、BN参数的C++数组初始化代码
# config.h: 包含每一层的配置宏（如输入输出尺寸、位宽、PE数量等）

# conv       0      1   2       3   4   5   6   7   8   9   10  11  12  13  14  15
w_bit   =   [4,     4,  4,      4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4]
in_bit  =   [8,     4,  4,      4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4] 
out_bit =   [4,     4,  4,      4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  32] 
# l_shift =   [0,     0,  0,      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
l_shift =   [8,     8,  8,      8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8]
# simd    =   [27,    16, 144,    24, 72, 6,  18, 2,  9,  2,  9,  2,  9,  2,  9]   
# pe      =   [2,     1,  6,      1,  3,  1,  3,  1,  2,  1,  2,  1,  2,  1,  2]
simd    =   [3,     16,  16,     16, 8,  8,  8,  8,  8,  8,  8,  8,  8,  2,  8, 8]   
pe      =   [16,    8,   8,      4,  2,  2,  2,  2,  2,  2,  2,  2,  2,  1,  2, 2]
    


if __name__ == "__main__":

    target_dir_hls_param = 'param/hls/'
    if not os.path.exists(target_dir_hls_param):
        os.makedirs(target_dir_hls_param)

    # 创建输出目录，打开两个输出文件
    hls_param_file = open(target_dir_hls_param + 'param.h', 'w')
    hls_config_file = open(target_dir_hls_param + 'config.h', 'w')

    # 加载配置文件，该文件包含每一层的输入输出尺寸等配置信息
    config_file = open('config.json', 'r', encoding='utf-8')
    config = json.load(config_file)
    # 创建QNNParamReader读取参数文件
    reader = QNNParamReader('ultranet_4w4a.npz')

    # conv_0 - 7
    for i in range(0, 8):
        processer = QNNLayerMemProcess('conv_' + str(i), reader, config, w_bit=w_bit[i], in_bit=in_bit[i], out_bit=out_bit[i], l_shift=l_shift[i], pe=pe[i], simd=simd[i])
        w, inc, bias = processer.conv()
        param_str = processer.layer_param_to_init_str(w, inc, bias)
        config_str = processer.conv_config_str()
        hls_param_file.write(param_str)
        hls_config_file.write(config_str)
    # conv_8 last
    processer = QNNLayerMemProcess('conv_' + str(8), reader, config, w_bit=w_bit[8], in_bit=in_bit[8], out_bit=out_bit[8], l_shift=l_shift[8], pe=pe[8], simd=simd[8])
    w = processer.last_conv()
    param_str = processer.last_layer_param_to_init_str(w)
    config_str = processer.last_conv_config_str()
    hls_param_file.write(param_str)
    hls_config_file.write(config_str)

    # 读取最后一层的偏置
    last_bias = reader.get_last()
    np.save('param/hls/last_bias', last_bias)  # 保存为npy文件
    last_bias.tofile('param/hls/last_bias.bin')  # 保存为二进制文件

    hls_param_file.close()
    hls_config_file.close()