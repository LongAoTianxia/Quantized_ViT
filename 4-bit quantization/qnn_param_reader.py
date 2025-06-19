import numpy as np
import quantization

# 从一个.npz文件中读取神经网络参数（包括卷积层、全连接层、BN层的参数），然后对这些参数进行量化处理，
# 并将量化后的参数（整数形式）保存为二进制文件。

# 读出参数
# 然后根据量化规则将其处理量化为指定位宽
# 使用int32类型储存
class QNNParamReader:
    def __init__(self, paramFile):
        self.param_dict = np.load(paramFile)    # 加载NPZ参数文件
        self.current_param_cnt = 0  # 当前参数索引
    
    def get_last(self):
        ret = self.param_dict["arr_" + str(self.current_param_cnt)]
        # self.current_param_cnt += 1
        return ret

    def __get_current(self):
        ret = self.param_dict["arr_" + str(self.current_param_cnt)]
        self.current_param_cnt += 1     # 移动到下一个参数
        return ret

    def read_conv_raw(self):
        w = self.__get_current()    # 读取原始卷积权重
        return w

    def read_linear_raw(self):
        w = self.__get_current()     # 读取原始全连接层权重
        return w

    def read_batch_norm_raw(self):
        gamma = self.__get_current()  # BN缩放参数
        beta = self.__get_current()  # BN平移参数
        mean = self.__get_current()  # 均值
        var = self.__get_current()  # 方差
        eps = self.__get_current()  # 数值稳定项
        return (gamma, beta, mean, var, eps)

    # 读量化后卷积层参数
    # 量化后用 int32 表示每个数据
    # 默认将卷积层位宽参数量化到2个bit
    # 符号位占用一个bit
    def read_qconv_weight(self, w_bit=2):
        w = self.read_conv_raw()
        # 执行 w 量化
        qw = quantization.weight_quantize_int(w, w_bit)
        return qw

    # 读量化后的全连接层的参数
    # 量化后用 int32 表示每个数据 实际有效的只有 w_bit
    # 符号位占用一个 bit
    def read_qlinear_weight(self, w_bit=2):
        w = self.read_linear_raw()
        # 权重量化
        qw = quantization.weight_quantize_int(w, w_bit)
        return qw  # 返回整数量化权重

    # 读取量化后的 bn 层参数
    # 将bn层和act层放在一起处理，量化后其可以用一个等差数列表示
    # 其中inc表示公差， bias表示初始值
    def read_qbarch_norm_act_param(self, w_bit=2, in_bit=4, out_bit=4, l_shift=4):
        gamma, beta, mean, var, eps = self.read_batch_norm_raw()
        # BN层量化融合
        qinc, qbias = quantization.bn_act_quantize_int(gamma, beta, mean, var, eps, w_bit=w_bit, in_bit=in_bit, out_bit=out_bit, l_shift=l_shift)
        return qinc, qbias  # 返回量化后的增量(步长)和偏置
    

if __name__ == "__main__":
    import os
    import sys

    # 1. 创建输出目录
    target_dir_int_param = 'param/int32/'
    if not os.path.exists(target_dir_int_param):
        os.makedirs(target_dir_int_param)

    # 2. 加载预训练模型参数
    qnn_read = QNNParamReader('miniConvNet.npz')      # 测试读出完整的参数，量化成int32后保存

    # 3. 处理4个卷积层及其对应的BN层
    # 网络有4个卷积层，两个全连接层
    # 卷积层和中间的 bn层
    for i in range(4):
        # 3.1 量化卷积权重（2-bit）
        con_w = qnn_read.read_qconv_weight(w_bit=2)
        # 3.2 设置输入位宽（第一层图像原始数据保持较高精度8bit输入，后续中间特征使用较低精度4bit）
        if i == 0:
            in_bit = 8
            print(con_w)
        else:
            in_bit = 4
        # 3.3 量化融合BN层（4-bit输入，4-bit输出）
        qinc, qbias = qnn_read.read_qbarch_norm_act_param(w_bit=2, in_bit=in_bit, out_bit=4, l_shift=0)

        # 3.4 保存量化参数
        # 二进制格式存储参数，优化存储以满足FPGA资源约束：二进制格式的紧凑，直接可加载到硬件加速器
        con_w.tofile(target_dir_int_param + 'conv_' + str(i) + '_w.bin')
        qinc.tofile(target_dir_int_param + 'conv_' + str(i) + '_bn_inc.bin')
        qbias.tofile(target_dir_int_param + 'conv_' + str(i) + '_bn_bias.bin')
    
    # 4. 处理全连接层
    # 4.1 第一个全连接层（带BN）
    linear_w0 = qnn_read.read_qlinear_weight(w_bit=2)
    linear_bn0_inc, linear_bn0_bias = qnn_read.read_qbarch_norm_act_param(w_bit=2, in_bit=4, out_bit=4, l_shift=0)

    linear_w0.tofile(target_dir_int_param + 'linear_0_w' + '.bin')
    linear_bn0_inc.tofile(target_dir_int_param + 'linear_0_bn_inc' + '.bin')
    linear_bn0_bias.tofile(target_dir_int_param + 'linear_0_bn_bias' + '.bin')

    # 4.2 第二个全连接层（无BN）
    linear_w1 = qnn_read.read_qlinear_weight(w_bit=2)
    linear_w1.tofile(target_dir_int_param + 'linear_1_w' + '.bin')
    print('generate parameter succeed')

#param/int32/
#├── conv_0_w.bin          # 第0层卷积权重 (2-bit)
#├── conv_0_bn_inc.bin     # 第0层BN步长
#├── conv_0_bn_bias.bin    # 第0层BN偏置
#├── conv_1_w.bin
#├── conv_1_bn_inc.bin
#├── conv_1_bn_bias.bin
#...
#├── linear_0_w.bin        # 全连接层0权重
#├── linear_0_bn_inc.bin   # 全连接层0BN步长
#├── linear_0_bn_bias.bin  # 全连接层0BN偏置
#└── linear_1_w.bin        # 全连接层1权重


