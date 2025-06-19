import numpy as np 


# 使用derefa量化 将浮点数量化表示
def uniform_quantize(input, bit=2):
    n = float(2 ** bit - 1)
    out = np.round(input * n) / n
    
    return out

# 量化w
# 量化到（-1， 1）之间的特定数值（-1， 0， 1）
def weight_quantize_float(input, bit=2):
    weight = np.tanh(input)
    weight = weight / np.max(np.abs(weight)) 
    # 这里是因为量化的数值限制在（-1，1）
    weight_q = uniform_quantize(weight, bit=bit-1)

    return weight_q

# 量化w
# 量化到 （-2**（bit-1）, + 2**(bit-1))
# 当bit=2时表示的是三元量化，即只能取到值-1， 0， 1
def weight_quantize_int(input, bit=2):
    weight = np.tanh(input) # 软裁剪：公式(5) soft_clip(x,1,1) = tanh(x) 逼近 clip(x,-1,1)
    weight = weight / np.max(np.abs(weight))    # 归一化到[-1,1]
    weight_q = weight * (2**(bit-1) - 1)    # 映射到整数范围： soft_clip(w,a,b) / s  , s = c / (2**(k-1) -1)
    weight_q = np.round(weight_q)  # 取整，q_w = round(soft_clip(w,a,b) / s )
    # print(weight_q)
    weight_q = weight_q.astype(np.int32)  # 转换为 int32, 返回整数量化权重
    return weight_q

# 这里计算出bn层等价的w和b
def bn_act_w_bias_float(gamma, beta, mean, var, eps):
    # 公式(17)BN层融合计算
    # BN 层：
    # w =  gamma / (sqrt(var) + eps)   # 这就是ρ
    # b_b = (b - μ) * ρ + φ
    # 卷积层和BN层融合时，卷积层的输出为o_c，BN层的输出为：
    # o_b = (o_c - μ) / sqrt(var+eps) * γ + β
    #     = (γ / sqrt(var+eps)) * o_c + (β - (γ * μ) / sqrt(var+eps))
    # 融合后的权重为：w = γ / sqrt(var + eps)
    w = gamma / (np.sqrt(var) + eps)
    # 融合后的偏置为：b = β - (γ * μ) / sqrt(var + eps)
    b = beta - (mean / (np.sqrt(var) + eps) * gamma)
    return w, b

# 将bn层与act层放在一起计算得到一个等差数列
# 等差数列的下标表示输入数据激活量化后的输出值
# 例如等差数为[3, 7, 11, 15, 19, 23, 27]
# 输入17应该返回4， 输入3返回0
# 注意特征数据是无符号的，权值参数是有符号的
# w 应该不为0
# l_shift是出于精度考虑将结果乘以一定的倍数保存
# 弃用
# def bn_act_quantize_int(gamma, beta, mean, var, eps, w_bit=2, in_bit=4, out_bit=4, l_shift=4):
#     # 先计算出等价的w和b
#     w, b = bn_act_w_bias_float(gamma, beta, mean, var, eps)
#     inc_f = 1 / w
#     bias_f = b / w
#     inc = inc_f * (2 ** (w_bit - 1) - 1) * (2 ** in_bit - 1) * (2 ** l_shift) / (2 ** out_bit - 1)
#     bias = bias_f * (2 ** (w_bit - 1) - 1) * (2 ** in_bit - 1) * (2 ** l_shift) + inc / 2
#     inc_q = (inc + 0.5).astype(np.int32)
#     bias_q = (bias + 0.5).astype(np.int32)
#     print(inc_q)
#     return inc_q, bias_q

def bn_act_quantize_int(gamma, beta, mean, var, eps, w_bit=2, in_bit=4, out_bit=4, l_shift=4):\
    # gamma, beta, mean, var, eps: BN层的参数
    # w_bit,in_bit,out_bit: 权重量化,输入量化,输出量化的比特数
    # l_shift: 左移位因子(放大因子)
    # 在量化推理中，我们希望整个计算都是整数运算。因此，需要将s_b和b_b也量化。
    # 同时，由于s_b通常是一个小于1的浮点数，直接量化会损失精度，
    # 所以论文提出通过左移位因子（l_shift）将其放大为整数，然后通过右移位（在硬件中实现）来恢复。

    # 计算融合后的浮点权重w和偏置b
    w, b = bn_act_w_bias_float(gamma, beta, mean, var, eps)     # 计算融合参数

    # 计算缩放因子：对应公式(21)-(22) (不符合论文)
    n = 2**(w_bit - 1 + in_bit + l_shift) / ((2 ** (w_bit-1) - 1) * (2 ** in_bit - 1))
    # 计算量化步长：公式(21) q_s=round(2^{l_shift}s_f)   (不符合论文)
    inc_q = (2 ** out_bit - 1) * n * w
    inc_q = np.round(inc_q).astype(np.int32)    # 取整，并转换为int32类型
    # 计算量化偏置：公式(22) q_b=round(2^{l_shift}b_b)   (不符合论文)
    bias_q = (2 ** (w_bit-1) - 1) * (2 ** in_bit - 1) * (2 ** out_bit - 1) * n * b
    bias_q = np.round(bias_q).astype(np.int32)
    print('inc_q: ', inc_q)
    print('bias_q: ', bias_q)
    return inc_q, bias_q




if __name__ == "__main__":
    a = np.array([-0.6, 0.1, -0.2, 0.5, 0.3, 0.8, -3.9])
    # print(a.astype(np.int32))


    # print(weight_quantize_float(a, bit=3))
    print(weight_quantize_int(a, bit=4))
