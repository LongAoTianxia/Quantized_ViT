import torch    
import numpy as np
import json
import mymodel
# 导出PyTorch模型的参数和配置信息，以便在硬件上部署

# 加载量化模型（小型量化网络）
model = mymodel.UltraNetQua()
# model = mymodel.TempNetQua()
# print(model)
# 加载预训练权重（4-bit权重量化、4-bit激活量化）
model.load_state_dict(torch.load('ultranet_4w4a.pt', map_location='cpu', weights_only=False)['model'])
# model.load_state_dict(torch.load('model.pkl', map_location='cpu'))

def generate_config(model, in_shape):  # 配置生成函数
    feature_map_shape = in_shape  # 输入形状 [C, H, W]
    print(in_shape)
    dic = {}  # 配置字典
    # 层计数器
    conv_cnt = 0
    pool_cnt = 0
    linear_cnt = 0
    # cnt = 0
    # 遍历模型所有模块
    for sub_module in model.modules():
        # 卷积层参数
        if type(sub_module).__base__ is torch.nn.Conv2d:
            conv_cur = {}
            conv_cur['in_shape'] = feature_map_shape[:]   # 输入形状
            # 计算输出形状
            feature_map_shape[0] = sub_module.out_channels
            feature_map_shape[1] = (feature_map_shape[1] + 2 * sub_module.padding[0] - sub_module.kernel_size[0]) // sub_module.stride[0] + 1
            feature_map_shape[2] = (feature_map_shape[2] + 2 * sub_module.padding[0] - sub_module.kernel_size[0]) // sub_module.stride[0] + 1
            conv_cur['out_shape'] = feature_map_shape[:]  # 输出形状
            conv_cur['k'] = sub_module.kernel_size[0]     # 卷积核大小
            conv_cur['s'] = sub_module.stride[0]          # 步长
            conv_cur['p'] = sub_module.padding[0]         # 填充
            
            dic['conv_' + str(conv_cnt)] = conv_cur
            
            conv_cnt = conv_cnt + 1
            # cnt = cnt + 1

        # 池化层配置
        elif type(sub_module) is torch.nn.MaxPool2d:
            pool_cur = {}
            pool_cur['in_shape'] = feature_map_shape[:]  # 输入形状
            pool_cur['p'] =  sub_module.kernel_size

            # 计算输出形状
            feature_map_shape[1] = feature_map_shape[1] // sub_module.kernel_size
            feature_map_shape[2] = feature_map_shape[2] // sub_module.kernel_size

            pool_cur['out_shape'] = feature_map_shape[:]  # 输出形状

            dic['pool_' + str(pool_cnt)] = pool_cur  # 池化大小

            pool_cnt = pool_cnt + 1
            # cnt = cnt + 1

        # 全连接层配置
        elif type(sub_module).__base__ is torch.nn.Linear:
            linear_cur = {}
            linear_cur['in_len'] = sub_module.in_features  # 输入维度
            linear_cur['out_len'] = sub_module.out_features  # 输出维度

            dic['linear_' + str(linear_cnt)] = linear_cur
            linear_cnt = linear_cnt + 1
            # cnt = cnt + 1
    
    return dic
#   config.json：
'''
{
    "conv_0": {
        "in_shape": [3, 160, 320],
        "out_shape": [16, 158, 318],
        "k": 3,
        "s": 1,
        "p": 0
    },
    "pool_0": {
        "in_shape": [16, 158, 318],
        "out_shape": [16, 79, 159],
        "p": 2
    },
    "linear_0": {
        "in_len": 1024,
        "out_len": 10
    }
}
'''

def generate_params(model):  # 参数导出函数
    dic = {}  # 参数字典
    cnt = 0  # 参数计数器
    # 遍历模型所有模块
    for sub_module in model.modules():
        # 卷积层参数
        if type(sub_module).__base__ is torch.nn.Conv2d:
            w = sub_module.weight.detach().numpy()
            dic['arr_' + str(cnt)] = w
            cnt = cnt + 1
            # 偏置（如果有）
            if sub_module.bias is not None:
                w = sub_module.bias.detach().numpy()
                dic['arr_' + str(cnt)] = w
                cnt = cnt + 1
        # 全连接层参数
        elif type(sub_module).__base__ is torch.nn.Linear:
            w = sub_module.weight.detach().numpy()
            dic['arr_' + str(cnt)] = w
            cnt = cnt + 1
        # BatchNorm层参数
        elif type(sub_module) is torch.nn.BatchNorm2d or type(sub_module) is torch.nn.BatchNorm1d:
            gamma = sub_module.weight.detach().numpy()  # 缩放因子
            dic['arr_' + str(cnt)] = gamma
            cnt = cnt + 1
            beta = sub_module.bias.detach().numpy()  # 平移因子
            dic['arr_' + str(cnt)] = beta
            cnt = cnt + 1
            mean = sub_module.running_mean.numpy()  # 运行均值
            dic['arr_' + str(cnt)] = mean
            cnt = cnt + 1
            var = sub_module.running_var.numpy()  # 运行方差
            dic['arr_' + str(cnt)] = var
            cnt = cnt + 1
            eps = sub_module.eps  # 数值稳定项
            dic['arr_' + str(cnt)] = eps
            cnt = cnt + 1
    return dic
#    ultranet_4w4a.npz：
'''
arr_0: 卷积层0权重
arr_1: BN层0的gamma
arr_2: BN层0的beta
arr_3: BN层0的mean
arr_4: BN层0的var
arr_5: BN层0的eps
arr_6: 卷积层1权重
...
'''

# 1. 导出模型参数到NPZ文件
dic = generate_params(model)
np.savez('ultranet_4w4a.npz', **dic)   # 保存为NPZ格式, 导出的参数将用于qnn_param_reader.py的整数量化

# 2. 生成模型配置信息
# dic = generate_config(model, [3, 416, 416])       # 标准输入尺寸
dic = generate_config(model, [3, 160, 320])     # 自定义输入尺寸
print(dic)

# 3. 保存配置为JSON文件
json_str = json.dumps(dic, indent=4)
with open('config.json' , 'w') as json_file:
    json_file.write(json_str)


            