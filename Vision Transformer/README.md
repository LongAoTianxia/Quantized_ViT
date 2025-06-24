4-bit量化的ViT模型
## 代码量化思路
graph TD
    A[原始神经网络模型] --> B[模型结构改造]
    B --> C[替换标准层为量化层]
    C --> D[设置量化参数]
    D --> E[量化感知训练QAT]
    E --> F[模型导出]
    F --> G[参数量化处理]
    G --> H[硬件部署格式转换]
    H --> I[最终量化模型]
    
    B --> B1[Conv2d → Conv2d_Q]
    B --> B2[BatchNorm2d → BatchNorm2d_Q]
    B --> B3[Linear → Linear_Q]
    B --> B4[添加激活量化层]
    
    D --> D1[权重位宽w_bit]
    D --> D2[激活位宽a_bit]
    D --> D3[输入输出位宽]
    
    G --> G1[权重量化到整数]
    G --> G2[BN层参数融合]
    G --> G3[生成查找表]
    
    H --> H1[生成param.h]
    H --> H2[生成config.h]
    H --> H3[内存布局优化]
