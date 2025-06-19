import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from vit_model import vit_base_patch16_224_in21k as create_model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # CIFAR-10图片预处理
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # CIFAR-10是32x32，需要resize到224x224
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 要预测的图片路径 - 请修改为您的测试图片路径
    img_path = "D:/python/PycharmProjects/data/CIFAR-10/test/1.png"  # 修改为实际测试图片路径
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)

    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # 读取类别索引文件
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # 创建模型  类别种数
    model = create_model(num_classes=10, has_logits=False).to(device)  # CIFAR-10有10个类别

    # 加载训练好的权重
    model_weight_path = "./weights/model-best-cifar10.pth"
    assert os.path.exists(model_weight_path), "weights file: '{}' not exist.".format(model_weight_path)

    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)

    print("预测结果:")
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()