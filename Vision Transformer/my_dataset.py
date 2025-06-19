from PIL import Image
import torch
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        """
        collate_fn方法的作用是定义数据加载时的批处理操作。
        在PyTorch中，数据加载器（DataLoader）将数据集按照指定的批次大小（batch size）加载
        到模型中进行训练或推断。当数据集中的样本具有不同的大小或形状时，需要对每个批次进行适当的处理，
        以便能够对其进行批处理计算。
        这段代码中，collate_fn方法接收一个批次（batch）的数据样本作为输入，其中每个样本是通过
        __getitem__方法返回的图片和标签。它的主要任务是将这些样本组装成一个批次，并对批次中的图片
        进行堆叠和标签的转换，以便能够输入到神经网络中进行处理。
        具体而言，collate_fn方法通过zip(*batch)将批次中的图片和标签分别组成两个元组images和
        labels。然后使用torch.stack方法将图片堆叠成一个张量，这对于需要输入固定大小的张量的模型
        非常重要。最后，将标签转换为torch.Tensor张量，并将最终的图片张量和标签张量作为结果返回。
        通过自定义collate_fn方法，可以根据数据集的特点和模型的需求来灵活处理数据批次，以提高训练
        效率并满足模型的输入要求。
        """
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
