import torch
# import torchvision.transforms.functional as F
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import random

# def data_augmentation(image):
#     random_numbers = random.sample(range(4), 2)
#     if 0 in random_numbers:
#         image = F.hflip(image)           # 水平翻转
#     if 1 in random_numbers:
#         image = F.vflip(image)           # 垂直翻转
#     if 2 in random_numbers:
#         image = F.rotate(image, 90)
#     if 3 in random_numbers:
#         image = F.adjust_brightness(image, brightness_factor=0.2)              # 调整亮度
#     return image

# augmented_data = torch.stack([data_augmentation(x[i]) for i in range(x.shape[0])])
import torch
import torch.nn.functional as F

# 创建示例向量
# vector1 = torch.tensor([[1.0, 2.0, 3.0],
#                        [10, 20, 30],
#                        [100,200,300]])
# vector2 = torch.tensor([[80, 800, 800]])

# print(vector1.shape)
# print(vector2.shape)

# # 计算余弦相似度
# similarities = F.cosine_similarity(vector1, vector2, dim=1)
# print(similarities)
# _, indices = similarities.topk(1, largest=True, dim=0)
# print(indices)


import torch
import torch.nn.functional as F


tensor1 = torch.randn(2,3,4,4)
tensor2 = tensor1.view(2*4*4,3)
tensor3 = tensor2.view(2,3,4,4)
print(tensor1)
print('*'*10)
print(tensor2)
print('*'*10)
print(tensor3)
# print(tensor1.view(3*3,1).view(3,3))
# tensor2 = torch.tensor([[200, 200, 300]])
# print(tensor1.shape,tensor2.shape)
# # 计算余弦相似度
# similarities = F.cosine_similarity(tensor1, tensor2, dim=1)
# print(similarities.shape)
# print(similarities)
# # 找到最大相似度的索引
# closest_index = torch.argmax(similarities)

# print("与 tensor2 最接近的向量的索引:", closest_index.item())
# import torch
# import torch.nn.functional as F

# # 创建示例张量
# tensor1 = torch.randn(50, 256)
# tensor2 = torch.randn(2, 256)

# # 计算余弦相似度
# similarities = F.cosine_similarity(tensor1, tensor2, dim=1)

# # 找到最大相似度的索引
# closest_index = torch.argmax(similarities)

# print("与 tensor2 最接近的向量的索引:", closest_index.item())




