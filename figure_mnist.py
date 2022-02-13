import torch
import torchvision.models as models
import torch.nn as nn

import matplotlib.pyplot as plt
figure = plt.figure()
num_of_images = 10
for imgs, targets in train_loader:
    break
for index in range(num_of_images):
    plt.subplot(6, 10, index+1)
    plt.axis('off')
    img = imgs[index, ...]
    output = model(img.reshape(-1, 28*28).to(DEVICE))
    _, predict = torch.max(output, dim=1)
    print(predict)
    plt.imshow(img.numpy().squeeze(), cmap='gray_r')
plt.show()
