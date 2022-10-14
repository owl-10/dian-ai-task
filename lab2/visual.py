from PIL import Image

import torch
import torch.nn as nn

from detector import Detector
import transforms
import numpy as np
import cv2
import random
#加载模型
classes = ['bird', 'car', 'dog', 'lizard', 'turtle']
state_dict = torch.load('model.pth')
model = Detector(backbone='resnet50', lengths=(2048 * 4 * 4, 2048, 512), num_classes=5)
model.load_state_dict(state_dict=state_dict)
model.eval()

#加载图片，随机给一个bbox好用来过transform
image_path = './data/tiny_vid/car/000002.JPEG'
bbox = np.random.random((1,4))

#图像预处理
transform =  transforms.Compose([
            transforms.LoadImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
image, bbox = transform(image_path, bbox)
image = image.unsqueeze(0)

#预测结果
cls, bbox_pred = model(image)

cls_pred = classes[torch.argmax(cls).item()]
bbox_pred = (bbox_pred.detach().numpy()* 128).astype(np.int)[0]


#可视化
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(5)]

img = cv2.imread(image_path)

cv2.rectangle(img, (bbox_pred[0],bbox_pred[1]), (bbox_pred[2], bbox_pred[3]), colors[torch.argmax(cls).item()], 2)
cv2.putText(img, cls_pred, (bbox_pred[0],bbox_pred[1]-5), 0, 0.5, color=colors[torch.argmax(cls).item()], thickness=2)
cv2.imwrite('visual2.png', img)