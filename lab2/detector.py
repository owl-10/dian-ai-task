import torch.nn as nn
import resnet
import torch
from torchvision import models
# TODO Design the detector.
# tips: Use pretrained `resnet` as backbone.
class Detector(nn.Module):
    def __init__(self, backbone='resnet50',lengths=(2048 * 4 * 4, 2048, 512), num_classes=5):
        super(Detector,self).__init__()
        # self.backbone = getattr(resnet, backbone)(pretrained=True)
        if backbone == 'resnet50':
            self.backbone = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
        self.lengths = lengths
        self.num_classes = num_classes
        self.cls_fc1 = nn.Linear(self.lengths[0], self.lengths[1])
        self.cls_relu1 = nn.ReLU(inplace=True)
        self.cls_fc2 = nn.Linear(self.lengths[1], self.lengths[2])
        self.cls_relu2 = nn.ReLU(inplace=True)
        self.cls_fc3 = nn.Linear(self.lengths[2], self.num_classes)

        self.reg_fc1 = nn.Linear(self.lengths[0], self.lengths[1])
        self.reg_relu1 = nn.ReLU(inplace=True)
        self.reg_fc2 = nn.Linear(self.lengths[1], self.lengths[2])
        self.reg_relu2 = nn.ReLU(inplace=True)
        self.reg_fc3 = nn.Linear(self.lengths[2], 4)


    def forward(self, x):

        x = self.backbone(x)
        x = x.flatten(1)

        cls = self.cls_fc1(x)
        cls = self.cls_relu1(cls)
        cls = self.cls_fc2(cls)
        cls = self.cls_relu2(cls)
        cls = self.cls_fc3(cls)

        bbox = self.reg_fc1(x)
        bbox = self.reg_relu1(bbox)
        bbox = self.reg_fc2(bbox)
        bbox = self.reg_relu2(bbox)
        bbox = self.reg_fc3(bbox)
        return cls, bbox


if __name__ == '__main__':

    x = torch.randn((1,3,128,128))
    net = Detector('resnet50')
    cls, bbox = net(x)

    print(cls.shape, bbox.shape)
...

# End of todo
