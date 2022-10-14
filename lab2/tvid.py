import os
from torch.utils import data
import transforms


class TvidDataset(data.Dataset):

    # TODO Implement the dataset class inherited
    # from `torch.utils.data.Dataset`.
    # tips: Use `transforms`.
    def __init__(self, root, mode) -> None:
        super(TvidDataset).__init__()
        self.root = root.replace('~', '.')
        self.mode = mode
        self.image_paths = []
        self.classes = ['bird', 'car', 'dog', 'lizard', 'turtle']
        self.image_infos = []

        for cls in self.classes:
            gt_file = os.path.join(self.root, cls + '_gt.txt')
            with open(gt_file, 'r') as gt:
                lines = gt.readlines()
                for i, line in enumerate(lines):
                    if (mode == 'train') & (i < 150):
                        items = line.split(' ')
                        id = int(items[0])
                        bbox = [int(x.strip()) for x in items[1:]]
                        path = os.path.join(self.root, cls, '%06d.JPEG' % id),
                        image_info = {'image_path': path[0], 'label': self.classes.index(cls), 'bbox': bbox}
                        self.image_infos.append(image_info)
                    elif (mode == 'test') & (150 <= i < 180):
                        items = line.split(' ')
                        id = int(items[0])
                        bbox = [int(x.strip()) for x in items[1:]]
                        path = os.path.join(self.root, cls, '%06d.JPEG' % id),
                        image_info = {'image_path': path[0], 'label':self.classes.index(cls), 'bbox': bbox}
                        self.image_infos.append(image_info)
                    else:
                        continue
        if self.mode == 'train':
            self.transform = transforms.Compose([
            transforms.LoadImage(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        else:
            self.transform =  transforms.Compose([
            transforms.LoadImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])


    def __getitem__(self, index: int):
        image_info = self.image_infos[index]
        image = image_info['image_path']
        label = image_info['label']
        bbox = image_info['bbox']
        image, bbox =  self.transform(image, bbox)
        return image,[label,bbox]


    def __len__(self):
        return len(self.image_infos)
    ...

    # End of todo


if __name__ == '__main__':

    dataset = TvidDataset(root='~/data/tiny_vid', mode='train')
    import pdb; pdb.set_trace()
