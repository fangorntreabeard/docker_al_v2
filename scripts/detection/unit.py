from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A
import torch
import json
import yaml
import os
from skimage import io


# with open('setting.yaml') as f:
with open('scripts/detection/setting.yaml') as f:
    templates = yaml.safe_load(f)

mean = [0.4914, 0.4822, 0.4465]
std = [0.2470, 0.2435, 0.2616]

data_transform = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean,  std=std)
])


# Object detection
def prepare_items_od(path_to_img, path_to_labels):

    if os.path.isfile(path_to_labels):
        with open(path_to_labels) as f:
            razmetka = json.load(f)

        annotations = razmetka['annotations']
        images = razmetka['images']
    else:
        annotations = []
        images = []
        for file in os.listdir(path_to_labels):
            with open(os.path.join(path_to_labels, file)) as f:
                razmetka = json.load(f)

            annotations = annotations + razmetka['annotations']
            images = images + razmetka['images']
    images = [(row['file_name'], row['id']) for row in images]

    annotations = [(row['bbox'], row['image_id']) for row in annotations]
    images = list(set(images))
    return images, annotations


class Dataset_objdetect(Dataset):
    def __init__(self, path_to_img, images, annotations, transforms):
        self.path_to_img = path_to_img
        self.images = images
        self.annotations = annotations
        self.transforms = transforms
        self.transA = A.Compose([A.Resize(224, 224)], bbox_params=A.BboxParams(format='coco',
                                                                               label_fields=['class_labels']))

    def __getitem__(self, idx):
        # шаг обучения
        imgx = self.images[idx]
        name_file = imgx[0]
        id_file = imgx[1]
        path = os.path.join(self.path_to_img, name_file)

        image = io.imread(path)
        boxs = [x[0] for x in self.annotations if x[1] == id_file]

        num_objs = len(boxs)
        class_labels = ['tag'] * num_objs
        transformed = self.transA(image=image, bboxes=boxs, class_labels=class_labels)
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']

        transformed_bboxes = [[x[0], x[1], x[0]+x[2], x[1]+x[3]] for x in transformed_bboxes]

        boxes = torch.as_tensor(transformed_bboxes, dtype=torch.float32)
        if num_objs == 0:
            labels = torch.zeros((num_objs, ), dtype=torch.int64)
        else:
            labels = torch.ones((num_objs, ), dtype=torch.int64)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.as_tensor(idx, dtype=torch.int64)
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            transformed_image = self.transforms(transformed_image)

        return transformed_image, target, idx



    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    pass