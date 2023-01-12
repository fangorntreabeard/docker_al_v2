import os

import torch
import json
import random
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms as t
from scripts.detection.engine import train_one_epoch

from scripts.detection.unit import Dataset_objdetect, prepare_items_od
import copy
import scripts.detection.utils as utils
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import GeneralizedRCNNTransform

with open('../detection/setting.yaml') as f:
# with open('setting.yaml') as f:
# with open('scripts/detection/setting.yaml') as f:
    templates = yaml.safe_load(f)



def get_transform():
    transforms = [t.ToTensor()]
    return t.Compose(transforms)


def train_model(pathtoimg, images, annotations, device, num_epochs=5):
    ds0 = Dataset_objdetect(pathtoimg, images, annotations, transforms=get_transform())
    train_dataloader = DataLoader(ds0, batch_size=8, shuffle=True, collate_fn=utils.collate_fn)
    num_classes = 2

    model = get_model_instance_segmentation(num_classes)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=1e-4)
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=100)

    return copy.deepcopy(model)


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.transform = GeneralizedRCNNTransform(min_size=224, max_size=224, image_mean=[0.485, 0.456, 0.406],
                                               image_std=[0.229, 0.224, 0.225])

    return model


def sampling_uncertainty(model, pathtoimg, unlabeled_data, add, device):
    with torch.no_grad():
        model.eval()
        dataset_train = Dataset_objdetect(pathtoimg, unlabeled_data, annotations=None, transforms=get_transform())
        train_dataloader = DataLoader(dataset_train, batch_size=32, shuffle=False, collate_fn=utils.collate_fn)
        indexs = []
        values = []
        for images, _, indx in train_dataloader:
            images = list(img.to(device) for img in images)
            outputs = model(images)
            prob = [x['scores'].tolist() for x in outputs]
            confidence = []
            for b_row in prob:
                if len(b_row) == 0:
                    confidence.append(0)
                else:
                    dd = []
                    for s in b_row:
                        a = s
                        k = (1 - a)
                        dd.append(k)
                    confidence.append(max(dd))

            indexs += [x for x in indx]
            values += confidence

    # out_name = []
    out_dict = {k: v for k, v in zip(indexs, values)}
    a = sorted(out_dict.items(), key=lambda x: x[1])

    temp = a[-add:]
    out_name = [unlabeled_data[k] for k, v in temp]
    return sorted(out_name)
    # alfa = 0.8
    # temp = sorted(values)[-int(add * alfa): ]
    # for ell in temp:
    #     indx = values.index(ell)
    #     out_name.append(indexs[indx])
    #
    # temp = sorted(values)[ :-int(add * alfa)]
    # temp = random.sample(temp, k=int(add * (1 - alfa)))
    # for ell in temp:
    #     indx = values.index(ell)
    #     out_name.append(indexs[indx])
    # return [unlabeled_data[i] for i in out_name]


def train_api(pathtoimg, pathtolabels, add, device_rest):
    if device_rest == 'gpu':
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"

    all_img = os.listdir(pathtoimg)
    images, annotations = prepare_items_od(pathtoimg, pathtolabels)

    model0 = train_model(pathtoimg, images, annotations, device, num_epochs=templates['n_epoch'])
    unlabeled_data = list(set(all_img) - set([x[0] for x in images]))
    # unlabeled_data = random.sample(unlabeled_data, k=30_000)

    add_to_label_items = sampling_uncertainty(model0, pathtoimg, unlabeled_data, add, device)

    return {'data': add_to_label_items}


if __name__ == '__main__':
    path_to_img = '/home/neptun/PycharmProjects/datasets/coco/val2017'
    path_to_labels = '/home/neptun/PycharmProjects/datasets/coco/labels'

    c = train_api(path_to_img, path_to_labels, templates['num_for_al'], 'gpu')

    print(c['data'])
