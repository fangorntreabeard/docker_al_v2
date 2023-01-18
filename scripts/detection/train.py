import os
from scripts.detection.engine import evaluate
import numpy as np
import torch
import json
import random
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms as t
from scripts.detection.engine import train_one_epoch
# from scripts.detection.eval import eval
from scripts.detection.unit import Dataset_objdetect, prepare_items_od
import copy
import scripts.detection.utils as utils
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import GeneralizedRCNNTransform

# with open('../detection/setting.yaml') as f:
# with open('setting.yaml') as f:
with open('scripts/detection/setting.yaml') as f:
    templates = yaml.safe_load(f)



def get_transform():
    transforms = [t.ToTensor()]
    return t.Compose(transforms)


def train_model(pathtoimg, images, annotations, device, num_epochs=5):
    ds0 = Dataset_objdetect(pathtoimg, images, annotations, transforms=get_transform())
    train_dataloader = DataLoader(ds0, batch_size=8, shuffle=True, collate_fn=utils.collate_fn)
    num_classes = 2
    best_model = None
    best_mape = 0

    model = get_model_instance_segmentation(num_classes)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=1e-4)
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=100)
        # outval = mAP(model)
        # mape = outval['mAP(0.5:0.95)']
        # if best_mape < mape:
        #     best_mape = mape
        best_model = copy.deepcopy(model)

    return best_model


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.DEFAULT')

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
                        dd.append(s)
                    p1 = max(dd)
                    # p2 = 1 - p1
                    # pp = sorted([p1, p2])
                    # ppp = 1 - (pp[1] - pp[0])
                    confidence.append(p1)

            indexs += [x for x in indx]
            values += confidence

    # out_name = []
    out_dict = {k: v for k, v in zip(indexs, values)}
    a = sorted(out_dict.items(), key=lambda x: x[1])

    # temp = []
    p_min = 0.8
    p_max = 1
    pp = [(row[0], row[1]) for row in a if p_min <= row[1] < p_max]
    temp = random.sample(pp, k=min(add, len(pp)))
    # for ii in range(5):
    #     p_min = 0.5 + ii * 0.1
    #     p_max = 0.5 + (ii + 1) * 0.1
    #     pp = [(row[0], row[1]) for row in a if p_min <= row[1] < p_max]
    #     temp = temp + random.sample(pp, k=min(add//5, len(pp)))
    if len(temp) < add:
        temp = temp + random.sample(list(set(a) - set(temp)), k=add - len(temp))

    # temp = a[-add:]
    out_name = [unlabeled_data[k] for k, v in temp]
    return sorted(out_name)


def train_api(pathtoimg, pathtolabels, add, device_rest, model=None):
    if device_rest == 'gpu':
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"

    all_img = os.listdir(pathtoimg)
    images, annotations = prepare_items_od(pathtoimg, pathtolabels)
    if model is None:
        model0 = train_model(pathtoimg, images, annotations, device, num_epochs=templates['n_epoch'])
    else:
        model0 = model
    unlabeled_data = list(set(all_img) - set([x[0] for x in images]))
    unlabeled_data = random.sample(unlabeled_data, k=5000)

    add_to_label_items = sampling_uncertainty(model0, pathtoimg, unlabeled_data, add, device)

    return {'data': add_to_label_items}

def mAP(model):
    path_to_labels_train = '/home/alex/PycharmProjects/dataset/coco/for_al'
    path_to_img_train = '/home/alex/PycharmProjects/dataset/coco/train2017'
    path_to_labels_val = '/home/alex/PycharmProjects/dataset/coco/labelsval'
    path_to_img_val = '/home/alex/PycharmProjects/dataset/coco/val2017'
    device_rest = 'gpu'
    return neval(path_to_labels_train, path_to_img_train, path_to_labels_val, path_to_img_val, device_rest, model)

def neval(path_to_labels_train, path_to_img_train, path_to_labels_val, path_to_img_val, device_rest, model=None):

    if device_rest == 'gpu':
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"

    if model is None:
        images_train, annotations_train = prepare_items_od(path_to_img_train, path_to_labels_train)
        model0 = train_model(path_to_img_train, images_train, annotations_train, device,
                             num_epochs=templates['n_epoch'])
    else:
        model0 = model

    images_test, annotations_test = prepare_items_od(path_to_img_val, path_to_labels_val)
    dataset_test = Dataset_objdetect(path_to_img_val, images_test, annotations_test, get_transform())
    data_loader_test = DataLoader(dataset_test, batch_size=32, shuffle=False, collate_fn=utils.collate_fn)

    coco_evaluator = evaluate(model0, data_loader_test, device=device)
    return {'mAP(0.5:0.95)': _summarize(coco_evaluator.coco_eval['bbox']), 'model': model0}

def _summarize(coco, ap=1, iouThr=None, areaRng='all', maxDets=100):
    p = coco.params
    iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
    titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
    typeStr = '(AP)' if ap == 1 else '(AR)'
    iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
        if iouThr is None else '{:0.2f}'.format(iouThr)

    aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
    mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
    if ap == 1:
        # dimension of precision: [TxRxKxAxM]
        s = coco.eval['precision']
        # IoU
        if iouThr is not None:
            t = np.where(iouThr == p.iouThrs)[0]
            s = s[t]
        s = s[:, :, :, aind, mind]
    else:
        # dimension of recall: [TxKxAxM]
        s = coco.eval['recall']
        if iouThr is not None:
            t = np.where(iouThr == p.iouThrs)[0]
            s = s[t]
        s = s[:, :, aind, mind]
    if len(s[s > -1]) == 0:
        mean_s = -1
    else:
        mean_s = np.mean(s[s > -1])
    # print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
    return mean_s


if __name__ == '__main__':
    path_to_img = '/home/neptun/PycharmProjects/datasets/coco/val2017'
    path_to_labels = '/home/neptun/PycharmProjects/datasets/coco/labels'

    c = train_api(path_to_img, path_to_labels, templates['num_for_al'], 'gpu')

    print(c['data'])
