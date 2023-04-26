from urllib import request, parse
import json
import os
from createrandomlabels import make_file
from list_to_cocofile import write_json
from scripts.detection.train import train_api
from scripts.detection.eval import eval
import torch

# path_labl = '/home/neptun/PycharmProjects/datasets/coco/labelstrain'
path_to_img_train = '/media/alex/DAtA2/Datasets/coco/my_dataset/train'
path_to_labels_train = '/media/alex/DAtA2/Datasets/coco/for_al'
path_to_img_val = '/media/alex/DAtA2/Datasets/coco/my_dataset/val'
path_to_labels_val = '/media/alex/DAtA2/Datasets/coco/my_dataset/labels_val/val.json'
path_to_img_test = '/media/alex/DAtA2/Datasets/coco/my_dataset/test'
path_to_labels_test = '/media/alex/DAtA2/Datasets/coco/my_dataset/labels_test/test.json'
device_rest = '1'
path_to_json_train = '/media/alex/DAtA2/Datasets/coco/my_dataset/labels_train/train.json'
path_do_dir_model = '../../weight'
pretrain = True


def al(add, path_model=''):
    # url = 'http://127.0.0.1:5000/active_learning'
    # params = {
    #     'path_to_labels': '/home/neptun/PycharmProjects/datasets/coco/labelstrain/',
    #     'path_to_img': '/home/neptun/PycharmProjects/datasets/coco/train2017/',
    #     'add': '1000',
    #     'device': 'gpu'
    # }
    #
    # querystring = parse.urlencode(params)
    #
    # u = request.urlopen(url + '?' + querystring)
    # resp = u.read()
    # out = json.loads(resp.decode('utf-8'))['data']
    # return out
    # pathtoimg = '/home/neptun/PycharmProjects/datasets/coco/train2017/'
    # pathtolabels = '/home/neptun/PycharmProjects/datasets/coco/labelstrain/'
    # path_to_boxes = '/home/neptun/PycharmProjects/datasets/coco/boxes/'
    # path_to_classes = '/home/neptun/PycharmProjects/datasets/coco/classification/'
    # add = 1000
    # save_model = model is None

    return train_api(path_to_img_train, path_to_labels_train,
                     path_to_img_val, path_to_labels_val,
                     add, device_rest, path_model, batch_unlabeled=-1, pretrain=pretrain, save_model=False)

def mAP():
    # url = 'http://127.0.0.1:5000/eval'
    # params = {
    #     'device': 'gpu',
    #     'path_to_labels_train': '/home/neptun/PycharmProjects/datasets/coco/labelstrain',
    #     'path_to_img_train': '/home/neptun/PycharmProjects/datasets/coco/train2017',
    #     'path_to_labels_val': '/home/neptun/PycharmProjects/datasets/coco/labelsval',
    #     'path_to_img_val': '/home/neptun/PycharmProjects/datasets/coco/val2017',
    # }
    #
    # querystring = parse.urlencode(params)
    #
    # u = request.urlopen(url + '?' + querystring)
    # resp = u.read()
    # out = json.loads(resp.decode('utf-8'))['mAP(0.5:0.95)']
    # return out
    save_model = True

    return eval(path_to_img_train, path_to_labels_train,
                path_to_img_val, path_to_labels_val,
                path_to_img_test, path_to_labels_test,
                device_rest, save_model, path_do_dir_model, pretrain)

if __name__ == '__main__':
    L = []
    N_train = len(os.listdir(path_to_img_train))
    n_al = [N_train // 64, N_train // 32, N_train // 16, N_train // 8, N_train // 4, N_train // 2]
    print(n_al)
    for i in range(4):
        files_in_labels = os.listdir(path_to_labels_train)
        for file in files_in_labels:
            os.remove(os.path.join(path_to_labels_train, file))
        make_file(n_al[0],
                  path_to_json_train=path_to_json_train,
                  path_to_out=os.path.join(path_to_labels_train, 'first.json'))
        a = []
        for kk in range(len(n_al)):
            out = mAP()
            f, path_model = out['mAP(0.5:0.95)'], out['model']
            a.append(f)
            print('mAP', n_al[kk], f)
            if kk != len(n_al) - 1:
                step = al(n_al[kk], path_model)
                write_json(step['data'],
                           kk,
                           path_to_out=path_to_labels_train,
                           full_train_json=path_to_json_train)
        print(a)
        L.append(a)
    print(L)
