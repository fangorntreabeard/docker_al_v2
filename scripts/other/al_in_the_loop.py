from urllib import request, parse
import json
import os
from createrandomlabels import make_file
from list_to_cocofile import write_json
from scripts.detection.train import train_api
from scripts.detection.eval import eval

path_labl = '/home/neptun/PycharmProjects/datasets/coco/for_al'

def al():
    # url = 'http://127.0.0.1:5000/active_learning'
    # params = {
    #     'path_to_labels': '/home/neptun/PycharmProjects/datasets/coco/for_al/',
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
    pathtoimg = '/home/neptun/PycharmProjects/datasets/coco/train2017/'
    pathtolabels = '/home/neptun/PycharmProjects/datasets/coco/for_al/'
    add = 1000
    device_rest = 'gpu'
    return train_api(pathtoimg, pathtolabels, add, device_rest)

def mAP():
    # url = 'http://127.0.0.1:5000/eval'
    # params = {
    #     'device': 'gpu',
    #     'path_to_labels_train': '/home/neptun/PycharmProjects/datasets/coco/for_al',
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
    path_to_labels_train = '/home/neptun/PycharmProjects/datasets/coco/for_al'
    path_to_img_train = '/home/neptun/PycharmProjects/datasets/coco/train2017'
    path_to_labels_val = '/home/neptun/PycharmProjects/datasets/coco/labelsval'
    path_to_img_val = '/home/neptun/PycharmProjects/datasets/coco/val2017'
    device_rest = 'gpu'
    return eval(path_to_labels_train, path_to_img_train, path_to_labels_val, path_to_img_val, device_rest)

if __name__ == '__main__':
    L = []
    for i in range(1):
        files_in_labels = os.listdir(path_labl)
        for file in files_in_labels:
            os.remove(os.path.join(path_labl, file))
        make_file(2000)
        f = mAP()
        # print('mAP0', f)
        for kk in range(3):
            step = al()
            write_json(step['data'], kk)
            f = mAP()
            # print(f'mAP1-{kk}', f)
        L.append(f)
    print(L)
