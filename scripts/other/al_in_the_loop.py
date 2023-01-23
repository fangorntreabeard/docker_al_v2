from urllib import request, parse
import json
import os
from createrandomlabels import make_file
from list_to_cocofile import write_json
from scripts.detection.train import train_api
from scripts.detection.eval import eval

path_labl = '/home/neptun/PycharmProjects/datasets/coco/labelstrain'

def al(model=None):
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
    pathtoimg = '/home/neptun/PycharmProjects/datasets/coco/train2017/'
    pathtolabels = '/home/neptun/PycharmProjects/datasets/coco/labelstrain/'
    path_to_boxes = '/home/neptun/PycharmProjects/datasets/coco/boxes/'
    add = 1500
    device_rest = 'gpu'
    return train_api(pathtoimg, pathtolabels, path_to_boxes, add, device_rest, model)

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
    path_to_labels_train = '/home/neptun/PycharmProjects/datasets/coco/labelstrain'
    path_to_img_train = '/home/neptun/PycharmProjects/datasets/coco/train2017'
    path_to_labels_val = '/home/neptun/PycharmProjects/datasets/coco/labelsval'
    path_to_img_val = '/home/neptun/PycharmProjects/datasets/coco/val2017'
    device_rest = 'gpu'
    return eval(path_to_labels_train, path_to_img_train, path_to_labels_val, path_to_img_val, device_rest)

if __name__ == '__main__':
    L = []
    for i in range(4):
        files_in_labels = os.listdir(path_labl)
        for file in files_in_labels:
            os.remove(os.path.join(path_labl, file))
        make_file(1000)
        out = mAP()
        f, model = out['mAP(0.5:0.95)'], out['model']
        print('mAP0', f)
        a = [f, ]
        for kk in range(30):
            step = al(model)
            write_json(step['data'], kk)
            out = mAP()
            f, model = out['mAP(0.5:0.95)'], out['model']
            a.append(f)
            print(f'mAP1-{kk}', f)
        print(a)
        L.append(a)
    print(L)
