from urllib import request, parse
import json
from createrandomlabels import make_file
import matplotlib.pyplot as plt
from scripts.detection.eval import  eval

save_model = False


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
    path_to_img_train       = '/media/alex/DAtA2/Datasets/coco/my_dataset/train'
    path_to_labels_train    = '/media/alex/DAtA2/Datasets/coco/labelstrain/first.json'
    path_to_img_val         = '/media/alex/DAtA2/Datasets/coco/my_dataset/val'
    path_to_labels_val      = '/media/alex/DAtA2/Datasets/coco/my_dataset/labels_val/val.json'
    path_to_img_test        = '/media/alex/DAtA2/Datasets/coco/my_dataset/test'
    path_to_labels_test     = '/media/alex/DAtA2/Datasets/coco/my_dataset/labels_test/test.json'
    # path_to_labels_train = '/home/neptun/PycharmProjects/datasets/coco/labelstrain/first.json'
    # path_to_img_train = '/home/neptun/PycharmProjects/datasets/coco/train2017'
    # path_to_labels_val = '/home/neptun/PycharmProjects/datasets/coco/labelsval/val.json'
    # path_to_img_val = '/home/neptun/PycharmProjects/datasets/coco/val2017'
    device_rest = '0'
    path_do_dir_model = '../../weight'
    pretrain = True
    return eval(path_to_img_train, path_to_labels_train,
                path_to_img_val, path_to_labels_val,
                path_to_img_test, path_to_labels_test, save_model, path_do_dir_model,
                device_rest, pretrain)

if __name__ == '__main__':
    p = [1_000, 10_000, 30_000]
    k = 5
    L = []
    for i in p:
        mean = 0
        m = []
        for j in range(k):
            make_file(i)
            f = mAP()['mAP(0.5:0.95)']
            print(f)
            mean += f
            m.append(f)
        L.append(mean/k)
        plt.scatter([i]*k, m)
        print(i, m)

    plt.plot(p, L)
    plt.grid(True)
    plt.show()
    # print(L)