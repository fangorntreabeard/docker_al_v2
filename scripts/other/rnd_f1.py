from urllib import request, parse
import json
from createrandomlabels import make_file
import matplotlib.pyplot as plt

def mAP():
    url = 'http://127.0.0.1:5000/eval'
    params = {
        'device': 'gpu',
        'path_to_labels_train': '/home/neptun/PycharmProjects/datasets/coco/labelstrain',
        'path_to_img_train': '/home/neptun/PycharmProjects/datasets/coco/train2017',
        'path_to_labels_val': '/home/neptun/PycharmProjects/datasets/coco/labelsval',
        'path_to_img_val': '/home/neptun/PycharmProjects/datasets/coco/val2017',
    }

    querystring = parse.urlencode(params)

    u = request.urlopen(url + '?' + querystring)
    resp = u.read()
    out = json.loads(resp.decode('utf-8'))['mAP(0.5:0.95)']
    return out


if __name__ == '__main__':
    p = [20_000, 40_000, 80_000]
    k = 10
    L = []
    for i in p:
        mean = 0
        m = []
        for j in range(k):
            make_file(i)
            f = mAP()
            # print(f)
            mean += f
            m.append(f)
        L.append(mean/k)
        plt.scatter([i]*k, m)
        print(i, m)

    plt.plot(p, L)
    plt.grid(True)
    plt.show()
    # print(L)