from flask import Flask, jsonify
from flask_restful import Api, Resource, reqparse
from scripts.detection.train import train_api
from scripts.detection.eval import eval


app = Flask(__name__)
api = Api(app)


class ActiveLearning(Resource):
    @staticmethod
    def get():
        print('start al')
        device = reqparse.request.args['device']
        add = int(reqparse.request.args['add'])
        batch_unlabeled = int(reqparse.request.args['batch_unlabeled'])
        path_to_labels_train = reqparse.request.args['path_to_labels_train']
        path_to_img_train = reqparse.request.args['path_to_img_train']
        path_to_labels_val = reqparse.request.args['path_to_labels_val']
        path_to_img_val = reqparse.request.args['path_to_img_val']

        return make_train(path_to_img_train, path_to_labels_train, path_to_img_val, path_to_labels_val, add,
                          batch_unlabeled, device)


class Evaluate(Resource):
    @staticmethod
    def get():
        device = reqparse.request.args['device']
        path_to_labels_train = reqparse.request.args['path_to_labels_train']
        path_to_img_train = reqparse.request.args['path_to_img_train']
        path_to_labels_val = reqparse.request.args['path_to_labels_val']
        path_to_img_val = reqparse.request.args['path_to_img_val']
        path_to_labels_test = reqparse.request.args['path_to_labels_test']
        path_to_img_test = reqparse.request.args['path_to_img_test']
        return make_eval(path_to_img_train, path_to_labels_train,
                         path_to_img_val, path_to_labels_val,
                         path_to_labels_test, path_to_img_test,
                         device)


def make_train(path_to_img_train, path_to_labels_train, path_to_img_val, path_to_labels_val, add, batch_unlabeled,
               device):
    out = train_api(path_to_img_train, path_to_labels_train, path_to_img_val, path_to_labels_val, add,
                    batch_unlabeled, device)
    return jsonify(out)


def make_eval(path_to_img_train, path_to_labels_train, path_to_img_val, path_to_labels_val,
              path_to_labels_test, path_to_img_test, device):
    out = eval(path_to_img_train, path_to_labels_train, path_to_img_val, path_to_labels_val,
               path_to_labels_test, path_to_img_test, device)
    return jsonify(out)


api.add_resource(ActiveLearning, '/active_learning')
api.add_resource(Evaluate, '/eval')

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True)
    # device = 'gpu'
    # add = 10000
    # path_to_labels_train = '/home/neptun/PycharmProjects/datasets/coco/labelstrain'
    # path_to_img_train = '/home/neptun/PycharmProjects/datasets/coco/train2017'
    # path_to_labels_val = '/home/neptun/PycharmProjects/datasets/coco/labelsval'
    # path_to_img_val = '/home/neptun/PycharmProjects/datasets/coco/val2017'
    #
    # eval(path_to_labels_train, path_to_img_train, path_to_labels_val, path_to_img_val, device)
