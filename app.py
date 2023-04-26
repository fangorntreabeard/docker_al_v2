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
        device = reqparse.request.args['gpu']
        add = int(reqparse.request.args['add'])
        batch_unlabeled = int(reqparse.request.args['batch_unlabeled'])

        path_to_labels_train = reqparse.request.args['path_to_labels_train']
        path_to_img_train = reqparse.request.args['path_to_img_train']

        path_to_labels_val = reqparse.request.args['path_to_labels_val']
        path_to_img_val = reqparse.request.args['path_to_img_val']

        path_model = reqparse.request.args['path_model']
        pretrain = reqparse.request.args['pretrain']
        save_model = reqparse.request.args['save_model']

        use_val_test = reqparse.request.args['use_val_test_in_train']


        pretrain = True if pretrain == 'T' else False
        save_model = True if save_model == 'T' else False
        path_model = None if path_model == '' else path_model
        use_val_test = True if use_val_test == 'T' else False


        return make_train(path_to_img_train, path_to_labels_train,
                     path_to_img_val, path_to_labels_val,
                     add, device, path_model, batch_unlabeled, pretrain,
                     save_model, use_val_test)


class Evaluate(Resource):
    @staticmethod
    def get():
        device = reqparse.request.args['gpu']
        path_to_labels_train = reqparse.request.args['path_to_labels_train']
        path_to_img_train = reqparse.request.args['path_to_img_train']
        path_to_labels_val = reqparse.request.args['path_to_labels_val']
        path_to_img_val = reqparse.request.args['path_to_img_val']
        path_to_labels_test = reqparse.request.args['path_to_labels_test']
        path_to_img_test = reqparse.request.args['path_to_img_test']

        pretrain = reqparse.request.args['pretrain']
        save_model = reqparse.request.args['save_model']

        pretrain = True if pretrain == 'T' else False
        save_model = True if save_model == 'T' else False

        return make_eval(path_to_img_train, path_to_labels_train, path_to_img_val, path_to_labels_val,
              path_to_labels_test, path_to_img_test, device, pretrain, save_model)


def make_train(path_to_img_train, path_to_labels_train,
                     path_to_img_val, path_to_labels_val,
                     add, device, path_model, batch_unlabeled, pretrain,
                     save_model, use_val_test):
    out = train_api(path_to_img_train, path_to_labels_train,
                     path_to_img_val, path_to_labels_val,
                     add, device, path_model, batch_unlabeled, pretrain,
                     save_model, use_val_test)
    return jsonify(out)


def make_eval(path_to_img_train, path_to_labels_train, path_to_img_val, path_to_labels_val,
              path_to_labels_test, path_to_img_test, device, pretrain, save_model):
    out = eval(path_to_img_train, path_to_labels_train,
                path_to_img_val, path_to_labels_val,
                path_to_img_test, path_to_labels_test, device, save_model,
                pretrain)
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
