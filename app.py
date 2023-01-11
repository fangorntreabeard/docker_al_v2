from flask import Flask, jsonify
from flask_restful import Api, Resource, reqparse
from scripts.detection.train import train_api
from scripts.detection.eval import eval


app = Flask(__name__)
api = Api(app)


class ActiveLearning(Resource):
    @staticmethod
    def get():
        device = reqparse.request.args['device']
        add = int(reqparse.request.args['add'])
        path_to_txt_labels = reqparse.request.args['path_to_labels']
        path_to_dataset_img = reqparse.request.args['path_to_img']
        return make_train(path_to_dataset_img, add, path_to_txt_labels, device)


class Evaluate(Resource):
    @staticmethod
    def get():
        device = reqparse.request.args['device']
        path_to_labels_train = reqparse.request.args['path_to_labels_train']
        path_to_img_train = reqparse.request.args['path_to_img_train']
        path_to_labels_val = reqparse.request.args['path_to_labels_val']
        path_to_img_val = reqparse.request.args['path_to_img_val']
        return make_eval(path_to_labels_train, path_to_img_train, path_to_labels_val, path_to_img_val, device)


def make_train(path_to_dataset_img, add, path_to_txt_labels, device):
    out = train_api(path_to_dataset_img, path_to_txt_labels, add, device)
    return jsonify(out)


def make_eval(path_to_labels_train, path_to_img_train, path_to_labels_val, path_to_img_val, device):
    out = eval(path_to_labels_train, path_to_img_train, path_to_labels_val, path_to_img_val, device)
    return jsonify(out)


api.add_resource(ActiveLearning, '/active_learning')
api.add_resource(Evaluate, '/eval')

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True)
