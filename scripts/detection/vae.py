import os
from autoencoder.PyTorch_VAE.models import *
from autoencoder.PyTorch_VAE.experiment import VAEXperiment
from autoencoder.PyTorch_VAE.run_remote import train_vae
from pytorch_lightning.utilities.seed import seed_everything
from autoencoder.PyTorch_VAE.dataset import VAEDataset
import yaml
import random
import matplotlib.pyplot as plt

# path_yaml = '/scripts/detection/models/vae_celeba.yaml'
path_yaml = '../detection/models/vae_celeba.yaml'

class Feature_vae:
    def __init__(self, device, path_check, path_yaml):
        with open(path_yaml, 'r') as file:
            try:
                config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)

        model = vae_models[config['model_params']['name']](**config['model_params'])
        experiment = VAEXperiment(model,
                                  config['exp_params'])
        experiment.load_from_checkpoint(path_check, vae_model=model, params=config['exp_params'])

        experiment.model.eval().to(device)
        self.model = experiment.model

    def predict(self, x):
        return self.model(x)

    def loss_function(self, *args, **kwargs):
        return self.model.loss_function(*args, **kwargs)



def train_vae_od(device, pathtoimg, images, annotations, max_epochs):
    # path_yaml = '/scripts/detection/models/vae_celeba.yaml'
    path_model_vae = train_vae(path_yaml, pathtoimg, images, annotations, max_epochs)
    current_vae = Feature_vae(device, path_model_vae, path_yaml)
    return current_vae

def find_loss_vae(model_vae, path_to_boxes):
    with open(path_yaml, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    seed_everything(config['exp_params']['manual_seed'], True)
    config['exp_params']['M_N'] = config['exp_params']['kld_weight']
    config["data_params"]['pathtoimg'] = path_to_boxes
    config["data_params"]['images'] = [(x, i) for i, x in enumerate(sorted(os.listdir(path_to_boxes)))]
    config["data_params"]['annotations'] = [(None, i) for i in range(len(os.listdir(path_to_boxes)))]
    dict_name = {i: x for i, x in enumerate(sorted(os.listdir(path_to_boxes)))}


    data = VAEDataset(**config["data_params"])
    data.setup()

    err = []
    ind = []
    train_dataset = data.train_dataloader()
    for x, l in train_dataset:
        args = model_vae.predict(x.to('cuda:0'))
        loss = model_vae.loss_function(*args, **config['exp_params'])
        err = err + loss['Reconstruction_Loss_batch']
        ind = ind + l.tolist()
    out = {k: v for k, v in zip(ind, err)}
    a = sorted(out.items(), key=lambda x: x[1], reverse=False)
    return [(dict_name[i], e) for i, e in a]


def plot_err_vae(err):
    plt.hist([x[1] for x in err], bins=100)
    plt.show()
    a = [err[0], err[-1]] + random.sample(err, k=50)
    a = sorted(a, key=lambda x: x[1], reverse=False)
    print(a)


if __name__ == '__main__':
    pass
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # training_data = ''
    # unlabeled_data = ''
    # path_to_dir = '/home/neptun/PycharmProjects/datasets/coco/train2017'
    # num_sample = 200
    # num_labels = 2
    # dict_id = ''
    # max_epochs = 200
    # get_vae_samples(device, training_data, unlabeled_data, path_to_dir, num_sample, num_labels, dict_id, max_epochs)