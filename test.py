import argparse
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
import torch
from utils import generate_number_dict, show_number_dict
import numpy as np
import os

def args_init():
    parser = argparse.ArgumentParser(description="VAES")
    parser.add_argument("--VAE", action="store_true")
    parser.add_argument("--CVAE", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--img_size", type=int, default=28*28)
    parser.add_argument("--class_num", type=int, default=10)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--z_dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lr_decay", type=float, default=0.5)
    parser.add_argument("--sample_num", type=int, default=64, help="generated sample num")
    parser.add_argument("--eval_iter_num", type=int, default=1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--model_save_path", type=str, default="output/models/")
    parser.add_argument("--random_samples_save_path", type=str, default="output/random_samples/")
    parser.add_argument("--similar_samples_save_path", type=str, default="output/similar_samples/")
    parser.add_argument("--interpolation_samples_save_path", type=str, default="output/interpolation_samples/")
    parser.add_argument("--exit_threshold", type=int, default=3)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = args_init()
    args.device = ('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = datasets.MNIST(root='./data/',
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)
    test_dataset = datasets.MNIST(root='./data/',
                                  train=False,
                                  transform=transforms.ToTensor())
    labels = [train_dataset[i][1] for i in range(len(train_dataset))]
    ss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    train_indices, val_indices = list(ss.split(np.array(labels)[:, np.newaxis], labels))[0]
    val_dataset = torch.utils.data.Subset(train_dataset, val_indices)
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    train_iter = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_iter = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_iter = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    number_dict = generate_number_dict(train_iter)
    if args.train:
        # show_number_dict(number_dict)
        if not os.path.exists(args.model_save_path):
            os.makedirs(args.model_save_path)
        if args.VAE:
            args.model_save_path = args.model_save_path + 'vae_model.pkl'
            from models.VAE import VAE, model_train
            model = VAE(args).to(args.device)
            model = model_train(model, train_iter, val_iter, args)
        elif args.CVAE:
            args.model_save_path = args.model_save_path + 'cvae_model.pkl'
            from models.CVAE import CVAE, model_train
            model = CVAE(args).to(args.device)
            model = model_train(model, train_iter, val_iter, args)
        else:
            print("Illegal input")
    elif args.test:
        if not os.path.exists(args.random_samples_save_path):
            os.makedirs(args.random_samples_save_path)
        if not os.path.exists(args.similar_samples_save_path):
            os.mkdir(args.similar_samples_save_path)
        if not os.path.exists(args.interpolation_samples_save_path):
            os.mkdir(args.interpolation_samples_save_path)
        if args.VAE:
            args.model_save_path = args.model_save_path + 'vae_model.pkl'
            args.random_samples_save_path += 'vae_img.jpg'
            args.similar_samples_save_path += 'vae_img.jpg'
            args.interpolation_samples_save_path += 'vae_img.jpg'
            model = torch.load(args.model_save_path)
            model.generate_samples(number_dict[0], number_dict[1], args)
        elif args.CVAE:
            args.model_save_path = args.model_save_path + 'cvae_model.pkl'
            args.random_samples_save_path += 'cvae_img.jpg'
            args.similar_samples_save_path += 'cvae_img.jpg'
            args.interpolation_samples_save_path += 'cvae_img.jpg'
            model = torch.load(args.model_save_path)
            model.generate_samples(0, 1, args)
    else:
        print("Illegal input")
    #model_test(model, test_iter)