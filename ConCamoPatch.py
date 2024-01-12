from torchvision import transforms
from ImageNetModels import ImageNetModel
from LossFunctions import UnTargeted
import numpy as np
import argparse
import os

from time import time

from torchvision import transforms
from PIL import *


def pytorch_switch(tensor_image):
    return tensor_image.permute(1, 2, 0)


from CamoPatch import Attack
if __name__ == "__main__":

    load_image = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="0 or 1", type=int, default=0)
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--temp", type=float, default=300.)
    parser.add_argument("--mut", type=float, default=0.3)
    parser.add_argument("--s", type=int, default=40)
    parser.add_argument("--queries", type=int, default=10000)
    parser.add_argument("--li", type=int, default=4)

    parser.add_argument("--image_dir", type=str, help="Image File directory")
    parser.add_argument("--true_label", type=int, help="Number of the correct label of ImageNet inputted image")
    parser.add_argument("--save_directory", type=str, help="Where to store the .npy files with the results")
    args = parser.parse_args()

    model = ImageNetModel(args.model)

    image_dir = args.image_dir
    x_test = load_image(Image.open(image_dir))

    loss = UnTargeted(model, args.true_label, to_pytorch=True)
    x = pytorch_switch(x_test).detach().numpy()
    params = {
        "x": x,
        "eps": args.s**2,
        "n_queries": args.queries,
        "save_directory": args.save_directory + ".npy",
        "c": x.shape[2],
        "h": x.shape[0],
        "w": x.shape[1],
        "N": args.N,
        "update_loc_period": args.li,
        "mut": args.mut,
        "temp": args.temp
    }
    attack = Attack(params)
    attack.optimise(loss)
