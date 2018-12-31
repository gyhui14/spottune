import torch 
from torch.autograd import Variable                                                                                            
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import time
import argparse
import numpy as np
import json
import collections

import torch.utils.data as data
import pickle
from PIL import Image
from pycocotools.coco import COCO
import os.path

import sys
sys.path.append('../')
from utils import *
from spottune_models import *
import agent_net
from gumbel_softmax import *

parser = argparse.ArgumentParser(description='PyTorch SpotTune')
parser.add_argument('--datadir', default='../data/decathlon-1.0/', help='folder containing data folder')
parser.add_argument('--imdbdir', default='../data/decathlon-1.0/annotations/', help='annotation folder')

args = parser.parse_args()

datasets = [
    ("aircraft", 0),
    ("cifar100", 1),
    ("daimlerpedcls", 2),
    ("dtd", 3),
    ("gtsrb", 4),
    #("imagenet12", 5),
    ("omniglot", 5),
    ("svhn", 6),
    ("ucf101", 7),
    ("vgg-flowers", 8)]

num_classes = [100, 100, 2, 47, 43, 1623, 10, 101, 102]

datasets = collections.OrderedDict(datasets)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def pil_loader(path):
    return Image.open(path).convert('RGB')

class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, target_transform=None, index=None,
            labels=None ,imgs=None,loader=pil_loader,skip_label_indexing=0):
        
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        if index is not None:
            imgs = [imgs[i] for i in index]
        self.imgs = imgs

        if index is not None:
            if skip_label_indexing == 0:
                labels = [labels[i] for i in index]

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index][0]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, self.imgs[index][1]

    def __len__(self):
        return len(self.imgs)

def prepare_data_loaders(dataset_names, data_dir, imdb_dir, shuffle_train=True, index=None):
    val_loaders = []
    val = [0]

    imdb_names_val   = [imdb_dir + '/' + dataset_names[i] + '_test_stripped.json' for i in range(len(dataset_names))]
    imdb_names = [imdb_names_val]

    with open(data_dir + 'decathlon_mean_std.pickle', 'rb') as handle:
        dict_mean_std = pickle.load(handle)
    
    for i in range(len(dataset_names)):
        imgnames_val = []
        for itera1 in val:
            annFile = imdb_names[itera1][i]
            coco = COCO(annFile)
            imgIds = coco.getImgIds()
            annIds = coco.getAnnIds(imgIds=imgIds)
            anno = coco.loadAnns(annIds)
            images = coco.loadImgs(imgIds) 
            timgnames = [img['file_name'] for img in images]
            timgnames_id = [img['id'] for img in images]

            imgnames = []
            for j in range(len(timgnames)):
                imgnames.append((data_dir + '/' + timgnames[j],timgnames_id[j]))

            if itera1 in val:
                imgnames_val += imgnames

        means = dict_mean_std[dataset_names[i] + 'mean']
        stds = dict_mean_std[dataset_names[i] + 'std']

        if dataset_names[i] in ['gtsrb', 'omniglot','svhn']: 
            transform_test = transforms.Compose([
            transforms.Resize(72),
            transforms.CenterCrop(72),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
            ])
        elif dataset_names[i] in ['daimlerpedcls']:
            transform_test = transforms.Compose([
            transforms.Resize(72),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
            ])
        else:
            transform_test = transforms.Compose([
                transforms.Resize(72),
            	transforms.CenterCrop(72),
                transforms.ToTensor(),
                transforms.Normalize(means, stds),
            ])
        
        img_path = data_dir
        valloader = torch.utils.data.DataLoader(ImageFolder(data_dir, transform_test, None, None, None, imgnames_val), batch_size=100, shuffle=False, num_workers=4, pin_memory=True)
        val_loaders.append(valloader) 
    
    return val_loaders 

def test(dataset, val_loader, net, agent, task_id):
    net.eval()
    if agent is not None:
	   agent.eval()

    # Test the model
    with torch.no_grad():
        for idx, (images, image_ids) in enumerate(val_loader):
            if use_cuda:
                images  = images.cuda(async=True)
                images  = Variable(images, volatile=True)

            probs = agent(images)
            action = gumbel_softmax(probs.view(probs.size(0), -1, 2))
            policy = action[:,:,1]
            outputs = net.forward(images, policy)

            _, predicted = torch.max(outputs.data, 1)
	    
            for image_idx, prediction in enumerate(predicted.data.cpu().numpy()):
                res_dict = {}
                res_dict['category_id'] = int(10e6 * (task_id+1) + prediction + 1)
                res_dict['image_id'] = image_ids.data.cpu().numpy()[image_idx]
                results.append(res_dict)

#####################################
# Prepare data loaders
val_loaders = prepare_data_loaders(datasets.keys(), args.datadir, args.imdbdir, True)

results = []
for dataset in datasets.keys():
    print dataset 
    num_class = num_classes[datasets[dataset]]
    task_id = datasets[dataset]
    net, agent = get_net_and_agent('resnet26', num_class, dataset)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net.cuda()
        if agent is not None:
            agent.cuda()
        cudnn.benchmark = True

    if task_id >= 5:
	   task_id += 1
    test(dataset, val_loaders[datasets[dataset]], net, agent, task_id) 
    
f =  "./results.json"
with open(f, 'wb') as fh:
    json.dump(results, fh)     

