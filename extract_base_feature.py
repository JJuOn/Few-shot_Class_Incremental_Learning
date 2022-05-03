import torch
import pickle

import numpy as np

from models.util import create_model
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image


class miniImageNet(Dataset):
    def __init__(self, base_classes=None, transform=None):
        f = open('data/miniImageNet/all.pickle', 'rb')
        pkl = pickle.load(f)
        self.data = pkl['data']
        self.labels = pkl['labels']
        self.catname2label = pkl['catname2label']
        self.transform = transform
        if base_classes is not None:
            indices = np.isin(self.labels, base_classes)
            self.data = self.data[indices, :, :, :]
            self.labels = np.array(self.labels)[indices]
        f.close()
    
    def __getitem__(self, idx):
        img = self.data[idx, :, :, :]
        if self.transform is None:
            img = transforms.functional.to_tensor(img)
        else:
            img = self.transform(img)
        return img, self.labels[idx]

    def __len__(self):
        return len(self.data)


def main(args):
    n_cls = 20
    ckpt = torch.load(args.model_path)
    opt = ckpt['opt']
    base_classes = list(ckpt['training_classes'].keys())
    label2human= ckpt['label2human']
    print('Base classes:', label2human[:len(base_classes)])
    model = create_model('resnet18', n_cls, opt, vocab=None, dataset='miniImageNet')
    model.load_state_dict(ckpt['model'])
    model = model.to('cuda')
    model.eval()
    mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
    std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
    normalize = transforms.Normalize(mean=mean, std=std)

    transform = transforms.Compose([
                    lambda x: Image.fromarray(x),
                    transforms.RandomCrop(84, padding=8),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    lambda x: np.array(x),
                    transforms.ToTensor(),
                    normalize
                ])

    data = miniImageNet(base_classes=base_classes, transform=transform)
    loader = DataLoader(data, batch_size=64, shuffle=False, num_workers=0)
    feature_dict = {}
    with torch.no_grad():
        for data, label in tqdm(loader):
            data, label = data.cuda(), label.cuda()
            feats, x = model(data, is_feat=True)
            feat = feats[-1]
            for f, l in zip(feat, label):
                l = l.cpu().item()
                if not l in feature_dict:
                    feature_dict[l] = [f.unsqueeze(0)]
                else:
                    feature_dict[l].append(f.unsqueeze(0))
    for k in feature_dict.keys():
        feature_dict[k] = torch.cat(feature_dict[k], dim=0).cpu()
    torch.save(feature_dict, 'dumped/backbones/continual/resnet18/2/base20/base_features.pth')

    
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, default='dumped/backbones/continual/resnet18/2/base20/resnet18_last.pth')
    args = parser.parse_args()
    main(args)