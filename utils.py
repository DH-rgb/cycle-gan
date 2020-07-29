import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


import random
import os
import glob
from memory_profiler import profile
import pdb

class ImagePool():
    def __init__(self,pool_size=50):
        self.pool_size = pool_size
        self.buffer = []
    
    
    def get_images(self,pre_images):
        return_imgs = []
        for img in pre_images:
            # pdb.set_trace()
            img = torch.unsqueeze(img,0)
            if len(self.buffer) < self.pool_size:
                self.buffer.append(img)
                return_imgs.append(img)
            else:
                if random.randint(0,1)>0.5:
                    i = random.randint(0,self.pool_size-1)
                    tmp = self.buffer[i].clone()
                    self.buffer[i]=img
                    return_imgs.append(tmp)
                else:
                    return_imgs.append(img)
        return torch.cat(return_imgs,dim=0)


class UnalignedDataset(torch.utils.data.Dataset):

    def __init__(self, image_size, is_train):
        super(torch.utils.data.Dataset, self).__init__()

        root_dir = os.path.join('data', 'horse2zebra')
        
        if is_train:
            dir_A = os.path.join(root_dir, 'trainA')
            dir_B = os.path.join(root_dir, 'trainB')
        else:
            dir_A = os.path.join(root_dir, 'testA')
            dir_B = os.path.join(root_dir, 'testB')

        self.image_size = image_size

        self.image_paths_A = self._make_dataset(dir_A)
        self.image_paths_B = self._make_dataset(dir_B)

        self.size_A = len(self.image_paths_A)
        self.size_B = len(self.image_paths_B)
        
        self.transform = self._make_transform(is_train)

    def __getitem__(self, index):
        index_A = index % self.size_A
        path_A = self.image_paths_A[index_A]
        
        # クラスBの画像はランダムに選択
        index_B = random.randint(0, self.size_B - 1)
        path_B = self.image_paths_B[index_B]

        img_A = Image.open(path_A).convert('RGB')
        img_B = Image.open(path_B).convert('RGB')
        
        # データ拡張
        A = self.transform(img_A)
        B = self.transform(img_B)
        
        return {'A': A, 'B': B, 'path_A': path_A, 'path_B': path_B}
    
    def __len__(self):
        return max(self.size_A, self.size_B)

    def _make_dataset(self, dir):
        images = []
        for fname in os.listdir(dir):
            if fname.endswith('.jpg'):
                path = os.path.join(dir, fname)
                images.append(path)
        sorted(images)
        return images

    def _make_transform(self, is_train):
        transforms_list = []
        transforms_list.append(transforms.Resize(int(self.image_size*1.12), Image.BICUBIC))
        transforms_list.append(transforms.RandomCrop(self.image_size))
        if is_train:
            transforms_list.append(transforms.RandomHorizontalFlip())
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))  # [0, 1] => [-1, 1]
        return transforms.Compose(transforms_list)


available_datasets = ['horse2zebra', 'facades']
base_url = 'https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/'

class CycleGANDataset(Dataset):
    """
    Class used to read images in
    """
    def __init__(self, data_root=f"{__file__}/../data", dataset_name='horse2zebra', transform=None, unaligned=True, mode='train'):

        # Check whether the specified dataset is available
        if dataset_name not in available_datasets:
            print(f"{dataset_name}という名前のデータセットが存在しないようです")
            sys.exit(1)
        # Check whether the dataset is downloaded
        base_dir = os.path.abspath(data_root)
        dataset_dir = os.path.join(base_dir, dataset_name)
        if not os.path.exists(dataset_dir):
            if not os.path.exists(base_dir):
                os.makedirs(base_dir, True)
            print(f"{base_url}からデータセットをダウンロードし{base_dir}に解凍してください")
            sys.exit(1)

        self.transform = transform
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(dataset_dir, f"{mode}/A/*.*")))
        self.files_B = sorted(glob.glob(os.path.join(dataset_dir, f"{mode}/B/*.*")))

    def __getitem__(self, index):

        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B)-1)]))
        else:
            item_B = self.transform(Image.open(self.files_A[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))