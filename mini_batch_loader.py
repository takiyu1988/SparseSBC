import os, glob
import numpy as np
import cv2
from torch.utils.data import Dataset
import torchvision.datasets.mnist as mnist
from PIL import Image
import torchvision.transforms as transforms

class datset_mnist(Dataset):
    def __init__(self):
        self.root = "./mnist/MNIST/raw/"
        self.train_set = mnist.read_image_file(os.path.join(self.root, 'train-images.idx3-ubyte'))
    
    def __getitem__(self, idx):
        return self.train_set[idx].unsqueeze(0)/255.
 
    def __len__(self):
        return self.train_set.shape[0]

class dataset_general(Dataset):
    def __init__(self):
        self.root = "../BSD68/"
        self.transform = transforms.Compose([
        transforms.RandomCrop(64, padding=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        ])
        self.data_lib = []
        file_lists = glob.glob(self.root+'*.png')
        for file_name in file_lists:
            tmp = Image.open(file_name)
            self.data_lib.append(tmp)
    
    def __getitem__(self, idx):
        return self.transform(self.data_lib[idx])
    
    def __len__(self):
        return len(self.data_lib)


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class Dataset_cifar10(Dataset):
    def __init__(self, _path, GRAY_SCALE=False, _transforms=None, mode='train'):
        self._path = _path
        if _transforms: self._transforms = _transforms
        else:
            if not GRAY_SCALE:
                self._transforms = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
                ])
            else:
                self._transforms = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[.5], std=[.5])
                ])
        if mode =='train':
            tmp_data = [unpickle(os.path.join(self._path, 'data_batch_%s'%(i+1)))[b'data'] for i in range(5)]
            tmp_data = np.vstack(tmp_data)
            self._data = [np.reshape(x,(3,32,32)).transpose((1,2,0)) for x in tmp_data]
        else:
            tmp_data = [unpickle(os.path.join(self._path, 'test_batch'))[b'data']]
            tmp_data = np.vstack(tmp_data)
            self._data = [np.reshape(x, (3, 32, 32)).transpose((1, 2, 0)) for x in tmp_data]
            self._label = [unpickle(os.path.join(self._path, 'test_batch'))[b'labels']]

    def __getitem__(self, index):
        return self._transforms(Image.fromarray(self._data[index]))

    def __len__(self):
        return len(self._data)

class MiniBatchLoader(object):
 
    def __init__(self, train_path, test_path, image_dir_path, crop_size):
 
        self.training_path_infos = self.read_paths(train_path, image_dir_path)
        self.testing_path_infos = self.read_paths(test_path, image_dir_path)
 
        self.crop_size = crop_size
 
    @staticmethod
    def path_label_generator(txt_path, src_path):
        for line in open(txt_path):
            line = line.strip()
            line = line.replace('\\','/')
            src_full_path = os.path.join(src_path, line)
            if os.path.isfile(src_full_path):
                yield src_full_path
 
    @staticmethod
    def count_paths(path):
        c = 0
        for _ in open(path):
            c += 1
        return c
 
    @staticmethod
    def read_paths(txt_path, src_path):
        cs = []
        for pair in MiniBatchLoader.path_label_generator(txt_path, src_path):
            cs.append(pair)
        return cs
 
    def load_training_data(self, indices):
        return self.load_data(self.training_path_infos, indices, augment=True)
 
    def load_testing_data(self, indices):
        return self.load_data(self.testing_path_infos, indices)
 
    def load_data(self, path_infos, indices, augment=False):
        mini_batch_size = len(indices)
        in_channels = 1

        if augment:
            xs = np.zeros((mini_batch_size, in_channels, self.crop_size, self.crop_size)).astype(np.float32)
            
            for i, index in enumerate(indices):
                path = path_infos[index]
                
                img = cv2.imread(path,0)
                if img is None:
                    raise RuntimeError("invalid image: {i}".format(i=path))
                h, w = img.shape

                if np.random.rand() > 0.5:
                    img = np.fliplr(img)

                if np.random.rand() > 0.5:
                    angle = 10*np.random.rand()
                    if np.random.rand() > 0.5:
                        angle *= -1
                    M = cv2.getRotationMatrix2D((w/2,h/2),angle,1)
                    img = cv2.warpAffine(img,M,(w,h))

                rand_range_h = h-self.crop_size
                rand_range_w = w-self.crop_size
                x_offset = np.random.randint(rand_range_w)
                y_offset = np.random.randint(rand_range_h)
                img = img[y_offset:y_offset+self.crop_size, x_offset:x_offset+self.crop_size]
                xs[i, 0, :, :] = (img/255).astype(np.float32)

        elif mini_batch_size == 1:
            for i, index in enumerate(indices):
                path = path_infos[index]
                
                img = cv2.imread(path,0)
                if img is None:
                    raise RuntimeError("invalid image: {i}".format(i=path))

            h, w = img.shape
            xs = np.zeros((mini_batch_size, in_channels, h, w)).astype(np.float32)
            xs[0, 0, :, :] = (img/255).astype(np.float32)

        else:
            raise RuntimeError("mini batch size must be 1 when testing")
 
        return xs


if __name__ =='__main__':
    ss = datset_mnist()
    ss.train_set