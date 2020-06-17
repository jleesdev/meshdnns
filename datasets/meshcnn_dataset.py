import torch.utils.data
import numpy as np
import pickle
import os

import torch
from models.layers.mesh import Mesh

class BaseDataset(torch.utils.data.Dataset):

    def __init__(self, opt):
        self.opt = opt
        self.mean = 0
        self.std = 1
        self.ninput_channels = None
        super(BaseDataset, self).__init__()

    def get_mean_std(self):
        """ Computes Mean and Standard Deviation from Training Data
        If mean/std file doesn't exist, will compute one
        :returns
        mean: N-dimensional mean
        std: N-dimensional standard deviation
        ninput_channels: N
        (here N=5)
        """

        mean_std_cache = os.path.join(self.root, 'mean_std_cache.p')
        if not os.path.isfile(mean_std_cache):
            print('computing mean std from train data...')
            # doesn't run augmentation during m/std computation
            num_aug = self.opt.num_aug
            self.opt.num_aug = 1
            mean, std = np.array(0), np.array(0)
            for i, data in enumerate(self):
                if i % 500 == 0:
                    print('{} of {}'.format(i, self.size))
                features = data['edge_features']
                mean = mean + features.mean(axis=1)
                std = std + features.std(axis=1)
            mean = mean / (i + 1)
            std = std / (i + 1)
            transform_dict = {'mean': mean[:, np.newaxis], 'std': std[:, np.newaxis],
                              'ninput_channels': len(mean)}
            with open(mean_std_cache, 'wb') as f:
                pickle.dump(transform_dict, f)
            print('saved: ', mean_std_cache)
            self.opt.num_aug = num_aug
        # open mean / std from file
        with open(mean_std_cache, 'rb') as f:
            transform_dict = pickle.load(f)
            print('loaded mean / std from cache')
            self.mean = transform_dict['mean']
            self.std = transform_dict['std']
            self.ninput_channels = transform_dict['ninput_channels']

############################################################################
def is_mesh_file(filename):
    return any(filename.endswith(extension) for extension in MESH_EXTENSIONS)

def pad(input_arr, target_length, val=0, dim=1):
    shp = input_arr.shape
    npad = [(0, 0) for _ in range(len(shp))]
    npad[dim] = (0, target_length - shp[dim])
    return np.pad(input_arr, pad_width=npad, mode='constant', constant_values=val)

class ClassificationData(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.root = opt.dataroot
        self.dir = os.path.join(opt.dataroot)
        self.classes, self.class_to_idx = self.find_classes(self.dir)
        print(self.classes, self.class_to_idx)
        self.paths = self.make_dataset_by_class(self.dir, self.class_to_idx, opt.phase)
        self.nclasses = len(self.classes)
        self.size = len(self.paths)
        self.get_mean_std()
        # modify for network later.
        opt.nclasses = self.nclasses
        opt.input_nc = self.ninput_channels

    def __getitem__(self, index):
        path = self.paths[index][0]
        label = self.paths[index][1]
        mesh = Mesh(file=path, opt=self.opt, hold_history=False, export_folder=self.opt.export_folder, label=self.classes[label])
        meta = {'mesh': mesh, 'label': label}
        # get edge features
        edge_features = mesh.extract_features()
        edge_features = pad(edge_features, self.opt.ninput_edges)
        meta['edge_features'] = (edge_features - self.mean) / self.std
        return meta

    def __len__(self):
        return self.size

    # this is when the folders are organized by class...
    @staticmethod
    def find_classes(dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        if '.ipynb_checkpoints' in classes:
            classes.remove('.ipynb_checkpoints')
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    @staticmethod
    def make_dataset_by_class(dir, class_to_idx, phase):
        meshes = []
        dir = os.path.expanduser(dir)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                ## LEE
                #file_num = 0
                for fname in sorted(fnames):
                    if is_mesh_file(fname) and (root.count(phase)==1):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        meshes.append(item)

                        ## LEE
                        #file_num += 1
                        #if file_num == 100 :
                        #   break
        return meshes

############################################################################
def collate_fn(batch):
    """Creates mini-batch tensors
    We should build custom collate_fn rather than using default collate_fn
    """
    meta = {}
    keys = batch[0].keys()
    for key in keys:
        meta.update({key: np.array([d[key] for d in batch])})
    return meta

class DataLoader:
    """multi-threaded data loading"""

    def __init__(self, opt):
        self.opt = opt
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads),
            collate_fn=collate_fn)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data
