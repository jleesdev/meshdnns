import argparse
import glob
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm
from psbody.mesh import Mesh
import os
import sys
sys.path.insert(1, './utils/coma')
from utils import get_vert_connectivity
from transform_clsf import Normalize

class ComaDataset(InMemoryDataset):
    def __init__(self, data_path, dtype='train', split='clsf', transform=Normalize(), pre_transform=None):
        self.data_path = data_path
        self.dtype = dtype
        self.split = split
        self.transform = transform
        self.pre_tranform = pre_transform
        # self.processed_dir = './processed/coma/'
        # Downloaded data is present in following format root_dir/*/*/*.py
        self.filepaths, self.categories = self.gather_paths()
        super(ComaDataset, self).__init__(data_path, transform, pre_transform)

        if dtype == 'train':
            data_path = self.processed_paths[0]
        elif dtype == 'test':
            data_path = self.processed_paths[1]
        else:
            raise Exception("train and test are supported data types")

        norm_path = self.processed_paths[2]
        norm_dict = torch.load(norm_path)
        self.mean, self.std = norm_dict['mean'], norm_dict['std']
        self.data, self.slices = torch.load(data_path)

        if self.transform is not None:
            if hasattr(self.transform, 'mean') and hasattr(self.transform, 'std'):
                if self.tranform.mean is None:
                    self.tranform.mean = self.mean
                if self.transform.std is None:
                    self.tranform.std = self.std
            self.data = [self.transform(td) for td in self.data]

    @property
    def processed_dir(self):
        return os.path.join('./', 'processed/coma')
    
    @property
    def raw_file_names(self):
        all_fps = []
        for key in self.filepaths.keys() :
            all_fps += self.filepaths[key]
        return all_fps

    @property
    def processed_file_names(self):
        processed_files = ['training.pt', 'test.pt', 'norm.pt']
        processed_files = [self.split+'_'+pf for pf in processed_files]
        return processed_files

    def gather_paths(self):
        filepaths = dict()

        df = pd.read_csv(self.data_path)
        fps = df['mesh_fsl']
        dxs = df['dx']
        categories = sorted(set(dxs))
        for i in range(len(df)) :
            if dxs[i] not in filepaths.keys() :
                filepaths[dxs[i]] = []
                filepaths[dxs[i]].append(fps[i])
            else :
                filepaths[dxs[i]].append(fps[i])

        print(self.dtype)
        print('total number of dataset: %d'%(len(df)))
        for key in filepaths.keys() :
            print('%s: %d.'%(key, len(filepaths[key])))
        return filepaths, categories

    def process(self):
        dataset = []
        vertices = []
        for dx in self.filepaths :
            for fp in self.filepaths[dx] :
                mesh = Mesh(filename=fp)
                mesh_verts = torch.Tensor(mesh.v)
                adjacency = get_vert_connectivity(mesh.v, mesh.f).tocoo()
                edge_index = torch.Tensor(np.vstack((adjacency.row, adjacency.col)))

                i = self.categories.index(dx)
                label = np.zeros(len(self.categories))
                label[i] = 1
                data = Data(x=mesh_verts, y=torch.Tensor(label), edge_index=edge_index)

                dataset.append(data)
                vertices.append(mesh.v)

        if self.dtype == 'train' :
            mean_train = torch.Tensor(np.mean(vertices, axis=0))
            std_train = torch.Tensor(np.std(vertices, axis=0))
            norm_dict = {'mean': mean_train, 'std': std_train}
            torch.save(self.collate(dataset), self.processed_paths[0])
            torch.save(norm_dict, self.processed_paths[2])
        elif self.dtype == 'test' :
            torch.save(self.collate(dataset), self.processed_paths[1])


def prepare_clsf_dataset(path, dtype):
    ComaDataset(path, dtype=dtype, split='clsf', pre_transform=None, transform=None)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ADNI2 Data preparation for Convolutional Mesh Autoencoders')
    parser.add_argument('-s', '--split', default='clsf', help='split can be clsf')
    parser.add_argument('-dt', '--data_type', default='train', help='train or test')
    parser.add_argument('-d', '--data_path', help='path where the downloaded data is stored, or csv file contains information of data files')

    args = parser.parse_args()
    split = args.split
    data_path = args.data_path
    dtype = args.data_type
    if split == 'clsf':
        prepare_clsf_dataset(data_path, dtype)
    else:
        raise Exception("Only clsf split are supported")


