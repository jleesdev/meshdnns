import argparse
import glob
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm

from psbody.mesh import Mesh
from utils import get_vert_connectivity
from transform_clsf import Normalize

class ComaDataset(InMemoryDataset):
    def __init__(self, root_dir, dtype='train', split='sliced', split_term='sliced', nVal = 100, transform=None, pre_transform=None):
        self.root_dir = root_dir
        self.split = split
        self.split_term = split_term
        self.nVal = nVal
        self.transform = transform
        self.pre_tranform = pre_transform
        # Downloaded data is present in following format root_dir/*/*/*.py
        self.data_file = self.gather_paths(self.split_term)
        super(ComaDataset, self).__init__(root_dir, transform, pre_transform)
        if dtype == 'train':
            data_path = self.processed_paths[0]
        #elif dtype == 'val':
        #    data_path = self.processed_paths[1]
        elif dtype == 'test':
            data_path = self.processed_paths[1]
        else:
            raise Exception("train, val and test are supported data types")

        norm_path = self.processed_paths[2]
        norm_dict = torch.load(norm_path)
        self.mean, self.std = norm_dict['mean'], norm_dict['std']
        self.data, self.slices = torch.load(data_path)
        if self.transform:
            self.data = [self.transform(td) for td in self.data]

    @property
    def raw_file_names(self):
        all_data_files = []
        for key in self.data_file.keys() :
            all_data_files += self.data_file[key]
        return all_data_files

    @property
    def processed_file_names(self):
        processed_files = ['training.pt', 'test.pt', 'norm.pt']
        processed_files = [self.split_term+'_'+pf for pf in processed_files]
        return processed_files

    def gather_paths(self, st):
        datapaths = dict()
        count = 0
        if st == 'clsfb' :
            datapaths['ad'] = dict()
            datapaths['cn'] = dict()
            datapaths['ad']['train'] = []
            datapaths['ad']['test'] = []
            datapaths['cn']['train'] = []
            datapaths['cn']['test'] = []

            imgids = np.load('data/adni2_clsfb_dict.npy', allow_pickle=True)
            imgids = imgids.item()
            for dx in imgids.keys() :
                for tt in imgids[dx].keys() :
                    for i in imgids[dx][tt] :
                        datapaths[dx][tt].append(self.root_dir+'/'+i+'-L_Hipp_first.obj')
                        count += 1
        print('total number of dataset: %d'%(count))
        return datapaths

    def process(self):
        train_data, val_data, test_data = [], [], []
        train_vertices = []
        for dx in self.data_file :
            for tt in self.data_file[dx] :
                for f in tqdm(self.data_file[dx][tt]):
                    mesh = Mesh(filename=f)
                    mesh_verts = torch.Tensor(mesh.v)
                    adjacency = get_vert_connectivity(mesh.v, mesh.f).tocoo()
                    edge_index = torch.Tensor(np.vstack((adjacency.row, adjacency.col)))
                    if dx == 'ad' :
                        data = Data(x=mesh_verts, y=torch.Tensor([1,0]), edge_index=edge_index)
                    elif dx == 'cn' :
                        data = Data(x=mesh_verts, y=torch.Tensor([0,1]), edge_index=edge_index)

                    if tt == 'test' :
                        test_data.append(data)
                    elif tt == 'train' :
                        train_data.append(data)
                        train_vertices.append(mesh.v)

        mean_train = torch.Tensor(np.mean(train_vertices, axis=0))
        std_train = torch.Tensor(np.std(train_vertices, axis=0))
        norm_dict = {'mean': mean_train, 'std': std_train}
        if self.pre_transform is not None:
            if hasattr(self.pre_transform, 'mean') and hasattr(self.pre_transform, 'std'):
                if self.pre_tranform.mean is None:
                    self.pre_tranform.mean = mean_train
                if self.pre_transform.std is None:
                    self.pre_tranform.std = std_train
            train_data = [self.pre_transform(td) for td in train_data]
            val_data = [self.pre_transform(td) for td in val_data]
            test_data = [self.pre_transform(td) for td in test_data]

        torch.save(self.collate(train_data), self.processed_paths[0])
        #torch.save(self.collate(val_data), self.processed_paths[1])
        torch.save(self.collate(test_data), self.processed_paths[1])
        torch.save(norm_dict, self.processed_paths[2])

def prepare_clsfb_dataset(path):
    ComaDataset(path, split='clsfb', split_term='clsfb', pre_transform=Normalize())


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ADNI2 Data preparation for Convolutional Mesh Autoencoders')
    parser.add_argument('-s', '--split', default='clsfb', help='split can be gnrt, clsf, lgtd, lgtdvc, or lgtddx')
    parser.add_argument('-d', '--data_dir', help='path where the downloaded data is stored')

    args = parser.parse_args()
    split = args.split
    data_dir = args.data_dir
    if split == 'clsfb':
        prepare_clsfb_dataset(data_dir)
    else:
        raise Exception("Only gnrt, clsf, and lgtd split are supported")


