import os
import os.path as osp
from glob import glob

import torch
from torch_geometric.data import InMemoryDataset, extract_zip
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Data, Batch

from tqdm import tqdm
import numpy as np
import openmesh as om

import sys
sys.path.insert(1, './utils/spiralnet')
from read import read_mesh

class SpiralNetDataset(InMemoryDataset):
    def __init__(self,
                 data_path,
                 train=True,
                 split='clsf',
                 transform=None,
                 pre_transform=None):
        self.split = split
        self.data_path = data_path
        self.train = train

        self.filepaths, self.categories = self.gather_paths(self.split)
        super().__init__(data_path, transform, pre_transform)

        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def processed_dir(self):
        return os.path.join('./', 'processed/spiralnet')

    @property
    def raw_file_names(self):
        all_fps = []
        for key in self.filepaths.keys() :
            all_fps += self.filepaths[key]
        return all_fps

    @property
    def processed_file_names(self):
        return [
            self.split + '_training.pt',
            self.split + '_testing.pt'
        ]

    def gather_paths(self, split):
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
        print('Processing...')
        dataset = []
        for dx in self.filepaths :
            for fp in self.filepaths[dx] :
                mesh = om.read_trimesh(fp)
                face = torch.from_numpy(mesh.face_vertex_indices()).T.type(torch.long)
                x = torch.tensor(mesh.points().astype('float32'))
                edge_index = torch.cat([face[:2], face[1:], face[::2]], dim=1)
                edge_index = to_undirected(edge_index)

                i = self.categories.index(dx)
                label = np.zeros(len(self.categories))
                label[i] = 1
                data = Data(x=x, y=torch.Tensor(label), edge_index=edge_index, face=face)

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                dataset.append(data)

        if self.train :
            torch.save(self.collate(train_data_list), self.processed_paths[0])
        else :
            torch.save(self.collate(test_data_list), self.processed_paths[1])


############################################################################
class SpiralNetDatasets(object):
    def __init__(self,
                 train_data,
                 test_data,
                 template_fp,
                 split='clsfb',
                 transform=None,
                 pre_transform=None):
        self.train_data = train_data
        self.test_data = test_data
        self.template_fp = template_fp
        self.split = split
        self.transform = transform
        self.pre_transform = pre_transform
        self.train_dataset = None
        self.test_dataste = None
        self.template_points = None
        self.template_face = None
        self.mean = None
        self.std = None
        self.num_nodes = None

        self.load()

    def load(self):
        self.train_dataset = SpiralNetDataset(self.train_data,
                                  train=True,
                                  split=self.split,
                                  transform=self.transform,
                                  pre_transform=self.pre_transform)
        self.test_dataset = SpiralNetDataset(self.test_data,
                                 train=False,
                                 split=self.split,
                                 transform=self.transform,
                                 pre_transform=self.pre_transform)

        tmp_mesh = om.read_trimesh(self.template_fp)
        self.template_points = tmp_mesh.points()
        self.template_face = tmp_mesh.face_vertex_indices()
        self.num_nodes = self.train_dataset[0].num_nodes

        self.num_train_graph = len(self.train_dataset)
        self.num_test_graph = len(self.test_dataset)
        self.mean = self.train_dataset.data.x.view(self.num_train_graph, -1,
                                                   3).mean(dim=0)
        self.std = self.train_dataset.data.x.view(self.num_train_graph, -1,
                                                  3).std(dim=0)
        self.normalize()

    def normalize(self):
        print('Normalizing...')
        self.train_dataset.data.x = (
            (self.train_dataset.data.x.view(self.num_train_graph, -1, 3) -
             self.mean) / self.std).view(-1, 3)
        self.test_dataset.data.x = (
            (self.test_dataset.data.x.view(self.num_test_graph, -1, 3) -
             self.mean) / self.std).view(-1, 3)
        print('Done!')

    def save_mesh(self, fp, x):
        x = x * self.std + self.mean
        om.write_mesh(fp, om.TriMesh(x.numpy(), self.template_face))

############################################################################
class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        def collate(data_list):
            batch = Batch()
            batch.batch = []
            for key in data_list[0].keys:
                batch[key] = default_collate([d[key] for d in data_list])
            for i, data in enumerate(data_list):
                num_nodes = data.num_nodes
                if num_nodes is not None:
                    item = torch.full((num_nodes, ), i, dtype=torch.long)
                    batch.batch.append(item)
            batch.batch = torch.cat(batch.batch, dim=0)
            return batch

        super(DataLoader, self).__init__(dataset,
                                         batch_size,
                                         shuffle,
                                         collate_fn=collate,
                                         **kwargs)
