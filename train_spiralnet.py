import pickle
import time
import argparse
import os
import os.path as osp
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import sklearn
import torch_geometric.transforms as T
from psbody.mesh import Mesh

import sys
sys.path.insert(1, './utils/spiralnet')
sys.path.insert(1, './models')
sys.path.insert(1, './datasets')
from spiralnet_model import SpiralNet
from spiralnet_dataset import SpiralNetDatasets, DataLoader
import utils, writer, mesh_sampling

def run(model, train_loader, test_loader, epochs, optimizer, scheduler, writer,
        device):

    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('runs/spiralnet/', 'clsf-' + current_time)
    tsbwriter = SummaryWriter(log_dir+'')

    train_losses, test_losses = [], []
    best_test_acc = float('inf')

    for epoch in range(1, epochs + 1):
        t = time.time()
        train_loss, train_acc = train(model, optimizer, train_loader, device)
        t_duration = time.time() - t
        test_loss, test_acc, clsf_rpt = test(model, test_loader, device)
        scheduler.step()
        info = {
            'current_epoch': epoch,
            'epochs': epochs,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'train_acc': train_acc,
            'test_acc': test_acc,
            't_duration': t_duration,
            'clsf_rpt': clsf_rpt,
        }

        writer.print_info(info)
        print('Train Acc: {:.4f}, Test Acc: {:.4f}'.format(info['train_acc'], info['test_acc']))

        tsbwriter.add_scalar('data/train_loss', train_loss, epoch)
        tsbwriter.add_scalar('data/test_loss', test_loss, epoch)
        tsbwriter.add_scalar('data/train_acc', train_acc, epoch)
        tsbwriter.add_scalar('data/test_acc', test_acc, epoch)
        print(clsf_rpt)

        if test_acc < best_test_acc :
            writer.save_checkpoint(model, optimizer, scheduler, epoch)
            best_test_acc = test_acc
        if epoch == epochs or epoch % 100 == 0:
            writer.save_checkpoint(model, optimizer, scheduler, epoch)

    tsbwriter.close()


def train(model, optimizer, loader, device):
    model.train()
    crt = 0
    total_loss = 0
    total_len = 0
    for data in loader:
        optimizer.zero_grad()
        x = data.to(device)
        out = model(x)
        y = torch.reshape(data.y, (data.num_graphs, 2))
        y = torch.argmax(y, 1)
        loss = F.cross_entropy(out, y)
        loss.backward()
        total_loss += data.num_graphs * loss.item()
        crt += y.eq(out.data.max(1)[1]).sum().item()
        total_len += data.num_graphs
        optimizer.step()
    return total_loss / total_len, crt / total_len


def test(model, loader, device):
    model.eval()
    crt = 0
    total_loss = 0
    total_len = 0
    preds = []
    labels = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            x = data.to(device)
            out = model(x)
            y = torch.reshape(data.y, (data.num_graphs, 2))
            y = torch.argmax(y, 1)
            loss = F.cross_entropy(out, y)
            total_loss += data.num_graphs * loss.item()
            crt += y.eq(out.data.max(1)[1]).sum().item()
            preds.append(out.data.max(1)[1].detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())
            total_len += data.num_graphs

    clsf_rpt = sklearn.metrics.classification_report(np.concatenate(labels, axis=None), \
        np.concatenate(preds, axis=None))
    return total_loss / total_len, crt / total_len, clsf_rpt

#################################################################################

parser = argparse.ArgumentParser(description='mesh classifier')
parser.add_argument('--exp_name', type=str, default='default')
parser.add_argument('--train_data', type=str, default='pcs_mesh_mask_vols_train_set_1.csv')
parser.add_argument('--test_data', type=str, default='pcs_mesh_mask_vols_test_set_1.csv')
parser.add_argument('--split', type=str, default='clsf')
parser.add_argument('--n_threads', type=int, default=4)
parser.add_argument('--device_idx', type=int, default=1)

# network hyperparameters
parser.add_argument('--out_channels',
                    nargs='+',
                    default=[32, 32, 32, 64],
                    type=int)
parser.add_argument('--latent_channels', type=int, default=16)
parser.add_argument('--in_channels', type=int, default=3)
parser.add_argument('--seq_length', type=int, default=[9, 9, 9, 9], nargs='+')
parser.add_argument('--dilation', type=int, default=[1, 1, 1, 1], nargs='+')

# optimizer hyperparmeters
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_decay', type=float, default=0.99)
parser.add_argument('--decay_step', type=int, default=1)
parser.add_argument('--weight_decay', type=float, default=0)

# training hyperparameters
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=300)

# others
parser.add_argument('--seed', type=int, default=1)

args = parser.parse_args()

args.work_dir = osp.dirname(osp.realpath(__file__))
args.out_dir = osp.join(args.work_dir, 'logs', 'spiralnet')
args.checkpoints_dir = osp.join(args.work_dir, 'ckpts', 'spiralnet', args.exp_name)
print(args)

utils.makedirs(args.out_dir)
utils.makedirs(args.checkpoints_dir)

writer = writer.Writer(args)
device = torch.device('cuda', args.device_idx) if torch.cuda.is_available() else torch.device('cpu')
torch.set_num_threads(args.n_threads)

# deterministic
torch.manual_seed(args.seed)
cudnn.benchmark = False
cudnn.deterministic = True

# load dataset
template_fp = osp.join('./template', 'L_Hipp_template_om_2922.ply')
meshdata = SpiralNetDatasets(args.train_data, args.test_data,
                             template_fp, split=args.split)
train_loader = DataLoader(meshdata.train_dataset,
                          batch_size=args.batch_size,
                          shuffle=True)
test_loader = DataLoader(meshdata.test_dataset, batch_size=1, shuffle=False)

# generate/load transform matrices
transform_fp = osp.join('./processed/spiralnet', 'matrices.pkl')
if not osp.exists(transform_fp):
    print('Generating transform matrices...')
    mesh = Mesh(filename=template_fp)
    ds_factors = [2, 2, 2, 2]
    _, A, D, U, F, V = mesh_sampling.generate_transform_matrices(
        mesh, ds_factors)
    tmp = {
        'vertices': V,
        'face': F,
        'adj': A,
        'down_transform': D,
        'up_transform': U
    }

    with open(transform_fp, 'wb') as fp:
        pickle.dump(tmp, fp)
    print('Done!')
    print('Transform matrices are saved in \'{}\''.format(transform_fp))
else:
    with open(transform_fp, 'rb') as f:
        tmp = pickle.load(f, encoding='latin1')

spiral_indices_list = [
    utils.preprocess_spiral(tmp['face'][idx], args.seq_length[idx],
                            tmp['vertices'][idx],
                            args.dilation[idx]).to(device)
    for idx in range(len(tmp['face']) - 1)
]
down_transform_list = [
    utils.to_sparse(down_transform).to(device)
    for down_transform in tmp['down_transform']
]
up_transform_list = [
    utils.to_sparse(up_transform).to(device)
    for up_transform in tmp['up_transform']
]

model = SpiralNet(args.in_channels, args.out_channels, args.latent_channels,
           spiral_indices_list, down_transform_list,
           up_transform_list).to(device)
print('Number of parameters: {}'.format(utils.count_parameters(model)))
print(model)

optimizer = torch.optim.Adam(model.parameters(),
                             lr=args.lr,
                             weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            args.decay_step,
                                            gamma=args.lr_decay)
t = time.time()
run(model, train_loader, test_loader, args.epochs, optimizer, scheduler,
    writer, device)
t_duration = time.time() - t
print('total time : {:.3f}s'.format(t_duration))
