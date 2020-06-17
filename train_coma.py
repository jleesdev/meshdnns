import argparse
import os
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from psbody.mesh import Mesh, MeshViewers
import mesh_operations
from config_parser import read_config
from data_clsf import ComaDataset
from model_clsf import Coma
from transform_clsf import Normalize
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
import sklearn
import torch.backends.cudnn as cudnn

torch.manual_seed(1)
cudnn.benchmark = False
cudnn.deterministic = True
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def scipy_to_torch_sparse(scp_matrix):
    values = scp_matrix.data
    indices = np.vstack((scp_matrix.row, scp_matrix.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = scp_matrix.shape

    sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return sparse_tensor

def adjust_learning_rate(optimizer, lr_decay):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * lr_decay

def save_model(coma, optimizer, epoch, train_loss, val_loss, checkpoint_dir):
    checkpoint = {}
    checkpoint['state_dict'] = coma.state_dict()
    checkpoint['optimizer'] = optimizer.state_dict()
    checkpoint['epoch_num'] = epoch
    checkpoint['train_loss'] = train_loss
    checkpoint['val_loss'] = val_loss
    torch.save(checkpoint, os.path.join(checkpoint_dir, 'checkpoint_'+ str(epoch)+'.pt'))

def main(args):
    if not os.path.exists(args.conf):
        print('Config not found' + args.conf)

    config = read_config(args.conf)
    for k in config.keys() :
        print(k, config[k])

    print('Initializing parameters')
    template_file_path = config['template_fname']
    template_mesh = Mesh(filename=template_file_path)

    if args.checkpoint_dir:
        checkpoint_dir = args.checkpoint_dir
    else:
        checkpoint_dir = config['checkpoint_dir']
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    visualize = config['visualize']
    output_dir = config['visual_output_dir']
    if visualize is True and not output_dir:
        print('No visual output directory is provided. Checkpoint directory will be used to store the visual results')
        output_dir = checkpoint_dir

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    eval_flag = config['eval']
    lr = config['learning_rate']
    lr_decay = config['learning_rate_decay']
    weight_decay = config['weight_decay']
    total_epochs = config['epoch']
    workers_thread = config['workers_thread']
    opt = config['optimizer']
    batch_size = config['batch_size']
    val_losses, accs, durations = [], [], []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Generating transforms')
    M, A, D, U = mesh_operations.generate_transform_matrices(template_mesh, config['downsampling_factors'])

    D_t = [scipy_to_torch_sparse(d).to(device) for d in D]
    U_t = [scipy_to_torch_sparse(u).to(device) for u in U]
    A_t = [scipy_to_torch_sparse(a).to(device) for a in A]
    num_nodes = [len(M[i].v) for i in range(len(M))]

    print('Loading Dataset')
    if args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = config['data_dir']

    normalize_transform = Normalize()
    dataset = ComaDataset(data_dir, dtype='train', split=args.split, split_term=args.split_term, pre_transform=normalize_transform)
    dataset_test = ComaDataset(data_dir, dtype='test', split=args.split, split_term=args.split_term, pre_transform=normalize_transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers_thread)
    val_loader = DataLoader(dataset_test, batch_size=1, shuffle=True, num_workers=workers_thread)
    test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=workers_thread)

    print('Loading model')
    start_epoch = 1
    coma = Coma(dataset, config, D_t, U_t, A_t, num_nodes)
    if opt == 'adam':
        optimizer = torch.optim.Adam(coma.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt == 'sgd':
        optimizer = torch.optim.SGD(coma.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise Exception('No optimizer provided')

    checkpoint_file = config['checkpoint_file']
    print(checkpoint_file)
    if checkpoint_file:
        checkpoint = torch.load(checkpoint_file)
        start_epoch = checkpoint['epoch_num']
        coma.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        #To find if this is fixed in pytorch
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    coma.cuda()

    if eval_flag:
        val_loss = evaluate(coma, output_dir, test_loader, dataset_test, template_mesh, device, visualize)
        print('val loss', val_loss)
        return

    best_val_loss = float('inf')
    val_loss_history = []
    best_val_acc = float('inf')

    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('runs/clsf', current_time)
    writer = SummaryWriter(log_dir+'raw_ds2_e300')

    t = time.time()
    num_total_params = sum(p.numel() for p in coma.parameters())
    num_trainable_params = sum(p.numel() for p in coma.parameters() if p.requires_grad)
    print('Number of total paramters: {}, number of trainable parameters: {}'\
        .format(num_total_params, num_trainable_params))

    for epoch in range(start_epoch, total_epochs + 1):
        print("Training for epoch ", epoch)
        train_loss, train_acc = train(coma, train_loader, len(dataset), optimizer, device)
        val_loss, val_acc, clsf_rpt = evaluate(coma, output_dir, val_loader, dataset_test, template_mesh, device, epoch, visualize=visualize)

        writer.add_scalar('data/train_loss', train_loss, epoch)
        writer.add_scalar('data/test_loss', val_loss, epoch)
        writer.add_scalar('data/train_acc', train_acc, epoch)
        writer.add_scalar('data/test_acc', val_acc, epoch)

        print('epoch ', epoch,' Train loss ', train_loss, ' Val loss ', val_loss)
        print(' Train acc ', train_acc, ' Val acc ', val_acc)
        print(clsf_rpt)
        #if val_loss < best_val_loss:
        if val_acc < best_val_acc :
            save_model(coma, optimizer, epoch, train_loss, val_loss, checkpoint_dir)
            best_val_acc = val_acc

        if epoch == total_epochs or epoch % 100 == 0:
            save_model(coma, optimizer, epoch, train_loss, val_loss, checkpoint_dir)

        val_loss_history.append(val_loss)
        val_losses.append(best_val_loss)

        if opt=='sgd':
            adjust_learning_rate(optimizer, lr_decay)

    t_duration = time.time() - t
    print('total time : {:.3f}s'.format(t_duration))
    print('Number of paramters: {}'\
        .format(num_total_params))

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    writer.close()

def train(coma, train_loader, len_dataset, optimizer, device):
    coma.train()
    total_loss = 0
    crt = 0
    for data in tqdm(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = coma(data)
        y = torch.reshape(data.y, (data.num_graphs, 2))
        y = torch.argmax(y, 1)
        #print(out, out.data.max(1)[1])
        loss = F.cross_entropy(out, y)
        crt += y.eq(out.data.max(1)[1]).sum().item()
        #print(crt)
        total_loss += data.num_graphs * loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / len_dataset, crt / len_dataset

def evaluate(coma, output_dir, test_loader, dataset, template_mesh, device, epoch, visualize=False):
    coma.eval()
    total_loss = 0
    crt = 0
    preds = []
    labels = []
    for i, data in enumerate(test_loader):
        data = data.to(device)
        with torch.no_grad():
            out = coma(data)
        y = torch.reshape(data.y, (data.num_graphs, 2))
        y = torch.argmax(y, 1)
        loss = F.cross_entropy(out, y)
        crt += y.eq(out.data.max(1)[1]).sum().item()
        preds.append(out.data.max(1)[1].detach().cpu().numpy())
        labels.append(y.detach().cpu().numpy())
        total_loss += data.num_graphs * loss.item()

    clsf_rpt = sklearn.metrics.classification_report(np.concatenate(labels, axis=None), \
        np.concatenate(preds, axis=None))
    return total_loss/len(dataset), crt/len(dataset), clsf_rpt


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pytorch Trainer for Convolutional Mesh Autoencoders')
    parser.add_argument('-c', '--conf', default='cfgs/clsf.cfg', help='path of config file')
    parser.add_argument('-s', '--split', default='clsfb', help='split can be gnrt, clsf, or lgtd')
    parser.add_argument('-st', '--split_term', default='clsfb', help='split can be gnrt, clsf, or lgtd')
    parser.add_argument('-d', '--data_dir', default='data/ADNI2_data', help='path where the downloaded data is stored')
    parser.add_argument('-cp', '--checkpoint_dir', help='path where checkpoints file need to be stored')

    args = parser.parse_args()

    if args.conf is None:
        args.conf = os.path.join(os.path.dirname(__file__), 'default.cfg')
        print('configuration file not specified, trying to load '
              'it from current directory', args.conf)

    main(args)
