import time
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import torch

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import sys
sys.path.insert(1, './utils/meshcnn')
sys.path.insert(1, './models')
sys.path.insert(1, './datasets')
from writer import Writer
from test import run_test
from train_options import TrainOptions
from meshcnn_dataset import DataLoader
from meshcnn_model import ClassifierModel

if __name__ == '__main__':

    opt = TrainOptions().parse()

    '''CREATE DIR'''
    log_dir = Path('./logs/meshcnn/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("MeshCNN")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('./logs/meshcnn/train_%s_'%opt.name+ str(datetime.datetime.now().strftime('%Y-%m-%d %H-%M'))+'.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('---------------------------------------------------TRANING---------------------------------------------------')
    logger.info('PARAMETER ...')
    logger.info(opt)

    logger.info('Load Dataset ...')
    dataset = DataLoader(opt, opt.train_data)
    dataset_size = len(dataset)
    print('#training meshes = %d' % dataset_size)
    logger.info('#training meshes = %d', dataset_size)

    model = ClassifierModel(opt)
    num_total_params = sum(p.numel() for p in model.net.parameters())
    num_trainable_params = sum(p.numel() for p in model.net.parameters() if p.requires_grad)
    print('Number of total paramters: %d, number of trainable parameters: %d' % (num_total_params, num_trainable_params))
    logger.info('Number of total paramters: %d, number of trainable parameters: %d', num_total_params, num_trainable_params)

    writer = Writer(opt)
    total_steps = 0
    train_start_time = time.time()
    best_tst_acc = 0.0

    torch.manual_seed(1)
    cudnn.benchmark = False
    cudnn.deterministic = True

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        heappop_error_train = 0
        logger.info('Epoch %d started ...', epoch)
        writer.reset_counter()

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            try:
                model.optimize_parameters(writer)
            except IndexError:
                total_steps -= opt.batch_size
                epoch_iter -= opt.batch_size
                heappop_error_train += 1
                print('(%d) Index Error Occured, Total Step: %d, Epoch: %d, Epoch Iter: %d' %
                     (heappop_error_train, total_steps, epoch, epoch_iter))
                continue

            if total_steps % opt.print_freq == 0:
                loss = model.loss
                t = (time.time() - iter_start_time) / opt.batch_size
                writer.print_current_losses(epoch, epoch_iter, loss, t, t_data)
                #writer.plot_loss(loss, epoch, epoch_iter, dataset_size)

            if i % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_network('latest_net')

            iter_data_time = time.time()

        writer.plot_loss(loss, epoch, 1, 1)

        print('TRAIN ACC [%.3f]'%(writer.acc))
        writer.plot_train_acc(writer.acc, epoch)
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_network('latest_net')
            # model.save_network('epoch_%d' % (epoch))


        model.update_learning_rate()
        if opt.verbose_plot:
            writer.plot_model_wts(model, epoch)

        if epoch % opt.run_test_freq == 0:
            acc = run_test(opt.test_data, epoch)
            writer.plot_acc(acc, epoch)

            logger.info('Loss: %f, Acc: %f', loss, acc)
            logger.info('IndexError count - train: %d', heappop_error_train)
            print('End of epoch %d / %d \t Time Taken: %d sec' %
                  (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
            logger.info('End of epoch %d / %d \t Time Taken: %d sec',
                        epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time)

            if (acc >= best_tst_acc) and epoch > 5:
                best_tst_acc = acc
                model.save_network('%.6f-%04d' % (acc, epoch))
                print('Saving model....')


    writer.close()
    train_end_time = time.time()
    print('Train is finished, Time taken: %d sec' % (train_end_time - train_start_time))
