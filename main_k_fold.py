import datetime
import os
import sys
import argparse
import logging
import cv2
import torch
import torch.utils.data
import torch.optim as optim
from torchsummary import summary
from sklearn.model_selection import KFold
from traning import train, validate
from utils.data import get_dataset
from models.swin import SwinTransformerSys
# from models.Swin_without_skipconcetion import SwinTransformerSys
logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description='TF-Grasp')

    # Network


    # Dataset & Data & Training
    parser.add_argument('--dataset', type=str,default="cornell", help='Dataset Name ("cornell" or "jaquard or multi")')
    parser.add_argument('--dataset-path', type=str,default="/home/sam/Desktop/archive111" ,help='Path to dataset')
    parser.add_argument('--use-depth', type=int, default=1, help='Use Depth image for training (1/0)')
    parser.add_argument('--use-rgb', type=int, default=0, help='Use RGB image for training (0/1)')
    parser.add_argument('--split', type=float, default=0.9, help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split for cross validation.')
    parser.add_argument('--num-workers', type=int, default=8, help='Dataset workers')

    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=2000, help='Training epochs')
    parser.add_argument('--batches-per-epoch', type=int, default=500, help='Batches per Epoch')
    parser.add_argument('--val-batches', type=int, default=50, help='Validation Batches')
    # Logging etc.
    parser.add_argument('--description', type=str, default='', help='Training description')
    parser.add_argument('--outdir', type=str, default='output/models/', help='Training Output Directory')

    args = parser.parse_args()
    return args


def run():
    args = parse_args()
    # Set-up output directories
    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
    net_desc = '{}_{}'.format(dt, '_'.join(args.description.split()))

    save_folder = os.path.join(args.outdir, net_desc)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Load Dataset
    logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    Dataset = get_dataset(args.dataset)


    dataset = Dataset(args.dataset_path, start=0.0, end=1.0, ds_rotate=args.ds_rotate,
                                              random_rotate=True, random_zoom=True,
                                              include_depth=args.use_depth, include_rgb=args.use_rgb)
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True)
    logging.info('Done')
    logging.info('Loading Network...')
    input_channels = 1*args.use_depth + 3*args.use_rgb
    net = SwinTransformerSys(in_chans=input_channels,embed_dim=48,num_heads=[1,2,4,8])
    device = torch.device("cuda:0")
    net = net.to(device)
    optimizer = optim.AdamW(net.parameters(), lr=5e-4)
    listy = [x *7 for x in range(1,1000,3)]
    schedule=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=listy,gamma=0.6)
    logging.info('Done')
    # Print model architecture.
    summary(net, (input_channels, 224, 224))
    f = open(os.path.join(save_folder, 'arch.txt'), 'w')
    sys.stdout = f
    summary(net, (input_channels, 224, 224))
    sys.stdout = sys.__stdout__
    f.close()

    best_iou = 0.0
    for epoch in range(args.epochs):
        accuracy=0.
        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):

            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
            trainloader = torch.utils.data.DataLoader(
                           dataset,
                           batch_size=args.batch_size,num_workers=args.num_workers, sampler=train_subsampler)
            testloader = torch.utils.data.DataLoader(
                           dataset,
                           batch_size=1,num_workers=args.num_workers, sampler=test_subsampler)


            logging.info('Beginning Epoch {:02d}'.format(epoch))
            print("lr:",optimizer.state_dict()['param_groups'][0]['lr'])
            train_results = train(epoch, net, device, trainloader, optimizer, args.batches_per_epoch, )
            schedule.step()

        # Run Validation
            logging.info('Validating...')
            test_results = validate(net, device, testloader, args.val_batches)
            logging.info('%d/%d = %f' % (test_results['correct'], test_results['correct'] + test_results['failed'],
                                     test_results['correct']/(test_results['correct']+test_results['failed'])))


            iou = test_results['correct'] / (test_results['correct'] + test_results['failed'])
            accuracy+=iou
            if iou > best_iou or epoch == 0 or (epoch % 50) == 0:
                torch.save(net, os.path.join(save_folder, 'epoch_%02d_iou_%0.2f' % (epoch, iou)))
                # torch.save(net.state_dict(), os.path.join(save_folder, 'epoch_%02d_iou_%0.2f_statedict.pt' % (epoch, iou)))
                best_iou = iou
        schedule.step()
        print("the accuracy:",accuracy/k_folds)


if __name__ == '__main__':
    run()


