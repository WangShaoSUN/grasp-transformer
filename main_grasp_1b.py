import datetime
from doctest import FAIL_FAST
import os
import sys
import argparse
import logging
from tqdm import tqdm
import cv2

import torch
import torch.utils.data
import torch.optim as optim

from torchsummary import summary


from traning import train, validate
from utils.visualisation.gridshow import gridshow

from utils.dataset_processing import evaluation
from utils.data import get_dataset
from models.common import post_process_output
from models.swin import SwinTransformerSys
logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description='TF-Grasp')

    # Network
    # Dataset & Data & Training
    parser.add_argument('--dataset', type=str,default="graspnet1b", help='Dataset Name ("cornell" or "jaquard")')
    parser.add_argument('--use-depth', type=int, default=1, help='Use Depth image for training (1/0)')
    parser.add_argument('--use-rgb', type=int, default=0, help='Use RGB image for training (0/1)')
    parser.add_argument('--split', type=float, default=1., help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split for cross validation.')
    parser.add_argument('--num-workers', type=int, default=32, help='Dataset workers')

    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=500, help='Training epochs')
    parser.add_argument('--batches-per-epoch', type=int, default=500, help='Batches per Epoch')
    parser.add_argument('--val-batches', type=int, default=100, help='Validation Batches')
    parser.add_argument('--output-size', type=int, default=224,
                        help='the output size of the network, determining the cropped size of dataset images')

    parser.add_argument('--camera', type=str, default='realsense',
                        help='Which camera\'s data to use, only effective when using graspnet1b dataset')
    parser.add_argument('--scale', type=int, default=2,
                        help='the scale factor for the original images, only effective when using graspnet1b dataset')
    # Logging etc.
    parser.add_argument('--description', type=str, default='', help='Training description')
    parser.add_argument('--outdir', type=str, default='output/models/', help='Training Output Directory')
    parser.add_argument('--logdir', type=str, default='tensorboard/', help='Log directory')
    parser.add_argument('--vis', default=False,help='Visualise the training process')

    args = parser.parse_args()
    return args


def validate(net, device, val_data, batches_per_epoch,no_grasps=1):
    """
    Run validation.
    :param net: Network
    :param device: Torch device
    :param val_data: Validation Dataset
    :param batches_per_epoch: Number of batches to run
    :return: Successes, Failures and Losses
    """
    net.eval()

    results = {
        'correct': 0,
        'failed': 0,
        'loss': 0,
        'losses': {

        }
    }

    ld = len(val_data)
    
    with torch.no_grad():
        batch_idx = 0
        while batch_idx < batches_per_epoch:
            for x, y, didx, rot, zoom_factor in tqdm(val_data):
                batch_idx += 1
                if batches_per_epoch is not None and batch_idx >= batches_per_epoch:
                    break

                xc = x.to(device)
                yc = [yy.to(device) for yy in y]
                lossd = net.compute_loss(xc, yc)

                loss = lossd['loss']

                results['loss'] += loss.item()/ld
                for ln, l in lossd['losses'].items():
                    if ln not in results['losses']:
                        results['losses'][ln] = 0
                    results['losses'][ln] += l.item()/ld

                q_out, ang_out, w_out = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                            lossd['pred']['sin'], lossd['pred']['width'])
                # print("inde:",didx)

                s = evaluation.calculate_iou_match(q_out, ang_out,
                                                   val_data.dataset.get_gtbb(didx, rot, zoom_factor),
                                                   no_grasps=no_grasps,
                                                   grasp_width=w_out,
                                                   )

                if s:
                    results['correct'] += 1
                else:
                    results['failed'] += 1


    return results


def run():
    args = parse_args()

    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
    net_desc = '{}_{}'.format(dt, '_'.join(args.description.split()))

    save_folder = os.path.join(args.outdir, net_desc+"_d="+str(args.use_depth+args.use_rgb)+"_scale=3")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Load Dataset
    logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    Dataset = get_dataset(args.dataset)

    if args.dataset == 'graspnet1b':
        print("1 billion")
        train_dataset = Dataset( args.dataset_path, ds_rotate=args.ds_rotate,
                                output_size=args.output_size,
                                random_rotate=False, random_zoom=False,
                                include_depth=args.use_depth,
                                include_rgb=args.use_rgb,
                                camera=args.camera,
                                scale=args.scale,
                                split='train')
    else:
        train_dataset = Dataset(file_path=args.dataset_path, start=0.0, end=args.split, ds_rotate=args.ds_rotate,
                                output_size=args.output_size,
                                random_rotate=True, random_zoom=True,
                                include_depth=args.use_depth,
                                include_rgb=args.use_rgb)

    train_data = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False
    )
    train_validate_data = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers//4,
        pin_memory=False
    )
    if args.dataset == 'graspnet1b':
        val_dataset = Dataset(args.dataset_path, ds_rotate=args.ds_rotate,
                output_size=args.output_size,
                random_rotate=False, random_zoom=False,
                include_depth=args.use_depth,
                include_rgb=args.use_rgb,
                camera=args.camera,
                scale=args.scale,
                split='test_seen')
        val_dataset_1 = Dataset(args.dataset_path, ds_rotate=False,
                              output_size=args.output_size,
                              random_rotate=False, random_zoom=False,
                              include_depth=args.use_depth,
                              include_rgb=args.use_rgb,
                              camera=args.camera,
                              scale=args.scale,
                              split='test_similar')
        val_dataset_2 = Dataset(args.dataset_path, ds_rotate=False,
                                output_size=args.output_size,
                                random_rotate=False, random_zoom=False,
                                include_depth=args.use_depth,
                                include_rgb=args.use_rgb,
                                camera=args.camera,
                                scale=args.scale,
                                split='test_novel')

    else:
        val_dataset = Dataset(args.dataset_path, start=args.split, end=1.0, ds_rotate=args.ds_rotate,
                              output_size=args.output_size,
                              random_rotate=True, random_zoom=True,
                              include_depth=args.use_depth,
                              include_rgb=args.use_rgb)
    val_data = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,  # do not modify
        shuffle=True,
        num_workers=args.num_workers // 4,
        pin_memory=False,
    )
    val_data_1 = torch.utils.data.DataLoader(
        val_dataset_1,
        batch_size=1,  # do not modify
        shuffle=True,
        num_workers=args.num_workers // 4,
        pin_memory=False
    )
    val_data_2 = torch.utils.data.DataLoader(
        val_dataset_2,
        batch_size=1,  # do not modify
        shuffle=True,
        num_workers=args.num_workers // 4,
        pin_memory=False
    )
    logging.info('Done')

    # Load the network
    logging.info('Loading Network...')
    input_channels = 1*args.use_depth + 3*args.use_rgb
    print("channels:",input_channels)

    net=SwinTransformerSys(in_chans=input_channels,embed_dim=48,num_heads=[1, 2, 4, 8])
    device = torch.device("cuda:0")
    net = net.to(device)
    optimizer = optim.AdamW(net.parameters(),lr=1e-4)
    logging.info('Done')
    summary(net, (input_channels, 224, 224))
    f = open(os.path.join(save_folder, 'arch.txt'), 'w')
    sys.stdout = f
    summary(net, (input_channels, 224, 224))
    sys.stdout = sys.__stdout__
    f.close()

    best_iou = 0.0
    for epoch in range(args.epochs):
        logging.info('Beginning Epoch {:02d}'.format(epoch))
        train_results = train(epoch, net, device, train_data, optimizer, args.batches_per_epoch, vis=args.vis)
        

        test_results = validate(net, device, train_validate_data, args.val_batches)
        logging.info(' traning %d/%d = %f' % (test_results['correct'], test_results['correct'] + test_results['failed'],
                                           test_results['correct'] / (
                                                       test_results['correct'] + test_results['failed'])))
        logging.info('loss/train_loss: %f'%test_results['loss'])

        test_results = validate(net, device, val_data, args.val_batches)
        logging.info(' seen %d/%d = %f' % (test_results['correct'], test_results['correct'] + test_results['failed'],
                                     test_results['correct']/(test_results['correct']+test_results['failed'])))
        logging.info('loss/seen_loss: %f'%test_results['loss'])
        
        test_results = validate(net, device, val_data_1, args.val_batches)
        logging.info('similar %d/%d = %f' % (test_results['correct'], test_results['correct'] + test_results['failed'],
                                     test_results['correct'] / (test_results['correct'] + test_results['failed'])))
        logging.info('loss/similar_loss: %f'%test_results['loss'])

        test_results = validate(net, device, val_data_2, args.val_batches,no_grasps=2)
        logging.info('novel %d/%d = %f' % (test_results['correct'], test_results['correct'] + test_results['failed'],
                                     test_results['correct'] / (test_results['correct'] + test_results['failed'])))
        logging.info('loss/novel_loss: %f'%test_results['loss'])


        # Save best performing network
        iou = test_results['correct'] / (test_results['correct'] + test_results['failed'])
        if iou > best_iou or epoch == 0 or (epoch % 10) == 0:
            torch.save(net, os.path.join(save_folder, 'epoch_%02d_iou_%0.2f' % (epoch, iou)))
            # torch.save(net.state_dict(), os.path.join(save_folder, 'epoch_%02d_iou_%0.2f_statedict.pt' % (epoch, iou)))
            best_iou = iou


if __name__ == '__main__':
    run()
