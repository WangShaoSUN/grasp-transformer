import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data
from utils.data import get_dataset
from utils.dataset_processing.grasp import detect_grasps,GraspRectangles
from models.common import post_process_output
import cv2
import matplotlib
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
matplotlib.use("TkAgg")


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate GG-CNN')

    # Network
    parser.add_argument('--network', type=str,default="./output/models/220623_1311_/epoch_08_iou_0.97", help='Path to saved network to evaluate')

    # Dataset & Data & Training
    parser.add_argument('--dataset', type=str, default="cornell",help='Dataset Name ("cornell" or "jaquard")')
    parser.add_argument('--dataset-path', type=str,default="/home/sam/Desktop/archive111" ,help='Path to dataset')
    parser.add_argument('--use-depth', type=int, default=0, help='Use Depth image for training (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1, help='Use RGB image for training (0/1)')
    parser.add_argument('--split', type=float, default=0.9,
                        help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split for cross validation.')
    parser.add_argument('--num-workers', type=int, default=8, help='Dataset workers')

    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--vis', type=bool, default=False, help='vis')
    parser.add_argument('--epochs', type=int, default=2000, help='Training epochs')
    parser.add_argument('--batches-per-epoch', type=int, default=200, help='Batches per Epoch')
    parser.add_argument('--val-batches', type=int, default=32, help='Validation Batches')
    # Logging etc.
    parser.add_argument('--description', type=str, default='', help='Training description')
    parser.add_argument('--outdir', type=str, default='output/models/', help='Training Output Directory')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print(args.network)
    print(args.use_rgb,args.use_depth)
    net = torch.load(args.network)
    # net_ggcnn = torch.load('./output/models/211112_1458_/epoch_30_iou_0.75')
    device = torch.device("cuda:0")
    Dataset = get_dataset(args.dataset)

    val_dataset = Dataset(args.dataset_path, start=args.split, end=1.0, ds_rotate=args.ds_rotate,
                          random_rotate=True, random_zoom=False,
                          include_depth=args.use_depth, include_rgb=args.use_rgb)
    val_data = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers
    )
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
        # fig = plt.figure(figsize=(10, 10))
        # ax = fig.add_subplot(1, 4, 1)
        # while batch_idx < 100:
        for id,(x, y, didx, rot, zoom_factor) in enumerate( val_data):
                # batch_idx += 1

                print(id)
                print(x.shape)
                xc = x.to(device)
                yc = [yy.to(device) for yy in y]
                lossd = net.compute_loss(xc, yc)

                loss = lossd['loss']

                results['loss'] += loss.item() / ld
                for ln, l in lossd['losses'].items():
                    if ln not in results['losses']:
                        results['losses'][ln] = 0
                    results['losses'][ln] += l.item() / ld

                q_out, ang_out, w_out = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                            lossd['pred']['sin'], lossd['pred']['width'])
                gs_1 = detect_grasps(q_out, ang_out, width_img=w_out, no_grasps=1)
                rgb_img=val_dataset.get_rgb(didx, rot, zoom_factor, normalise=False)

                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(1, 4, 1)
                ax.imshow(rgb_img)

                ax = fig.add_subplot(1, 4, 2)
                plot = ax.imshow(q_out, cmap="jet", vmin=0, vmax=1)  # ？terrain
                plt.colorbar(plot)
                ax.axis('off')
                ax.set_title('q image')

                ax = fig.add_subplot(1, 4, 3)  # flag  prism jet
                plot = ax.imshow(ang_out, cmap="hsv", vmin=-np.pi / 2, vmax=np.pi / 2)
                plt.colorbar(plot)
                ax.axis('off')
                ax.set_title('angle')

                ax = fig.add_subplot(1, 4, 4)
                plot = ax.imshow(w_out, cmap='jet', vmin=-0, vmax=150)
                plt.colorbar(plot)
                ax.set_title('width')
                ax.axis('off')
                # print(rgb_img)

                plt.show()
                plt.savefig('RGB_1_%d.pdf' % 1, bbox_inches='tight')