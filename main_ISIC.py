import torch
import time
import shutil
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from loss.contrastive import CL
from loss.logitadjust import LogitAdjust
from loss.ProDual import ProDual
import math
from tensorboardX import SummaryWriter
from dataset.isic import isic2019_dataset
from dataset.isic2018 import isic2018_dataset

from models import resnet
import warnings
import torch.backends.cudnn as cudnn
import random
from randaugment import rand_augment_transform
import torchvision
from utils import GaussianBlur, shot_acc
import argparse
import os



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ISIC2019', choices=['ISIC2019'])
parser.add_argument('--arch', default='resnet50', choices=['resnet50', 'resnet34'])
parser.add_argument('--workers', default=1, type=int)
parser.add_argument('--save_freq', default=50, type=int)
parser.add_argument('--epochs', default=600, type=int)
parser.add_argument('--temp', default=0.07, type=float, help='scalar temperature for contrastive learning')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[160, 180], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

parser.add_argument('--weight_a', default=0.7, type=float, help='logit compensation loss (lc) weight')
parser.add_argument('--weight_b', default=0.3, type=float, help='contrastive loss (cl) weight')
parser.add_argument('--weight_c',default=0.5, type=float, help='ProDual loss weight')

parser.add_argument('--randaug', default=True, type=bool, help='use RandAugmentation for classification branch')
parser.add_argument('--cl_views', default='sim-sim', type=str, choices=['sim-sim', 'sim-rand', 'rand-rand'],
                    help='Augmentation strategy for contrastive learning views')
parser.add_argument('--feat_dim', default=2048, type=int, help='feature dimension of mlp head')
parser.add_argument('--warmup_epochs', default=0, type=int,
                    help='warmup epochs')
parser.add_argument('--cos', default=True, type=bool,
                    help='lr decays by cosine scheduler. ')
parser.add_argument('--use_norm', default=True, type=bool,
                    help='cosine classifier.')
parser.add_argument('--randaug_m', default=10, type=int, help='randaug-m')
parser.add_argument('--randaug_n', default=2, type=int, help='randaug-n')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')
parser.add_argument('--reload', default=False, type=bool, help='load supervised model')
parser.add_argument('--num_classes', default=8, type=int)
parser.add_argument('--root_log', type=str, default='./log')
parser.add_argument('--root_save', type=str, default='./save')
parser.add_argument('--data_path', type=str, default='../../ISIC2019')

def main():
    args = parser.parse_args()
    args.store_name = '_'.join(
        [args.dataset, args.arch, 'bs', str(args.batch_size),'lr', str(args.lr), args.cl_views,'dim',str(args.feat_dim)])

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
        
    ngpus_per_node = torch.cuda.device_count()

    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    print("ngpus_per_node@@@ ",ngpus_per_node )
    
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch == 'resnet50':
        model = resnet.ProConDual(name='resnet50', num_classes=args.num_classes, feat_dim=args.feat_dim,
                                 use_norm=args.use_norm)

    else:
        raise NotImplementedError('This model is not supported')

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:0')
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    cudnn.benchmark = True


    normalize = transforms.Normalize((0.466, 0.471, 0.380), (0.195, 0.194, 0.192)) if args.dataset == 'inat' \
        else transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    rgb_mean = (0.485, 0.456, 0.406)
    ra_params = dict(translate_const=int(224 * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]), )
    augmentation_randncls = [
        transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([

            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
        ], p=1.0),
        rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(args.randaug_n, args.randaug_m), ra_params),
        transforms.ToTensor(),
        transforms.RandomErasing(value=3), #new
        
        normalize,
    ]
    
    augmentation_sim = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        
        transforms.ToTensor(),
        transforms.RandomErasing(value=3), #new
        
        normalize
    ]
    
    
    transform_train = [transforms.Compose(augmentation_randncls), transforms.Compose(augmentation_sim),
                           transforms.Compose(augmentation_sim), ]

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    

    if args.dataset == 'ISIC2019':
        train_dataset=isic2019_dataset(path=args.data_path, transform=transform_train,num_classes=args.num_classes ,mode='train')
        args.cls_num_list =train_dataset.cls_num_list 

        train_loader = DataLoader(isic2019_dataset(path=args.data_path, transform=transform_train,num_classes=args.num_classes ,mode='train'),
                                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
        val_loader = DataLoader(isic2019_dataset(path=args.data_path, transform=val_transform,num_classes=args.num_classes, mode='valid'), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)  
    
    cls_num_list=args.cls_num_list
    args.cls_num=len(cls_num_list)

    criterion_lc = LogitAdjust(cls_num_list).cuda(args.gpu)
    criterion_cl = CL(cls_num_list, args.temp).cuda(args.gpu)
    criterion_produal=ProDual( tau=1, weight=None, batch_size=args.batch_size)

    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))


    if args.reload:
        test_loader = DataLoader(isic2019_dataset(path=args.data_path, transform=val_transform,num_classes=args.num_classes, mode='test'), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)  

        acc1= validate(test_loader, model, criterion_lc, 1, args, tf_writer)
        print('Acc@1: {:.3f}'.format(acc1))
        return




    args.save_folder = os.path.join(args.root_save, args.store_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)
        
    best_acc1 = 0.0
    best_many, best_med, best_few = 0.0, 0.0, 0.0


    for epoch in range(args.start_epoch, args.epochs):
        adjust_lr(optimizer, epoch, args)

        train(train_loader, model, criterion_lc, criterion_cl, criterion_produal,optimizer, epoch, args, tf_writer)

        val_acc= validate(train_loader, val_loader, model, criterion_lc, epoch, args,
                                        tf_writer)

        if (epoch+1)%args.save_freq==0 :
            save_path = '{}/ckpt_epoch_{}.pth'.format(args.save_folder,epoch+1)
                
            state_dict =  {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch
                }
            torch.save(state_dict, save_path)


def train(train_loader, model, criterion_lc, criterion_cl,criterion_produal, optimizer, epoch, args, tf_writer):
    batch_time = AverageMeter('Time', ':6.3f')
    lc_loss_all = AverageMeter('LC_Loss', ':.4e')
    produal_loss_all = AverageMeter('PRODUAL_Loss', ':.4e')
    cl_loss_all = AverageMeter('CL_Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    model.train()
    end = time.time()
    if epoch==args.start_epoch: 
        args.alpha=1
        args.beta=1
        args.gamma=1
        
    for i, data in enumerate(train_loader):
        inputs, targets = data
        inputs = torch.cat([inputs[0], inputs[1], inputs[2]], dim=0)
        inputs, targets = inputs.cuda(), targets.cuda()
        targets=torch.squeeze(targets)
        batch_size = targets.shape[0]
        feat_mlp, logits, centers, classifier_weight  = model(inputs)
        centers = centers[:args.cls_num]
        
        _, f2, f3 = torch.split(feat_mlp, [batch_size, batch_size, batch_size], dim=0)

        features = torch.cat([f2.unsqueeze(1), f3.unsqueeze(1)], dim=1)
        logits, _, __ = torch.split(logits, [batch_size, batch_size, batch_size], dim=0)
        cl_loss = criterion_cl(centers, features, targets).cuda()
        lc_loss = criterion_lc(logits, targets).cuda(cl_loss.device)
        produal_loss=criterion_produal(centers, classifier_weight ).cuda(cl_loss.device)
        
        loss = args.alpha * lc_loss + args.beta * cl_loss+args.gamma*produal_loss

        lc_loss_all.update(lc_loss.item(), batch_size)
        cl_loss_all.update(cl_loss.item(), batch_size)
        produal_loss_all.update(produal_loss.item(), batch_size)
        acc1 = accuracy(logits, targets, topk=(1,))
        top1.update(acc1[0].item(), batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}] \t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'LC_Loss {lc_loss.val:.4f} ({lc_loss.avg:.4f})\t'
                      'CL_Loss {cl_loss.val:.4f} ({cl_loss.avg:.4f})\t'
                      'PRODUAL_Loss {produal_loss.val:.4f} ({produal_loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                lc_loss=lc_loss_all, cl_loss=cl_loss_all,produal_loss=produal_loss_all, top1=top1))  # TODO
            print(output)

    if epoch==args.start_epoch:
        args.alpha=round(args.weight_a/lc_loss_all.avg,2)
        args.beta=round(args.weight_b/cl_loss_all.avg,2)
        args.gamma=round(args.weight_c/produal_loss_all.avg,2)
        
    tf_writer.add_scalar('LC loss/train', lc_loss_all.avg, epoch)
    tf_writer.add_scalar('CL loss/train', cl_loss_all.avg, epoch)
    tf_writer.add_scalar('PRODUAL loss/train', produal_loss_all.avg, epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)


def validate(val_loader, model, criterion_lc, epoch, args, tf_writer=None, flag='val'):
    model.eval()
    batch_time = AverageMeter('Time', ':6.3f')
    lc_loss_all = AverageMeter('LC_Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    total_logits = torch.empty((0, args.cls_num)).cuda()
    total_labels = torch.empty(0, dtype=torch.long).cuda()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            inputs, targets = data
            inputs, targets = inputs.cuda(), targets.cuda()
            targets=torch.squeeze(targets)
            
            batch_size = targets.size(0)
            feat_mlp, logits, centers, classifier_weight = model(inputs)
            lc_loss = criterion_lc(logits, targets)
            logits=logits.cuda(total_logits.device)
            total_logits = torch.cat((total_logits, logits))
            total_labels = torch.cat((total_labels, targets))

            acc1 = accuracy(logits, targets, topk=(1,))
            print("acc", acc1[0].item())
            lc_loss_all.update(lc_loss.item(), batch_size)
            top1.update(acc1[0].item(), batch_size)

            batch_time.update(time.time() - end)

        if i % args.print_freq == 0:
            output = ('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'LC_Loss {lc_loss.val:.4f} ({lc_loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, lc_loss=lc_loss_all, top1=top1, ))  # TODO
            print(output)

        tf_writer.add_scalar('LC loss/val', lc_loss_all.avg, epoch)
        tf_writer.add_scalar('acc/val_top1', top1.avg, epoch)


        return top1.avg


class TwoCropTransform:
    def __init__(self, transform1, transform2):
        self.transform1 = transform1
        self.transform2 = transform2

    def __call__(self, x):
        return [self.transform1(x), self.transform2(x), self.transform2(x)]


def adjust_lr(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if epoch < args.warmup_epochs:
        lr = lr / args.warmup_epochs * (epoch + 1)
    elif args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs + 1) / (args.epochs - args.warmup_epochs + 1)))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t().cuda(target.device)
        correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
