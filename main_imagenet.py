import torch
import time
import shutil
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from loss.contrastive import CL
from loss.logitadjust import LogitAdjust
from loss.ProDual import ProDual
import math
from tensorboardX import SummaryWriter
from dataset.imagenet import ImageNetLT
from models import  resnet, resnext
import warnings
import torch.backends.cudnn as cudnn
import random
from randaugment import rand_augment_transform
import torchvision
from utils import GaussianBlur, shot_acc
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='imagenet', choices=['inat', 'imagenet'])
parser.add_argument('--data', default='/data/ILSVRC/', metavar='DIR')
parser.add_argument('--arch', default='resnext50', choices=['resnet50', 'resnext50'])
parser.add_argument('--workers', default=12, type=int)
parser.add_argument('--epochs', default=180, type=int)
parser.add_argument('--temp', default=0.07, type=float, help='scalar temperature for contrastive learning')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.2, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[160, 180], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
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
parser.add_argument('--weight_c',default=0.5, type=float, help='class-prototype-based distribution alignment (CPDA) loss weight')
parser.add_argument('--randaug', default=True, type=bool, help='use RandAugmentation for classification branch')
parser.add_argument('--cl_views', default='sim-sim', type=str, choices=['sim-sim', 'sim-rand', 'rand-rand'],
                    help='Augmentation strategy for contrastive learning views')
parser.add_argument('--feat_dim', default=2048, type=int, help='feature dimension of mlp head')
parser.add_argument('--warmup_epochs', default=0, type=int,
                    help='warmup epochs')
parser.add_argument('--root_log', type=str, default='\log')
parser.add_argument('--root_save', type=str, default='\save')
parser.add_argument('--cos', default=True, type=bool,
                    help='lr decays by cosine scheduler. ')
parser.add_argument('--use_norm', default=True, type=bool,
                    help='cosine classifier.')
parser.add_argument('--randaug_m', default=10, type=int, help='randaug-m')
parser.add_argument('--randaug_n', default=2, type=int, help='randaug-n')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')
parser.add_argument('--reload', default=False, type=bool, help='load supervised model')
parser.add_argument('--num_classes', default=1000, type=int)
parser.add_argument('--cls_num', default=1000, type=int)
parser.add_argument('--save_freq', default=10, type=int)

def main():
    args = parser.parse_args()
    args.store_name = '_'.join(
        [args.dataset, args.arch, 'batchsize', str(args.batch_size), 'epochs', str(args.epochs), 'temp', str(args.temp),
         'lr', str(args.lr), args.cl_views])

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
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))

    if args.arch == 'resnext50':
        model = resnext.ProConDual(name='resnext50', num_classes=args.num_classes, feat_dim=args.feat_dim,
                                 use_norm=args.use_norm)
    else:
        raise NotImplementedError('This model is not supported')
    print(model)

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

    txt_train = f'dataset/ImageNet_LT/ImageNet_LT_train.txt' if args.dataset == 'imagenet' \
        else f'dataset/iNaturalist18/iNaturalist18_train.txt'
    txt_val = f'dataset/ImageNet_LT/ImageNet_LT_val.txt' if args.dataset == 'imagenet' \
        else f'dataset/iNaturalist18/iNaturalist18_val.txt'

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
        normalize,
    ]
    augmentation_randnclsstack = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(args.randaug_n, args.randaug_m), ra_params),
        transforms.ToTensor(),
        normalize,
    ]
    augmentation_sim = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize
    ]
    if args.cl_views == 'sim-sim':
        transform_train = [transforms.Compose(augmentation_randncls), transforms.Compose(augmentation_sim),
                           transforms.Compose(augmentation_sim), ]
    elif args.cl_views == 'sim-rand':
        transform_train = [transforms.Compose(augmentation_randncls), transforms.Compose(augmentation_randnclsstack),
                           transforms.Compose(augmentation_sim), ]
    elif args.cl_views == 'randstack-randstack':
        transform_train = [transforms.Compose(augmentation_randncls), transforms.Compose(augmentation_randnclsstack),
                           transforms.Compose(augmentation_randnclsstack), ]
    else:
        raise NotImplementedError("This augmentations strategy is not available for contrastive learning branch!")
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    val_dataset = ImageNetLT(
        root=args.data,
        txt=txt_val,
        transform=val_transform, train=False)

    train_dataset = ImageNetLT(
        root=args.data,
        txt=txt_train,
        transform=transform_train)

    cls_num_list = train_dataset.cls_num_list
    args.cls_num = len(cls_num_list)

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    criterion_lc = LogitAdjust(cls_num_list).cuda(args.gpu)
    criterion_cl = CL(cls_num_list, args.temp).cuda(args.gpu)
    criterion_produal=ProDual( tau=1, weight=None, batch_size=args.batch_size)

    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))

    best_acc1 = 0.0
    best_many, best_med, best_few = 0.0, 0.0, 0.0

    if args.reload:
        txt_test = f'dataset/ImageNet_LT/ImageNet_LT_test2.txt' if args.dataset == 'imagenet' \
            else f'dataset/iNaturalist18/iNaturalist18_val.txt'
        test_dataset =ImageNetLT(
            root=args.data,
            txt=txt_test,
            transform=val_transform, train=False)

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        acc1, many, med, few = validate(train_loader, test_loader, model, criterion_lc, 1, args, tf_writer)
        print('Prec@1: {:.3f}, Many Prec@1: {:.3f}, Med Prec@1: {:.3f}, Few Prec@1: {:.3f}'.format(acc1,
                                                                                                   many,
                                                                                                   med,
                                                                                                   few))
        return


    args.save_folder = os.path.join(args.root_save, args.store_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_lr(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion_lc, criterion_cl, criterion_produal, optimizer, epoch, args, tf_writer)

        # evaluate on validation set
        acc1, many, med, few = validate(train_loader, val_loader, model, criterion_lc, epoch, args,
                                        tf_writer)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if is_best:
            best_many = many
            best_med = med
            best_few = few
        print('Best Prec@1: {:.3f}, Many Prec@1: {:.3f}, Med Prec@1: {:.3f}, Few Prec@1: {:.3f}'.format(best_acc1,
                                                                                                        best_many,
                                                                                                        best_med,
                                                                                                best_few))
        if (epoch+1)%args.save_freq==0 :
            save_path = '{}/ckpt_epoch_{}.pth'.format(args.save_folder,epoch+1)
                
            state_dict =  {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'alpha': args.alpha,
                    'beta': args.beta,
                    'gamma': args.gamma,
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
        batch_size = targets.shape[0]
        feat_mlp, logits, centers, classifier_weight = model(inputs)
        
        if torch.cuda.device_count()>1:
            classifier_weight= model.module.fc.weight
        centers = centers[:args.cls_num]
        
        _, f2, f3 = torch.split(feat_mlp, [batch_size, batch_size, batch_size], dim=0)
        features = torch.cat([f2.unsqueeze(1), f3.unsqueeze(1)], dim=1)
        logits, _, __ = torch.split(logits, [batch_size, batch_size, batch_size], dim=0)
        cl_loss = criterion_cl(centers, features, targets).cuda()
        lc_loss = criterion_lc(logits, targets).cuda()
        produal_loss=criterion_produal(centers, classifier_weight).cuda()
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
                lc_loss=lc_loss_all, cl_loss=cl_loss_all,produal_loss=produal_loss_all, top1=top1, ))  # TODO
            print(output)
            
    if epoch==args.start_epoch:
        args.alpha=round(args.weight_a/lc_loss_all.avg,2)
        args.beta=round(args.weight_b/cl_loss_all.avg,2)
        args.gamma=round(args.weight_c/produal_loss_all.avg,2)
        
    tf_writer.add_scalar('LC loss/train', lc_loss_all.avg, epoch)
    tf_writer.add_scalar('CL loss/train', cl_loss_all.avg, epoch)
    tf_writer.add_scalar('PRODUAL loss/train', produal_loss_all.avg, epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)


def validate(train_loader, val_loader, model, criterion_lc, epoch, args, tf_writer=None, flag='val'):
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
            batch_size = targets.size(0)
            feat_mlp, logits, centers, classifier_weight = model(inputs)
            lc_loss = criterion_lc(logits, targets)

            total_logits = torch.cat((total_logits, logits))
            total_labels = torch.cat((total_labels, targets))

            acc1 = accuracy(logits, targets, topk=(1,))
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

        probs, preds = F.softmax(total_logits.detach(), dim=1).max(dim=1)
        many_acc_top1, median_acc_top1, low_acc_top1 = shot_acc(preds, total_labels, train_loader,
                                                                acc_per_cls=False)
        return top1.avg, many_acc_top1, median_acc_top1, low_acc_top1


def save_checkpoint(args, state, is_best):
    filename = os.path.join(args.root_log, args.store_name, 'bcl_ckpt.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


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
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
