import torch.nn.functional as F

from .softmax_loss import CrossEntropyLabelSmooth
from .center_loss import CenterLoss
from .triplet_loss import TripletLoss


def make_loss(Cfg, num_classes):    # modified by gu
    feat_dim = 2048

    if 'triplet' in Cfg.LOSS_TYPE:
        triplet = TripletLoss(Cfg.MARGIN)  # triplet loss

    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    if 'softmax' in Cfg.LOSS_TYPE:
        if Cfg.LOSS_LABELSMOOTH == 'on':
            xent = CrossEntropyLabelSmooth(num_classes=num_classes)  # new add by luo
            print("label smooth on, numclasses:", num_classes)

    def loss_func(score, feat, target):
        #feat: resnet提取出来的经过GAP的btz,2048
        #score: 将特征经过FC降维，成btz,num_class，如果是BNNECK，会在算
       
        #target: 
        if Cfg.LOSS_TYPE == 'triplet+softmax+center':
            #print('Train with center loss, the loss type is triplet+center_loss')
            if Cfg.LOSS_LABELSMOOTH == 'on':
                return Cfg.CE_LOSS_WEIGHT * xent(score, target) + \
                       Cfg.TRIPLET_LOSS_WEIGHT * triplet(feat, target)[0] + \
                       Cfg.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
            else:
                return Cfg.CE_LOSS_WEIGHT * F.cross_entropy(score, target) + \
                       Cfg.TRIPLET_LOSS_WEIGHT * triplet(feat, target)[0] + \
                        Cfg.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
        elif Cfg.LOSS_TYPE == 'softmax+center':
            #print('Train with center loss, the loss type is triplet+center_loss')
            if Cfg.LOSS_LABELSMOOTH == 'on':
                return Cfg.CE_LOSS_WEIGHT * xent(score, target) + \
                       Cfg.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
            else:
                return Cfg.CE_LOSS_WEIGHT * F.cross_entropy(score, target) + \
                        Cfg.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
        elif Cfg.LOSS_TYPE == 'triplet+softmax':
            #print('Train with center loss, the loss type is triplet+center_loss')
            if Cfg.LOSS_LABELSMOOTH == 'on':
                return Cfg.CE_LOSS_WEIGHT * xent(score, target) + \
                       Cfg.TRIPLET_LOSS_WEIGHT * triplet(feat, target)[0]
            else:
                return Cfg.CE_LOSS_WEIGHT * F.cross_entropy(score, target) + \
                       Cfg.TRIPLET_LOSS_WEIGHT * triplet(feat, target)[0]
        elif Cfg.LOSS_TYPE == 'softmax':
            if Cfg.LOSS_LABELSMOOTH == 'on':
                return xent(score, target)
            else:
                return F.cross_entropy(score, target)
        else:
            print('unexpected loss type')

    #为什么这里要return center_criterion?
    return loss_func, center_criterion