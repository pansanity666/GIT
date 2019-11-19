import torch

def make_optimizer(Cfg, model, center_criterion):

    #center_criterion是centerloss
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = Cfg.BASE_LR
        weight_decay = Cfg.WEIGHT_DECAY
        # if "bias" in key:
        #     lr = Cfg.BASE_LR * Cfg.BIAS_LR_FACTOR
        #     weight_decay = Cfg.WEIGHT_DECAY_BIAS
        params += [ {"params": [value], "lr": lr, "weight_decay": weight_decay} ]

    if Cfg.OPTIMIZER == 'SGD':

        #getattr用于获取属性值，就是取了SGD优化器
        optimizer = getattr(torch.optim, Cfg.OPTIMIZER)(params, momentum=Cfg.MOMENTUM)

    else:

        optimizer = getattr(torch.optim, Cfg.OPTIMIZER)(params)

    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=Cfg.CENTER_LR)

    #这个optimizer_center是干啥用的？
    return optimizer, optimizer_center



if __name__ == '__main__':
    from torchvision import models
    from config import Config
    Cfg = Config()
    x = models.resnet50(pretrained=True)
    make_optimizer(x,Cfg,)