from copy import deepcopy
import torch
import torch.optim as optim
from tta import tent,norm,cotta,wjh01,upl,meant,sar
from utils.sam import SAM
from conf import cfg
import math

def setup_sar(model):
    model = sar.configure_model(model)
    params, param_names = sar.collect_params(model)
    base_optimizer = torch.optim.SGD
    optimizer = SAM(params, base_optimizer, lr=cfg.OPTIM.LR, momentum=0.9)
    adapt_model = sar.SAR(model, optimizer, margin_e0=0.4*math.log(2))

    return adapt_model

def setup_norm(model):
    """Set up test-time normalization adaptation.

    Adapt by normalizing features with test batch statistics.
    The statistics are measured independently for each batch;
    no running average or other cross-batch estimation is used.
    """
    norm_model = norm.Norm(model)
    stats, stat_names = norm.collect_stats(model)
    print(stat_names)
   
    return norm_model

def setup_tent(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = setup_optimizer(params)
    tent_model = tent.Tent(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC)
    return model,tent_model
    
def setup_source(model):
    """Set up the baseline source model without adaptation."""
    model.eval()
    # logger.info(f"model for evaluation: %s", model)
    return model

def setup_cotta(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = cotta.configure_model(model)
    params, param_names = cotta.collect_params(model)
    optimizer = setup_optimizer(params)
    cotta_model = cotta.CoTTA(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC, 
                           mt_alpha=cfg.OPTIM.MT, 
                           rst_m=cfg.OPTIM.RST, 
                           ap=cfg.OPTIM.AP)
    return cotta_model
def setup_upl(model):
    model.train()
    num_dec = cfg.MODEL.NUM_DEC
    dec_list = []
    for i in range(1, num_dec+1):
        dec_i = deepcopy(model.dec1)
        setattr(dec_i, 'name', f'dec_{i}')  
        dec_list.append(dec_i)
    optimizer_params = []
    optimizer_params.extend(model.enc.parameters())
    optimizer = setup_optimizer(optimizer_params)
    upl_model = upl.TTA(model.enc, dec_list, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC, 
                           mt_alpha=cfg.OPTIM.MT, 
                           rst_m=cfg.OPTIM.RST, 
                           )
    return upl_model
def setup_wjh01(model):
    anchor_model = deepcopy(model)
    model.train()
    anchor_model.eval()
    optimizer = torch.optim.Adam(model.parameters(),lr=cfg.OPTIM.LR,betas=(0.5,0.999))
    cotta_model = wjh01.TTA(model, anchor_model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC, 
                           mt_alpha=cfg.OPTIM.MT, 
                           rst_m=cfg.OPTIM.RST, 
                           )
    return cotta_model
def setup_meant(model):
    anchor_model = deepcopy(model)
    model.train()
    anchor_model.eval()
    optimizer = torch.optim.Adam(model.parameters(),lr=cfg.OPTIM.LR,betas=(0.5,0.999))
    cotta_model = meant.TTA(model, anchor_model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC, 
                           mt_alpha=cfg.OPTIM.MT, 
                           rst_m=cfg.OPTIM.RST, 
                           )
    return cotta_model
def setup_optimizer(params):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params,
                    lr=cfg.OPTIM.LR,
                    betas=(cfg.OPTIM.BETA, 0.999),
                    weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD(params,
                   lr=cfg.OPTIM.LR,
                   momentum=cfg.OPTIM.MOMENTUM,
                   dampening=cfg.OPTIM.DAMPENING,
                   weight_decay=cfg.OPTIM.WD,
                   nesterov=cfg.OPTIM.NESTEROV)
    else:
        raise NotImplementedError