import logging
import time
import os
import numpy as np
import torch
from tqdm import tqdm
import torch.optim as optim

from robustbench.data import get_dataset,convert_2d,scal_spacing,scale
from robustbench.utils import load_model,setup_source
from robustbench.utils import clean_accuracy as accuracy
from robustbench.metrics import dice_eval,assd_eval, hd95_eval
from robustbench.losses import DiceLoss,DiceCeLoss,WeightedCrossEntropyLoss
import tent
import norm
import cotta
from unet import UNet
import SimpleITK as sitk
from conf import cfg, load_cfg_fom_args


logger = logging.getLogger(__name__)


def train_source(description,train_source = True,infer_test_data = True,save = False):
    logger.info(f"max_epochs: %s", cfg.SOURCE.MAX_EPOCHES)
    load_cfg_fom_args(description)
    # configure model
    base_model = load_model(cfg.MODEL.NETWORK, cfg.MODEL.CKPT_DIR,
                       cfg.ADAPTATION.DATASET,cfg.MODEL.ADAPTATION).cuda()
    logger.info("train source model: NONE")
    model = setup_source(base_model)
    dice_loss = DiceLoss(cfg.ADAPTATION.NUMBER_CLASS)#.to(torch.cuda())
    dice_loss.cuda()
    
    db_train,db_valid,db_test = get_dataset(dataset=cfg.ADAPTATION.DATASET, domain=cfg.SOURCE.SOURCE_DOMAIN)
    train_loader = torch.utils.data.DataLoader(db_train, 
                    batch_size = cfg.SOURCE.BATCH_SIZE, shuffle=False, num_workers= 4)
    valid_loader = torch.utils.data.DataLoader(db_valid, 
                    batch_size = 1, shuffle=False, num_workers= 4)
    test_loader = torch.utils.data.DataLoader(db_test, 
                    batch_size = 1, shuffle=False, num_workers= 4)
    iter_num = 0
    valid_best = 0
    save_model_dir = os.path.join('save_model',cfg.ADAPTATION.DATASET+'_'+cfg.MODEL.NETWORK)
    if ( not os.path.exists(save_model_dir)):
        os.mkdir(save_model_dir)
    if train_source:
        for epoch_num in tqdm(range(cfg.SOURCE.MAX_EPOCHES), ncols=70):
            time1 = time.time()
            for i_batch, sampled_batch in enumerate(train_loader):
                iter_num += 1*cfg.SOURCE.BATCH_SIZE
                # print('fetch data cost {}'.format(time2-time1))
                volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                volume_batch, label_batch = convert_2d(volume_batch.cuda(), label_batch.cuda())
                model.train_source(volume_batch, label_batch)
                if iter_num % cfg.SOURCE.EVAL_ITERS == 0:
                    with torch.no_grad():
                        acc = 0.0
                        for i, sampled_batch in enumerate(valid_loader):
                            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                            volume_batch, label_batch = convert_2d(volume_batch.cuda(), label_batch.cuda())
                            acc += accuracy(cfg.ADAPTATION.NUMBER_CLASS, model, volume_batch, label_batch)
                        dice_loss_value = acc/(i+1)
                        dice = 1. - dice_loss_value
                        if dice > valid_best:
                            valid_best = dice
                            torch.save(model.state_dict(),'{}/{}-{}-model-best.pth'.format(save_model_dir,cfg.MODEL.ADAPTATION,cfg.SOURCE.SOURCE_DOMAIN))
                            logger.info(f"Source valid best dice [{iter_num}]: {dice:.2%}")
                        logger.info(f"Source valid dice [{iter_num}]: {dice:.2%}")
        torch.save(model.state_dict(),'{}/{}-{}-model-latest.pth'.format(save_model_dir,cfg.MODEL.ADAPTATION,cfg.SOURCE.SOURCE_DOMAIN))
    if infer_test_data: 
        
        best_pth = '{}/{}-{}-model-best.pth'.format(save_model_dir,cfg.MODEL.ADAPTATION,cfg.SOURCE.SOURCE_DOMAIN)
        # best_pth = '/data2/jianghao/TTA-MT/TTA-MT/save_model/mms3d_unet/norm-model-best.pth'
        model.load_state_dict(torch.load(best_pth,map_location='cpu'))
        model.eval()
        for test_domain in cfg.SOURCE.ALL_DOMAIN:
            all_batch_dice = []
            all_batch_assd = []
            all_batch_hd = []
            _,db_test,_ = get_dataset(dataset=cfg.ADAPTATION.DATASET, domain=test_domain)
            test_loader = torch.utils.data.DataLoader(db_test, batch_size = 1, shuffle=False, num_workers= 4)
            # test for the test dataset!
            with torch.no_grad():
                for i, sampled_batch in enumerate(test_loader):
                    volume_batch, label_batch, names, spacing = sampled_batch['image'], sampled_batch['label'], sampled_batch['names'], sampled_batch['spacing']
                    volume_batch, label_batch = convert_2d(volume_batch.cuda(), label_batch.cuda())
                    output_soft = model(volume_batch).softmax(1)
                    output = output_soft.argmax(1).cpu().numpy()
                    if save:
                        output = scale(output,label_batch.shape)
                        name = names[0].split('/')[-1]
                        results = '{}/{}-{}-{}'.format('results',cfg.ADAPTATION.DATASET,test_domain,cfg.MODEL.ADAPTATION)
                        if(not os.path.exists(results)):
                            os.mkdir(results)
                        predict_dir  = os.path.join(results, name)
                        out_lab_obj = sitk.GetImageFromArray(output/1.0)
                        spacing_or = (spacing[2].numpy()[0],spacing[1].numpy()[0],spacing[0].numpy()[0])
                        
                        out_lab_obj.SetSpacing(spacing_or)
                        sitk.WriteImage(out_lab_obj, predict_dir)
                    label = label_batch.cpu().numpy().squeeze(1)
                    # spacing_3d = (spacing[0].numpy()[0],spacing[1].numpy()[0],spacing[2].numpy()[0])
                    # output,label = scal_spacing(output, label,spacing_3d)
                    one_case_dice = dice_eval(output,label,cfg.ADAPTATION.NUMBER_CLASS)
                    all_batch_dice += [one_case_dice]
                    try:
                        one_case_assd = assd_eval(output,label,cfg.ADAPTATION.NUMBER_CLASS)
                        if one_case_assd.mean(0) > 20:
                            one_case_assd = 20
                    except:
                        one_case_assd = 20.1236
                    # try:
                    #     one_case_hd95 = hd95_eval(output,label,cfg.ADAPTATION.NUMBER_CLASS)
                    #     if one_case_hd95 > 20:
                    #         one_case_hd95 = 20
                    # except:
                    #     one_case_hd95 = 20
                    all_batch_assd.append(one_case_assd)
                    # all_batch_hd.append(one_case_hd95)
            all_batch_dice = np.array(all_batch_dice)
            all_batch_assd = np.array(all_batch_assd)
            # all_batch_hd = np.array(all_batch_hd)
            mean_dice = np.mean(all_batch_dice,axis=0) 
            std_dice = np.std(all_batch_dice,axis=0) 
            mean_assd = np.mean(all_batch_assd,axis=0)
            std_assd = np.std(all_batch_assd,axis=0)
            # mean_hd = np.mean(all_batch_hd,axis=0)
            # std_hd = np.std(all_batch_hd,axis=0)
            print('-----------',cfg.ADAPTATION.DATASET,'---',test_domain)
            if cfg.ADAPTATION.DATASET=='mms3d' or 'mms2d':
                print('{}±{} {}±{} {}±{}'.format(np.round(mean_dice[0],2),np.round(std_dice[0],2),np.round(mean_dice[1],2),np.round(std_dice[1],2),np.round(mean_dice[2],2),np.round(std_dice[2],2)))
            elif cfg.ADAPTATION.DATASET=='fb':
                print('{}±{}'.format(np.round(mean_dice[0],2),np.round(std_dice[0],2)))
            if cfg.ADAPTATION.DATASET=='mms3d' or 'mms2d':
                # print('ASSD:')
                print('{}±{} {}±{} {}±{}'.format(np.round(mean_assd[0],2),np.round(std_assd[0],2),np.round(mean_assd[1],2),np.round(std_assd[1],2),np.round(mean_assd[2],2),np.round(std_assd[2],2)))
                # print('HD95:')
                # print('{}±{} {}±{} {}±{}'.format(np.round(mean_hd[0],2),np.round(std_hd[0],2),np.round(mean_hd[1],2),np.round(std_hd[1],2),np.round(mean_hd[2],2),np.round(std_hd[2],2)))
            elif cfg.ADAPTATION.DATASET=='fb':
                # print('ASSD:')
                print('{}±{}'.format(np.round(mean_assd[0],2),np.round(std_assd[0],2)))
                # print('HD95:')
                # print('{}±{}'.format(np.round(mean_hd[0],2),np.round(std_hd[0],2)))



if __name__ == '__main__':
    train_source('mms train source.',train_source = True,infer_test_data = True,save = False)

