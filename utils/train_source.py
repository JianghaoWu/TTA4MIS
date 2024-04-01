
def train_source(description,train_source = True,infer_test_data = True,save = False):
    logger.info(f"max_epochs: %s", cfg.SOURCE.MAX_EPOCHES)
    load_cfg_fom_args(description)
    # configure model
    base_model = load_model(cfg.MODEL.NETWORK, cfg.MODEL.CKPT_DIR,
                       cfg.ADAPTATION.DATASET,cfg.MODEL.METHOD).cuda()
    logger.info("train source model: NONE")
    model = setup_source(base_model)
    dice_loss = DiceLoss(cfg.ADAPTATION.NUMBER_CLASS)#.to(torch.cuda())
    dice_loss.cuda()
    
    db_train,db_valid,db_test = get_dataset(dataset=cfg.ADAPTATION.DATASET, domain=cfg.SOURCE.SOURCE_DOMAIN)
    train_loader = torch.utils.data.DataLoader(db_train, 
                    batch_size = cfg.SOURCE.BATCH_SIZE, shuffle=False, num_workers= 160)
    valid_loader = torch.utils.data.DataLoader(db_valid, 
                    batch_size = 1, shuffle=False, num_workers= 16)
    test_loader = torch.utils.data.DataLoader(db_test, 
                    batch_size = 1, shuffle=False, num_workers= 16)
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
                            torch.save(model.state_dict(),'{}/{}-{}-model-best.pth'.format(save_model_dir,cfg.MODEL.METHOD,cfg.SOURCE.SOURCE_DOMAIN))
                            logger.info(f"Source valid best dice [{iter_num}]: {dice:.2%}")
                        logger.info(f"Source valid dice [{iter_num}]: {dice:.2%}")
        torch.save(model.state_dict(),'{}/{}-{}-model-latest.pth'.format(save_model_dir,cfg.MODEL.METHOD,cfg.SOURCE.SOURCE_DOMAIN))
