


    def get_pseudo_label(self, net, x, mult=3):
        preditions_augs = []
        is_training = net.training
        # if is_training:
        #     net.eval()
        outnet = net(x)
        preditions_augs.append(F.softmax(outnet["out"], dim=1))

        for i in range(mult-1):
            # x_aug, rotate_affine = self.randomRotate(x)
            # x_aug, vflip_affine = self.randomVerticalFlip(x_aug)
            # x_aug, hflip_affine = self.randomHorizontalFlip(x_aug)

            x_aug, hflip_affine = self.randomHorizontalFlip(x)
            x_aug, crop_affine  = self.randomResizeCrop(x_aug)

            # get label on x_aug
            outnet = net(x_aug)
            pred_aug = outnet["out"]
            
            pred_aug = F.softmax(pred_aug, dim=1)
            pred_aug = self.apply_invert_affine(pred_aug, crop_affine)
            pred_aug = self.apply_invert_affine(pred_aug, hflip_affine)

            preditions_augs.append(pred_aug)


        preditions = torch.stack(preditions_augs, dim=0).mean(dim=0) # batch x n_classes x h x w
        # renormalize the probability (due to interpolation of zeros, mean does not imply probability distribution over the classes)
        preditions = preditions / torch.sum(preditions, dim=1, keepdim=True)
        # if is_training:
        #     net.train()
        return preditions