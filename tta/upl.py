from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import torch.nn.functional as F
import PIL
import torchvision.transforms as transforms
import my_transforms as my_transforms
from time import time
import logging


def get_tta_transforms(gaussian_std: float=0.005, soft=False, clip_inputs=False):
    img_shape = (320, 320, 3)
    n_pixels = img_shape[0]

    clip_min, clip_max = 0.0, 1.0

    p_hflip = 0.5

    tta_transforms = transforms.Compose([
        my_transforms.Clip(0.0, 1.0), 
        transforms.Pad(padding=int(n_pixels / 2), padding_mode='edge'),  
        transforms.RandomAffine(
            degrees=[-8, 8] if soft else [-15, 15],
            translate=(1/16, 1/16),
            scale=(0.95, 1.05) if soft else (0.9, 1.1),
            shear=None,
            resample=PIL.Image.BILINEAR,
            fillcolor=None
        ),
        transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
        transforms.CenterCrop(size=n_pixels),
        transforms.RandomHorizontalFlip(p=p_hflip),
        my_transforms.GaussianNoise(0, gaussian_std),
        my_transforms.Clip(clip_min, clip_max)
    ])
    return tta_transforms


def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model


class TTA(nn.Module):

    def __init__(self, model_enc,model_dec_list, optimizer, steps=1, episodic=False, mt_alpha=0.99, rst_m=0.1, ap=0.9):
        super().__init__()
        self.model_enc = model_enc
        self.model_dec_list = model_dec_list
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "cotta requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.segloss = DiceLoss(4).to(torch.device('cuda'))
        model_enc_state, model_dec_state, optimizer_state = \
            copy_model_and_optimizer(self.model_enc, self.model_dec_list, self.optimizer)
        self.transform = get_tta_transforms()    
        self.mt = mt_alpha
        self.rst = rst_m
        self.ap = ap
        self.pl_threshold = 0.9
    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.optimizer)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        # Use this line to also restore the teacher model                         
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)


    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, optimizer):
        self.get_pseudo_label(self.model_enc, self.model_dec_list, x, mult=len(self.model_dec_list))
        self.save_nii()

        self.get_pseudo_label(self.model_enc, self.model_dec_list, x, mult=len(self.model_dec_list))
        # optimizer.zero_grad()
        device = x.device
        pseudo_lab = torch.from_numpy(self.four_predict_map.copy()).float().to(device) 
        size_b,size_c,size_w,size_h = pseudo_lab.shape 

        eara1 = self.aux_seg_1 * pseudo_lab
        eara2 = self.aux_seg_2 * pseudo_lab
        eara3 = self.aux_seg_3 * pseudo_lab
        eara4 = self.aux_seg_4 * pseudo_lab
        diceloss = self.segloss(eara4,pseudo_lab,False)+self.segloss(eara3,pseudo_lab,False) +self.segloss(eara2,pseudo_lab,False)+self.segloss(eara1,pseudo_lab,False)
        diceloss = diceloss / 4.0
        
        mean_map =  (self.aux_seg_4+self.aux_seg_3 +self.aux_seg_2+self.aux_seg_1) / 4.0
        # mean_map_entropyloss = -(mean_map * torch.log2(mean_map + 1e-10)).sum() / (4.0 * size_c * size_b*size_w*size_h)
        # all_loss = diceloss + mean_map_entropyloss
        # all_loss.backward()
        # optimizer.step()

        # diceloss = diceloss.item()
        # mean_map_entropyloss = mean_map_entropyloss.item()
        # print('deiceloss:        ',diceloss)
        # print('mean_entropyloss: ',mean_map_entropyloss)
        return mean_map

    @torch.no_grad()  # ensure grads in possible no grad context for testing
    def save_nii(self):
        pred_aux1 = self.aux_seg_1.cpu().detach().numpy()
        pred_aux2 = self.aux_seg_2.cpu().detach().numpy()
        pred_aux3 = self.aux_seg_3.cpu().detach().numpy()
        pred_aux4 = self.aux_seg_4.cpu().detach().numpy()
        self.four_predict_map = (pred_aux3+pred_aux4+pred_aux2+pred_aux1)/4.0
        self.four_predict_map[self.four_predict_map > self.pl_threshold] = 1
        self.four_predict_map[self.four_predict_map < 1] = 0
        B,D,W,H = self.four_predict_map.shape
        for i in range(D):
            self.four_predict_map[:,i,:,:] = get_largest_component(self.four_predict_map[:,i,:,:])
    def get_pseudo_label(self, enc, dec_list, x, mult=4):
        A_1 = rotate_single_with_label(x, 1)
        A_2 = rotate_single_with_label(x, 2)
        A_3 = rotate_single_with_label(x, 3)
        blocks1, latent_A1 = enc(A_1)
        blocks2, latent_A2 = enc(A_2)
        blocks3, latent_A3 = enc(A_3)
        blocks4, latent_A4 = enc(x)
        
        self.aux_seg_1 = dec_list[0](latent_A1, blocks1)[0].softmax(1)
        self.aux_seg_2 = dec_list[1](latent_A2, blocks2)[0].softmax(1)
        self.aux_seg_3 = dec_list[2](latent_A3, blocks3)[0].softmax(1)
        self.aux_seg_4 = dec_list[3](latent_A4, blocks4)[0].softmax(1)
        self.aux_seg_1 = rotate_single_with_label(self.aux_seg_1,3)
        self.aux_seg_2 = rotate_single_with_label(self.aux_seg_2,2)
        self.aux_seg_3 = rotate_single_with_label(self.aux_seg_3,1)
    def get_pseudo_label__(self, enc, dec_list, x, mult=4):
        preditions_augs = []
        blocks1, latent_A1 = enc(x)
        output,last_x = dec_list[mult-1](latent_A1, blocks1)
        output = F.softmax(output, dim=1)
        preditions_augs.append(output)
        for i in range(mult-1):
            x_aug, rotate_affine = self.randomRotate(x)
            x_aug, vflip_affine = self.randomVerticalFlip(x_aug)
            x_aug, hflip_affine = self.randomHorizontalFlip(x_aug)

            # x_aug, hflip_affine = self.randomHorizontalFlip(x)
            # x_aug, crop_affine  = self.randomResizeCrop(x_aug)
            # get label on x_aug
            blocks1, latent_A1 = enc(x_aug)
            pred_aug,last_x = dec_list[i](latent_A1, blocks1)
            pred_aug = F.softmax(pred_aug, dim=1)
            pred_aug = apply_invert_affine(pred_aug, rotate_affine)
            pred_aug = apply_invert_affine(pred_aug, vflip_affine)
            pred_aug = apply_invert_affine(pred_aug, hflip_affine)

            preditions_augs.append(pred_aug)

        return preditions_augs
    def randomHorizontalFlip(self, x):
        affine = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3).repeat(x.size(0), 1, 1)
        horizontal_flip = torch.tensor([-1, 0, 0, 0, 1, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3)
        # randomly flip some of the images in the batch
        mask = (torch.rand(x.size(0), device=x.device) > 0.5)
        affine[mask] = affine[mask] * horizontal_flip
        x = apply_affine(x, affine)
        return x, affine.detach()
    def randomRotate(self, x):
        affine = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3).repeat(x.size(0), 1, 1)
        rotation = torch.tensor([0, -1, 0, 1, 0, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3)
        # randomly flip some of the images in the batch
        mask = (torch.rand(x.size(0), device=x.device) > 0.5)
        affine[mask] = rotation.repeat(mask.sum(), 1, 1)

        x = apply_affine(x, affine)
        
        return x, affine.detach()
    def randomVerticalFlip(self, x):
        affine = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3).repeat(x.size(0), 1, 1)
        vertical_flip = torch.tensor([1, 0, 0, 0, -1, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3)
        # randomly flip some of the images in the batch
        
        mask = (torch.rand(x.size(0), device=x.device) > 0.5)
        affine[mask] = affine[mask] * vertical_flip
        x = apply_affine(x, affine)
        return x, affine.detach()
    def randomResizeCrop(self, x):
        # TODO: Investigate different scale for x and y
        delta_scale_x = 0.2
        delta_scale_y = 0.2

        scale_matrix_x = torch.tensor([1, 0, 0, 0, 0, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3)
        scale_matrix_y = torch.tensor([0, 0, 0, 0, 1, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3)

        translation_matrix_x = torch.tensor([0, 0, 1, 0, 0, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3)
        translation_matrix_y = torch.tensor([0, 0, 0, 0, 0, 1], dtype=torch.float32, device=x.device).reshape(-1, 2, 3)

        delta_x = 0.5 * delta_scale_x * (2*torch.rand(x.size(0), 1, 1, device=x.device) - 1.0)
        delta_y = 0.5 * delta_scale_y * (2*torch.rand(x.size(0), 1, 1, device=x.device) -1.0)

        random_affine = (1 - delta_scale_x) * scale_matrix_x + (1 - delta_scale_y) * scale_matrix_y +\
                    delta_x * translation_matrix_x + \
                    delta_y * translation_matrix_y

        x = apply_affine(x, random_affine)
        return x, random_affine.detach()
    def infer(self, x):
        # outputs = self.model(x)
        # Teacher Prediction
        # anchor_prob = torch.nn.functional.softmax(self.model_anchor(x), dim=1).max(1)[0]
        anchor_prob = torch.nn.functional.softmax(self.model_anchor(x), dim=1).max()
       
        standard_ema = self.model_ema(x)
        
        outputs_ema = standard_ema
        
        return outputs_ema


@torch.jit.script
def softmax_entropy(x, x_ema):# -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    n, c, h, w =  x.shape
    entropy1 = -(x_ema.softmax(1) * x.log_softmax(1)).sum() / \
        (n * h * w * torch.log2(torch.tensor(c, dtype=torch.float)))
    return entropy1

def collect_params(model):
    """Collect all trainable parameters.

    Walk the model's modules and collect all parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if True:#isinstance(m, nn.BatchNorm2d): collect all 
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
                    print(nm, np)
    return params, names

def apply_invert_affine(x, affine):
    # affine shape should be batch x 2 x 3
    # x shape should be batch x ch x h x w

    # get homomorphic transform
    H = torch.nn.functional.pad(affine, [0, 0, 0, 1], "constant", value=0.0)
    H[..., -1, -1] += 1.0

    inv_H = torch.inverse(H)
    inv_affine = inv_H[:, :2, :3]

    grid = torch.nn.functional.affine_grid(inv_affine, x.size(), align_corners=False)
    x = torch.nn.functional.grid_sample(x, grid, padding_mode="reflection", align_corners=False)
    return x

def apply_affine(x, affine):
    grid = torch.nn.functional.affine_grid(affine, x.size(), align_corners=False)
    out = torch.nn.functional.grid_sample(x, grid, padding_mode="reflection", align_corners=False)
    return out

def copy_model_and_optimizer(model_enc, model_dec_list, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_enc_state = deepcopy(model_enc.state_dict())
    model_dec_state = []
    for i in range(len(model_dec_list)):
        model_dec_state.append( deepcopy(model_dec_list[i].state_dict()))
    optimizer_state = deepcopy(optimizer.state_dict())

    return model_enc_state, model_dec_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


from scipy import ndimage
import numpy as np
def get_largest_component(image):
    """
    get the largest component from 2D or 3D binary image
    image: nd array
    """
    dim = len(image.shape)
    if(image.sum() == 0 ):
        # print('the largest component is null')
        return image
    if(dim == 2):
        s = ndimage.generate_binary_structure(2,1)
    elif(dim == 3):
        s = ndimage.generate_binary_structure(3,1)
    else:
        raise ValueError("the dimension number should be 2 or 3")
    labeled_array, numpatches = ndimage.label(image, s)
    sizes = ndimage.sum(image, labeled_array, range(1, numpatches + 1))
    max_label = np.where(sizes == sizes.max())[0] + 1
    output = np.asarray(labeled_array == max_label, np.uint8)
    return  output

def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"
class DiceLoss(nn.Module):
    def __init__(self,n_classes):
        super().__init__()
        self.n_classes = n_classes
        
    def one_hot_encode(self,input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            tmp = (input_tensor==i) * torch.ones_like(input_tensor)
            tensor_list.append(tmp)
        output_tensor = torch.cat(tensor_list,dim=1)
        return output_tensor.float()
    
    def forward(self,inputs,target,one_hot):
        x_shape = list(target.shape)
        if(len(x_shape) == 5):
            [N, C, D, H, W] = x_shape
            new_shape = [N*D, C, H, W]
            target = torch.transpose(target, 1, 2)
            target = torch.reshape(target, new_shape)

        if one_hot:
            target = self.one_hot_encode(target)
        assert inputs.shape == target.shape,'size must match'
        class_wise_dice = []
        loss = 0.0
        for i in range(self.n_classes):
            diceloss = dice_loss(inputs[:,i,:,:], target[:,i,:,:])
            class_wise_dice.append(diceloss)
            loss += diceloss
        return loss/self.n_classes
def dice_loss(predict,target):
    target = target.float()
    smooth = 1e-4
    intersect = torch.sum(predict*target)
    dice = (2 * intersect + smooth)/(torch.sum(target)+torch.sum(predict*predict)+smooth)
    loss = 1.0 - dice
    return loss
def tensor_rot_90(x):
    x_shape = list(x.shape)
    if(len(x_shape) == 4):
        return x.flip(3).transpose(2, 3)
    else:
	    return x.flip(2).transpose(1, 2)
def tensor_rot_180(x):
    x_shape = list(x.shape)
    if(len(x_shape) == 4):
        return x.flip(3).flip(2)
    else:
	    return x.flip(2).flip(1)
def tensor_flip_2(x):
    x_shape = list(x.shape)
    if(len(x_shape) == 4):
        return x.flip(2)
    else:
	    return x.flip(1)
def tensor_flip_3(x):
    x_shape = list(x.shape)
    if(len(x_shape) == 4):
        return x.flip(3)
    else:
	    return x.flip(2)

def tensor_rot_270(x):
    x_shape = list(x.shape)
    if(len(x_shape) == 4):
        return x.transpose(2, 3).flip(3)
    else:
        return x.transpose(1, 2).flip(2)
    
def rotate_single_random(img):
    x_shape = list(img.shape)
    if(len(x_shape) == 5):
        [N, C, D, H, W] = x_shape
        new_shape = [N*D, C, H, W]
        x = torch.transpose(img, 1, 2)
        img = torch.reshape(x, new_shape)
    label = np.random.randint(0, 4, 1)[0]
    if label == 1:
        img = tensor_rot_90(img)
    elif label == 2:
        img = tensor_rot_180(img)
    elif label == 3:
        img = tensor_rot_270(img)
    else:
        img = img
    return img,label

def rotate_single_with_label(img, label):
    if label == 1:
        img = tensor_rot_90(img)
    elif label == 2:
        img = tensor_rot_180(img)
    elif label == 3:
        img = tensor_rot_270(img)
    else:
        img = img
    return img

def random_rotate(A,A_gt):
    target_ssh = np.random.randint(0, 8, 1)[0]
    A = rotate_single_with_label(A, target_ssh)
    A_gt = rotate_single_with_label(A_gt, target_ssh)
    return A,A_gt

def rotate_4(img):
    # target_ssh = np.random.randint(0, 4, 1)[0]
    A_1 = rotate_single_with_label(img, 1)
    A_2 = rotate_single_with_label(img, 2)
    A_3 = rotate_single_with_label(img, 3)
    return A_1,A_2,A_3