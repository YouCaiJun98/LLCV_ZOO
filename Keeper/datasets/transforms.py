import cv2
import random
import numpy as np



__all__ = ['crop_patch', 'crop_patches', 'data_augmentation',
           'Data_Augmentation', 'random_augmentation',
           'inverse_augmentation',
           'padding', 'augment', 'paired_random_crop']



def padding(img_lq, img_gt, gt_size):
    h, w, _ = img_lq.shape

    h_pad = max(0, gt_size - h)
    w_pad = max(0, gt_size - w)

    if h_pad == 0 and w_pad == 0:
        return img_lq, img_gt

    img_lq = cv2.copyMakeBorder(img_lq, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    img_gt = cv2.copyMakeBorder(img_gt, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    return img_lq, img_gt


def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False, vflip=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[ndarray]: Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.

    """
    hflip = hflip and random.random() < 0.5
    if vflip or rotation:
        vflip = random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
            if img.shape[2] == 6:
                img = img[:,:,[3,4,5,0,1,2]].copy() # swap left/right
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)
        else:
            return imgs


def paired_random_crop(img_gts, img_lqs, gt_patch_size, scale, gt_path):
    """Paired random crop.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    h_lq, w_lq, _ = img_lqs[0].shape
    h_gt, w_gt, _ = img_gts[0].shape
    lq_patch_size = gt_patch_size // scale

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(
            f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
            f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). '
                         f'Please remove {gt_path}.')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    img_lqs = [
        v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
        for v in img_lqs
    ]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gts = [
        v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]
        for v in img_gts
    ]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return img_gts, img_lqs








# randomly crop a patch from a single image.
def crop_patch(im, patch_size, gray=False, rnd=None):
    H = im.shape[0]
    W = im.shape[1]

    H_pad = patch_size - H if H < patch_size else 0
    W_pad = patch_size - W if W < patch_size else 0
    if H_pad != 0 or W_pad != 0:
        im = np.pad(im, (0, 0, W_pad, H_pad), 'reflect')
        H = im.shape[0]
        W = im.shape[1]

    if rnd:
        (rnd_H, rnd_W) = rnd
    else:
        rnd_H = random.randint(0, H-patch_size)
        rnd_W = random.randint(0, W-patch_size)
    pch = im[rnd_H:rnd_H + patch_size, rnd_W:rnd_W + patch_size] if gray \
        else im[rnd_H:rnd_H + patch_size, rnd_W:rnd_W + patch_size, :]
    return pch

# randomly crop a pair of image patches in np format.
def crop_patches(im1, im2, patch_size, gray=False):
    assert im1.shape == im2.shape, 'input images should be of the same size.'
    H, W = im1.shape[0], im1.shape[1]

    H_pad = patch_size - H if H < patch_size else 0
    W_pad = patch_size - W if W < patch_size else 0
    if H_pad != 0 or W_pad != 0:
        im1 = np.pad(im1, (0, 0, W_pad, H_pad), 'reflect')
        im2 = np.pad(im2, (0, 0, W_pad, H_pad), 'reflect')
    rand = (random.randint(0, H-patch_size),
            random.randint(0, W-patch_size))

    patch1 = crop_patch(im1, patch_size, gray, rand)
    patch2 = crop_patch(im2, patch_size, gray, rand)

    return patch1, patch2

# Needs TorchVision Implementation
def data_augmentation(image, mode):
    if mode == 0:
        # original
        out = image
    elif mode == 1:
        # flip up and down
        out = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    else:
        raise Exception('Invalid choice of image transformation')
    return out

# pytorch implemtation
import torch
def Data_Augmentation(img, mode):
    if   mode == 0:
        out = img
    elif mode == 1:
        out = img.flip(1)
    elif mode == 2:
        out = img.flip(2)
    elif mode == 3:
        out = torch.rot90(img, dims=(1,2))
    elif mode == 4:
        out = torch.rot90(img,dims=(1,2), k=2)
    elif mode == 5:
        out = torch.rot90(img,dims=(1,2), k=3)
    elif mode == 6:
        out = torch.rot90(img.flip(1),dims=(1,2))
    elif mode == 7:
        out = torch.rot90(img.flip(2),dims=(1,2))
    else:
        raise Exception('Invalid choice of image transformation')
    return out

def random_augmentation(*args):
    out = []
    flag_aug = random.randint(1,7)
    for data in args:
        out.append(data_augmentation(data, flag_aug).copy())
    return out

def inverse_augmentation(image, mode):
    if mode == 0:
        out = image
    elif mode == 1:
        out = np.flipud(image)
    elif mode == 2:
        out = np.rot90(image,k=-1)
    elif mode == 3:
        out = np.flipud(image)
        out = np.rot90(out, k=-1)
    elif mode == 4:
        out = np.rot90(image, k=-2)
    elif mode == 5:
        out = np.flipud(image)
        out = np.rot90(out, k=-2)
    elif mode == 6:
        out = np.rot90(image, k=-3)
    elif mode == 7:
        out = np.flipud(image)
        out = np.rot90(out, k=-3)
    else:
        raise Exception('Invalid choice of image transformation')
    return out

