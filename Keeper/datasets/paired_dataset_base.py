from torch.utils import data as data
from torchvision.transforms.functional import normalize


from .data_utils import (
    img2tensor,
    imfrombytes,
    paired_paths_from_folder,
    paired_paths_from_lmdb,
    paired_paths_from_meta_info_file)

from .transforms import (
    augment,
    paired_random_crop,
    padding)

from .file_client import FileClient


__all__ = ['PairedImageDataset']



class PairedImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If cfg['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If cfg['io_backend'] != lmdb and cfg['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.

    Args:
        cfg (dict): Config for datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            phase (str): 'train' or 'val'.
    """

    def __init__(self, cfg):
        super(PairedImageDataset, self).__init__()
        self.cfg = cfg
        # file client (io backend)
        self.file_client = None
        self.io_backend_cfg = cfg['io_backend']
        self.mean = cfg.get('mean', None)
        self.std = cfg.get('std', None)

        self.filename_tmpl = cfg.get('filename_tmpl', '{}')
        self.gt_path, self.lq_path = cfg['gt_path'], cfg['lq_path']
        self.data_range = cfg.get('data_range', None)

        if self.io_backend_cfg['type'] == 'lmdb':
            self.io_backend_cfg['db_paths'] = [self.lq_path, self.gt_path]
            self.io_backend_cfg['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_path, self.gt_path], ['lq', 'gt'], self.data_range
            )
        elif self.cfg.get('meta_info_file', None) is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_path, self.gt_path], ['lq', 'gt'],
                self.cfg['meta_info_file'], self.filename_tmpl, self.data_range
            )
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_path, self.gt_path], ['lq', 'gt'],
                self.filename_tmpl, self.data_range
            )


    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_cfg.pop('type'), **self.io_backend_cfg
            )
        scale = self.cfg.get('scale', 1)

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception(f'gt path {gt_path} not working.')

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception(f'lq path {lq_path} not working.')

        # TODO: check the difference between cv2 img read and that by SIDD
        # augmentation for training
        if self.cfg['phase'] == 'train':
            gt_size = self.cfg['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)

            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.cfg['use_flip'], self.cfg['use_rot'])


        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)

