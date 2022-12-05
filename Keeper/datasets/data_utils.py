import cv2
import random
import numpy as np
import os.path as osp
from glob import glob
from natsort import natsorted

import torch

from toolkits.utils import scandir


__all__ = ['image_show', 'imfrombytes', 'img2tensor',
           'paired_paths_from_lmdb',
           'paired_paths_from_meta_info_file',
           'paired_paths_from_folder']


def imfrombytes(content, flag='color', float32=False):
    """Read an image from bytes.

    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.
        float32 (bool): Whether to change to float32., If True, will also norm
            to [0, 1]. Default: False.

    Returns:
        ndarray: Loaded image array.
    """
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {
        'color': cv2.IMREAD_COLOR,
        'grayscale': cv2.IMREAD_GRAYSCALE,
        'unchanged': cv2.IMREAD_UNCHANGED
    }
    if img_np is None:
        raise Exception('Empty Image, please check.')
    img = cv2.imdecode(img_np, imread_flags[flag])
    if float32:
        img = img.astype(np.float32) / 255.
    return img


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def paired_paths_from_lmdb(folders, keys, data_range=None):
    """Generate paired paths from lmdb files.

    Contents of lmdb. Taking the `lq.lmdb` for example, the file structure is:

    lq.lmdb
    ├── data.mdb
    ├── lock.mdb
    ├── meta_info.txt

    The data.mdb and lock.mdb are standard lmdb files and you can refer to
    https://lmdb.readthedocs.io/en/release/ for more details.

    The meta_info.txt is a specified txt file to record the meta information
    of our datasets. It will be automatically created when preparing
    datasets by our provided dataset tools.
    Each line in the txt file records
    1)image name (with extension),
    2)image shape,
    3)compression level, separated by a white space.
    Example: `baboon.png (120,125,3) 1`

    We use the image name without extension as the lmdb key.
    Note that we use the same key for the corresponding lq and gt images.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
            Note that this key is different from lmdb keys.
        data_range(str): A string that denotes the filename range of the dataset,
            e.g. '0001-0800'
    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 2, (
        'The len of folders should be 2 with [input_folder, gt_folder]. '
        f'But got {len(folders)}')
    assert len(keys) == 2, (
        'The len of keys should be 2 with [input_key, gt_key]. '
        f'But got {len(keys)}')
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    if not (input_folder.endswith('.lmdb') and gt_folder.endswith('.lmdb')):
        raise ValueError(
            f'{input_key} folder and {gt_key} folder should both in lmdb '
            f'formats. But received {input_key}: {input_folder}; '
            f'{gt_key}: {gt_folder}')
    # ensure that the two meta_info files are the same
    with open(osp.join(input_folder, 'meta_info.txt')) as fin:
        input_lmdb_keys = [line.split('.')[0] for line in fin]
    with open(osp.join(gt_folder, 'meta_info.txt')) as fin:
        gt_lmdb_keys = [line.split('.')[0] for line in fin]
    if set(input_lmdb_keys) != set(gt_lmdb_keys):
        raise ValueError(
            f'Keys in {input_key}_folder and {gt_key}_folder are different.')
    else:
        paths = []
        if data_range is not None:
            data_range = data_range.split('-')
            data_start, data_end = data_range[0], data_range[1]
            data_format = len(data_start)
            for lmdb_key in sorted(input_lmdb_keys):
                if int(lmdb_key[:data_format]) <= int(data_end) and \
                        int(lmdb_key[:data_format]) >= int(data_start):
                    paths.append(
                        dict([(f'{input_key}_path', lmdb_key),
                            (f'{gt_key}_path', lmdb_key)]))
        else:
            for lmdb_key in sorted(input_lmdb_keys):
                paths.append(
                    dict([(f'{input_key}_path', lmdb_key),
                        (f'{gt_key}_path', lmdb_key)]))

        return paths


def paired_paths_from_meta_info_file(folders, keys, meta_info_file,
                                     filename_tmpl, data_range=None):
    """Generate paired paths from an meta information file.

    Each line in the meta information file contains the image names and
    image shape (usually for gt), separated by a white space.

    Example of an meta information file:
    ```
    0001_s001.png (480,480,3)
    0001_s002.png (480,480,3)
    ```

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
        meta_info_file (str): Path to the meta information file.
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.
        data_range(str): A string that denotes the filename range of the dataset,
            e.g. '0001-0800'

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 2, (
        'The len of folders should be 2 with [input_folder, gt_folder]. '
        f'But got {len(folders)}')
    assert len(keys) == 2, (
        'The len of keys should be 2 with [input_key, gt_key]. '
        f'But got {len(keys)}')
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    with open(meta_info_file, 'r') as fin:
        gt_names = [line.split(' ')[0] for line in fin]

    paths = []
    if data_range is None:
        for gt_name in gt_names:
            basename, ext = osp.splitext(osp.basename(gt_name))
            input_name = f'{filename_tmpl.format(basename)}{ext}'
            input_path = osp.join(input_folder, input_name)
            gt_path = osp.join(gt_folder, gt_name)
            paths.append(
                dict([(f'{input_key}_path', input_path),
                    (f'{gt_key}_path', gt_path)]))
    else:
        data_range = data_range.split('-')
        data_start, data_end = data_range[0], data_range[1]
        data_format = len(data_start)
        for gt_name in gt_names:
            basename, ext = osp.splitext(osp.basename(gt_name))
            input_name = f'{filename_tmpl.format(basename)}{ext}'
            input_path = osp.join(input_folder, input_name)
            gt_path = osp.join(gt_folder, gt_name)
            if int(input_name[:data_format]) <= int(data_end) and \
                    int(input_name[:data_format]) >= int(data_start):
                paths.append(
                    dict([(f'{input_key}_path', input_path),
                        (f'{gt_key}_path', gt_path)]))

    return paths



def paired_paths_from_folder(folders, keys, filename_tmpl, data_range=None):
    """Generate paired paths from folders.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.
        data_range(str): A string that denotes the filename range of the dataset,
            e.g. '0001-0800'

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 2, (
        'The len of folders should be 2 with [input_folder, gt_folder]. '
        f'But got {len(folders)}')
    assert len(keys) == 2, (
        'The len of keys should be 2 with [input_key, gt_key]. '
        f'But got {len(keys)}')
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    input_paths = natsorted(list(scandir(input_folder)))
    gt_paths = natsorted(list(scandir(gt_folder)))
    assert len(input_paths) == len(gt_paths), (
        f'{input_key} and {gt_key} datasets have different number of images: '
        f'{len(input_paths)}, {len(gt_paths)}.')
    paths = []
    if data_range is None:
        for idx in range(len(gt_paths)):
            gt_path = gt_paths[idx]
            basename, ext = osp.splitext(osp.basename(gt_path))
            input_path = input_paths[idx]
            basename_input, ext_input = osp.splitext(osp.basename(input_path))
            input_name = f'{filename_tmpl.format(basename)}{ext_input}'
            input_path = osp.join(input_folder, input_name)
            assert input_name in input_paths, (f'{input_name} is not in '
                                               f'{input_key}_paths.')
            gt_path = osp.join(gt_folder, gt_path)
            paths.append(
                dict([(f'{input_key}_path', input_path),
                    (f'{gt_key}_path', gt_path)]))
    else:
        data_range = data_range.split('-')
        data_start, data_end = data_range[0], data_range[1]
        data_format = len(data_start)

        for idx in range(len(gt_paths)):
            gt_path = gt_paths[idx]
            basename, ext = osp.splitext(osp.basename(gt_path))
            input_path = input_paths[idx]
            basename_input, ext_input = osp.splitext(osp.basename(input_path))
            input_name = f'{filename_tmpl.format(basename)}{ext_input}'
            input_path = osp.join(input_folder, input_name)
            assert input_name in input_paths, (f'{input_name} is not in '
                                               f'{input_key}_paths.')
            gt_path = osp.join(gt_folder, gt_path)
            if int(input_name[:data_format]) >= int(data_start) and \
                    int(input_name[:data_format]) <= int(data_end):
                paths.append(
                    dict([(f'{input_key}_path', input_path),
                        (f'{gt_key}_path', gt_path)]))

    return paths



def image_show(x, title=None, cbar=False, figsize=None,
               converted_from_tensor=True, cmap=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    if converted_from_tensor:
        # in this case, x should be a np.ndarray
        x = x.transpose((2, 1, 0))
    plt.imshow(x, interpolation='nearest', cmap=cmap)
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()



