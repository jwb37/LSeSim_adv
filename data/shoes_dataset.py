import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util
import torchvision.transforms as transforms

import re
from pathlib import Path

ImageSuffices = [
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.tif', '.tiff'
]


class ShoesDataset(BaseDataset):
    """
    Dataset class designed to load the ShoeV2 dataset
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.re_strip_end = re.compile( r"(.*)_\d+(\.(?:png|jpg|jpeg))$" )

        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted([p for p in Path(self.dir_A).iterdir() if p.suffix.casefold() in ImageSuffices], key=lambda p: p.name)
        self.B_paths = [Path(self.dir_B) / self.get_B_fname(A_path.name) for A_path in self.A_paths]

        # Batch loader throws a wobbly if these contain path objects
        self.A_paths = [str(path) for path in self.A_paths]
        self.B_paths = [str(path) for path in self.B_paths]

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        # apply image transformation
        # For FastCUT mode, if in finetuning phase (learning rate is decaying)
        # do not perform resize-crop data augmentation of CycleGAN.
        # print('current_epoch', self.current_epoch)
        self.transform = get_transform(opt, convert=False)
        if self.opt.isTrain and opt.augment:
            self.transform_aug = transforms.Compose(
                [transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3),
                 transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]
            )
        else:
            self.transform_aug = None
        self.transform_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    def get_B_fname(self, fname):
        m = self.re_strip_end.match(fname)
        if not m:
            return fname
        return ''.join(m.group(1,2))


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        A_pil = self.transform(A_img)
        B_pil = self.transform(B_img)
        A = self.transform_tensor(A_pil)
        B = self.transform_tensor(B_pil)
        if self.opt.isTrain and self.transform_aug is not None:
            A_aug = self.transform_aug(A_pil)
            B_aug = self.transform_aug(B_pil)
            return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'A_aug': A_aug, 'B_aug': B_aug}
        else:
            return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
