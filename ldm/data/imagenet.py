import clip
import cv2
import glob
import os
import pickle
import shutil
import tarfile
import torch
import yaml
import PIL

import albumentations as al
import numpy as np
import taming.data.utils as tdu
import torchvision.transforms.functional as TF

from abc import ABC, abstractmethod
from einops import rearrange
from omegaconf import OmegaConf
from pathlib import Path
from functools import partial
from random import random
from tqdm import tqdm
from torch.utils.data import Dataset, Subset
from PIL import Image

from taming.data.imagenet import (
    str_to_indices,
    give_synsets_from_indices,
    download,
    retrieve,
)
from taming.data.imagenet import ImagePaths
from ldm.modules.image_degradation import (
    degradation_fn_bsr,
    degradation_fn_bsr_light,
)
from typing import Dict, List


def synset2idx(path_to_yaml: str = "data/index_synset.yaml"):
    with open(path_to_yaml) as f:
        di2s = yaml.load(f)
    return dict((v,k) for k,v in di2s.items())


class ImageNetBase(Dataset):

    def __init__(self, config:Dict=None):
        self.config = config or OmegaConf.create()
        if not type(self.config)==dict:
            self.config = OmegaConf.to_container(self.config)
        self.keep_orig_class_label = self.config.get("keep_orig_class_label", False)
        self.process_images = True  # if False we skip loading & processing images and self.data contains filepaths
        self._prepare()
        self._prepare_synset_to_human()
        self._prepare_idx_to_synset()
        self._prepare_human_to_integer_label()
        self._load()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i:int):
        return self.data[i]

    def _prepare(self):
        raise NotImplementedError()

    def _filter_relpaths(self, relpaths: List):
        ignore = set([
            "n06596364_9591.JPEG",
        ])
        relpaths = [rpath for rpath in relpaths if not rpath.split("/")[-1] in ignore]
        if "sub_indices" in self.config:
            indices = str_to_indices(self.config["sub_indices"])
            synsets = give_synsets_from_indices(indices, path_to_yaml=self.idx2syn)  # returns a list of strings
            self.synset2idx = synset2idx(path_to_yaml=self.idx2syn)
            files = []
            for rpath in relpaths:
                syn = rpath.split("/")[0]
                if syn in synsets:
                    files.append(rpath)
            return files
        else:
            return relpaths

    def _prepare_synset_to_human(self):
        SIZE = 2655750
        URL = "https://heibox.uni-heidelberg.de/f/9f28e956cd304264bb82/?dl=1"
        self.human_dict = os.path.join(self.root, "synset_human.txt")
        if (not os.path.exists(self.human_dict) or not os.path.getsize(self.human_dict)==SIZE):
            download(URL, self.human_dict)

    def _prepare_idx_to_synset(self):
        URL = "https://heibox.uni-heidelberg.de/f/d835d5b6ceda4d3aa910/?dl=1"
        self.idx2syn = os.path.join(self.root, "index_synset.yaml")
        if (not os.path.exists(self.idx2syn)):
            download(URL, self.idx2syn)

    def _prepare_human_to_integer_label(self):
        URL = "https://heibox.uni-heidelberg.de/f/2362b797d5be43b883f6/?dl=1"
        self.human2integer = os.path.join(self.root, "imagenet1000_clsidx_to_labels.txt")
        if (not os.path.exists(self.human2integer)):
            download(URL, self.human2integer)
        with open(self.human2integer, "r") as f:
            lines = f.read().splitlines()
            assert len(lines) == 1000
            self.human2integer_dict = dict()
            for line in lines:
                value, key = line.split(":")
                self.human2integer_dict[key] = int(value)

    def _load(self):
        with open(self.txt_filelist, "r") as f:
            self.relpaths = f.read().splitlines()
            l1 = len(self.relpaths)
            self.relpaths = self._filter_relpaths(self.relpaths)
            print("Removed {} files from filelist during filtering.".format(l1 - len(self.relpaths)))

        self.synsets = [p.split("/")[0] for p in self.relpaths]
        self.abspaths = [os.path.join(self.datadir, p) for p in self.relpaths]

        unique_synsets = np.unique(self.synsets)
        class_dict = dict((synset, i) for i, synset in enumerate(unique_synsets))
        if not self.keep_orig_class_label:
            self.class_labels = [class_dict[s] for s in self.synsets]
        else:
            self.class_labels = [self.synset2idx[s] for s in self.synsets]

        with open(self.human_dict, "r") as f:
            human_dict = f.read().splitlines()
            human_dict = dict(line.split(maxsplit=1) for line in human_dict)

        self.human_labels = [human_dict[s] for s in self.synsets]

        labels = {
            "relpath": np.array(self.relpaths),
            "synsets": np.array(self.synsets),
            "class_label": np.array(self.class_labels),
            "human_label": np.array(self.human_labels),
        }

        if self.process_images:
            self.size = retrieve(self.config, "size", default=256)
            self.data = ImagePaths(
                self.abspaths,
                labels=labels,
                size=self.size,
                random_crop=self.random_crop,)
        else:
            self.data = self.abspaths


class ImageNetTrain(ImageNetBase):

    NAME = "ILSVRC2012_train"
    URL = "http://www.image-net.org/challenges/LSVRC/2012/"
    AT_HASH = "a306397ccf9c2ead27155983c254227c0fd938e2"
    FILES = ["ILSVRC2012_img_train.tar",]
    SIZES = [147897477120,]

    def __init__(self, process_images:bool=True, data_root:str=None, **kwargs):
        self.process_images = process_images
        self.data_root = data_root
        super().__init__(**kwargs)

    def _prepare(self):
        if self.data_root:
            self.root = os.path.join(self.data_root, self.NAME)
        else:
            cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
            self.root = os.path.join(cachedir, "autoencoders/data", self.NAME)

        self.datadir = os.path.join(self.root, "data")
        self.txt_filelist = os.path.join(self.root, "filelist.txt")
        self.expected_length = 1281167
        self.random_crop = retrieve(self.config, "ImageNetTrain/random_crop", default=True)
        if not tdu.is_prepared(self.root):
            # prep
            print("Preparing dataset {} in {}".format(self.NAME, self.root))

            datadir = self.datadir
            if not os.path.exists(datadir):
                path = os.path.join(self.root, self.FILES[0])
                if not os.path.exists(path) or not os.path.getsize(path)==self.SIZES[0]:
                    import academictorrents as at
                    atpath = at.get(self.AT_HASH, datastore=self.root)
                    assert atpath == path

                print("Extracting {} to {}".format(path, datadir))
                os.makedirs(datadir, exist_ok=True)
                with tarfile.open(path, "r:") as tar:
                    tar.extractall(path=datadir)

                print("Extracting sub-tars.")
                subpaths = sorted(glob.glob(os.path.join(datadir, "*.tar")))
                for subpath in tqdm(subpaths):
                    subdir = subpath[:-len(".tar")]
                    os.makedirs(subdir, exist_ok=True)
                    with tarfile.open(subpath, "r:") as tar:
                        tar.extractall(path=subdir)

            filelist = glob.glob(os.path.join(datadir, "**", "*.JPEG"))
            filelist = [os.path.relpath(p, start=datadir) for p in filelist]
            filelist = sorted(filelist)
            filelist = "\n".join(filelist)+"\n"
            with open(self.txt_filelist, "w") as f:
                f.write(filelist)

            tdu.mark_prepared(self.root)


class ImageNetValidation(ImageNetBase):

    NAME = "ILSVRC2012_validation"
    URL = "http://www.image-net.org/challenges/LSVRC/2012/"
    AT_HASH = "5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5"
    VS_URL = "https://heibox.uni-heidelberg.de/f/3e0f6e9c624e45f2bd73/?dl=1"
    FILES = ["ILSVRC2012_img_val.tar", "validation_synset.txt",]
    SIZES = [6744924160, 1950000,]

    def __init__(self, process_images: bool = True, data_root: str = None, **kwargs):
        self.data_root = data_root
        self.process_images = process_images
        super().__init__(**kwargs)

    def _prepare(self):
        if self.data_root:
            self.root = os.path.join(self.data_root, self.NAME)
        else:
            cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
            self.root = os.path.join(cachedir, "autoencoders/data", self.NAME)
        self.datadir = os.path.join(self.root, "data")
        self.txt_filelist = os.path.join(self.root, "filelist.txt")
        self.expected_length = 50000
        self.random_crop = retrieve(self.config, "ImageNetValidation/random_crop", default=False)
        if not tdu.is_prepared(self.root):
            # prep
            print("Preparing dataset {} in {}".format(self.NAME, self.root))

            datadir = self.datadir
            if not os.path.exists(datadir):
                path = os.path.join(self.root, self.FILES[0])
                if not os.path.exists(path) or not os.path.getsize(path)==self.SIZES[0]:
                    import academictorrents as at
                    atpath = at.get(self.AT_HASH, datastore=self.root)
                    assert atpath == path

                print("Extracting {} to {}".format(path, datadir))
                os.makedirs(datadir, exist_ok=True)
                with tarfile.open(path, "r:") as tar:
                    tar.extractall(path=datadir)

                vspath = os.path.join(self.root, self.FILES[1])
                if not os.path.exists(vspath) or not os.path.getsize(vspath)==self.SIZES[1]:
                    download(self.VS_URL, vspath)

                with open(vspath, "r") as f:
                    synset_dict = f.read().splitlines()
                    synset_dict = dict(line.split() for line in synset_dict)

                print("Reorganizing into synset folders")
                synsets = np.unique(list(synset_dict.values()))
                for s in synsets:
                    os.makedirs(os.path.join(datadir, s), exist_ok=True)
                for k, v in synset_dict.items():
                    src = os.path.join(datadir, k)
                    dst = os.path.join(datadir, v)
                    shutil.move(src, dst)

            filelist = glob.glob(os.path.join(datadir, "**", "*.JPEG"))
            filelist = [os.path.relpath(p, start=datadir) for p in filelist]
            filelist = sorted(filelist)
            filelist = "\n".join(filelist)+"\n"
            with open(self.txt_filelist, "w") as f:
                f.write(filelist)

            tdu.mark_prepared(self.root)


class ImageNetSR(Dataset):

    def __init__(
        self,
        size:int=None,
        degradation:str=None,
        downscale_f:int=4,
        min_crop_f:float=0.5,
        max_crop_f:float=1.,
        random_crop:bool=True,
    ):
        """
        Imagenet Superresolution Dataloader
        Performs following ops in order:
        1.  crops a crop of size s from image either as random or center crop
        2.  resizes crop to size with cv2.area_interpolation
        3.  degrades resized crop with degradation_fn

        :param size: resizing to size after cropping
        :param degradation: degradation_fn, e.g. cv_bicubic or bsrgan_light
        :param downscale_f: Low Resolution Downsample factor
        :param min_crop_f: determines crop size s,
          where s = c * min_img_side_len with c sampled from interval (min_crop_f, max_crop_f)
        :param max_crop_f: ""
        :param data_root:
        :param random_crop:
        """
        self.base = self.get_base()
        assert size
        assert (size / downscale_f).is_integer()
        self.size = size
        self.LR_size = int(size / downscale_f)
        self.min_crop_f = min_crop_f
        self.max_crop_f = max_crop_f
        assert(max_crop_f <= 1.)
        self.center_crop = not random_crop

        self.image_rescaler = al.SmallestMaxSize(max_size=size, interpolation=cv2.INTER_AREA)

        self.pil_interpolation = False # gets reset later if incase interp_op is from pillow

        if degradation == "bsrgan":
            self.degradation_process = partial(degradation_fn_bsr, sf=downscale_f)

        elif degradation == "bsrgan_light":
            self.degradation_process = partial(degradation_fn_bsr_light, sf=downscale_f)

        else:
            interpolation_fn = {
                "cv_nearest": cv2.INTER_NEAREST,
                "cv_bilinear": cv2.INTER_LINEAR,
                "cv_bicubic": cv2.INTER_CUBIC,
                "cv_area": cv2.INTER_AREA,
                "cv_lanczos": cv2.INTER_LANCZOS4,
                "pil_nearest": PIL.Image.NEAREST,
                "pil_bilinear": PIL.Image.BILINEAR,
                "pil_bicubic": PIL.Image.BICUBIC,
                "pil_box": PIL.Image.BOX,
                "pil_hamming": PIL.Image.HAMMING,
                "pil_lanczos": PIL.Image.LANCZOS,
            }[degradation]

            self.pil_interpolation = degradation.startswith("pil_")

            if self.pil_interpolation:
                self.degradation_process = partial(TF.resize, size=self.LR_size, interpolation=interpolation_fn)

            else:
                self.degradation_process = al.SmallestMaxSize(
                    max_size=self.LR_size,
                    interpolation=interpolation_fn)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i:int):
        example = self.base[i]
        image = Image.open(example["file_path_"])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        image = np.array(image).astype(np.uint8)

        min_side_len = min(image.shape[:2])
        crop_side_len = min_side_len * np.random.uniform(self.min_crop_f, self.max_crop_f, size=None)
        crop_side_len = int(crop_side_len)

        if self.center_crop:
            self.cropper = al.CenterCrop(height=crop_side_len, width=crop_side_len)

        else:
            self.cropper = al.RandomCrop(height=crop_side_len, width=crop_side_len)

        image = self.cropper(image=image)["image"]
        image = self.image_rescaler(image=image)["image"]

        if self.pil_interpolation:
            image_pil = PIL.Image.fromarray(image)
            LR_image = self.degradation_process(image_pil)
            LR_image = np.array(LR_image).astype(np.uint8)

        else:
            LR_image = self.degradation_process(image=image)["image"]

        example["image"] = (image/127.5-1.0).astype(np.float32)
        example["LR_image"] = (LR_image/127.5-1.0).astype(np.float32)

        return example


class ImageNetSRTrain(ImageNetSR):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_base(self):
        with open("data/imagenet_train_hr_indices.p", "rb") as f:
            indices = pickle.load(f)
        dset = ImageNetTrain(process_images=False,)
        return Subset(dset, indices)


class ImageNetSRValidation(ImageNetSR):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_base(self):
        with open("data/imagenet_val_hr_indices.p", "rb") as f:
            indices = pickle.load(f)
        dset = ImageNetValidation(process_images=False,)
        return Subset(dset, indices)


class ImageNetPatchInpaint(Dataset, ABC):

    def __init__(
        self,
        size:int,
        min_patch_f: float,
        max_patch_f: float,
        min_crop_f:float,
        max_crop_f:float,
        random_crop:bool,
        clip_model_name:str,
        cond_dropout:float=None,
        cond_noise:float=None,
        cond_style:float=None,
    ):
        super().__init__()
        self.img_rescler = al.SmallestMaxSize(max_size=size, interpolation=cv2.INTER_AREA)
        self.min_crop_f = min_crop_f
        self.max_crop_f = max_crop_f
        self.min_patch_f = min_patch_f
        self.max_patch_f = max_patch_f
        self.random_crop = random_crop
        self.size = size
        self.base = self.get_base()
        self.cond_dropout = cond_dropout
        self.cond_noise = cond_noise
        self.cond_style = cond_style
        n2r = {
            'ViT-B/32': 224,
            'ViT-L/14': 224,
        }
        res = n2r[clip_model_name]
        self.preprocess = clip.clip._transform(res)

    @abstractmethod
    def get_base(self) -> ImageNetBase:
        '''get the data'''

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i:int):
        item = self.base[i]
        img = Image.open(item['file_path_'])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = np.array(img).astype(np.uint8)

        min_crop_f, max_crop_f = self.min_crop_f, self.max_crop_f
        min_side_len = min(img.shape[:2])
        crop_side_len = min_side_len * np.random.uniform(min_crop_f, max_crop_f, size=None)
        crop_side_len = int(crop_side_len)

        crop_cls = al.RandomCrop if self.random_crop else al.CenterCrop
        self.cropper = crop_cls(height=crop_side_len, width=crop_side_len)

        img = self.cropper(image=img)['image']
        img = self.img_rescler(image=img)['image']

        min_patch_f, max_patch_f = self.min_patch_f, self.max_patch_f
        patch_height = self.size * np.random.uniform(min_patch_f, max_patch_f, size=None)
        patch_height = int(patch_height)
        patch_width = self.size * np.random.uniform(min_patch_f, max_patch_f, size=None)
        patch_width = int(patch_width)
        x1, y1, x2, y2 = al.get_random_crop_coords(
            self.size, self.size, patch_height, patch_width, random(), random()
        )
        patch = img[y1:y2, x1:x2]
        if random() < self.cond_style:
            j = int(random() * len(self))
            item_tgt = self.base[j]
            img_tgt = Image.open(item_tgt['file_path_']).convert('RGB')
            img_tgt = np.array(img_tgt).astype(np.uint8)
            style = al.FDA([img_tgt], p=1, read_fn=lambda x: x)
            patch = style(image=patch)['image']
        patch_img = Image.fromarray(patch.astype(np.uint8))
        patch = self.preprocess(patch_img)
        if random() < self.cond_dropout:
            patch.fill_(0.)
        mask_img = np.copy(img)
        mask_img[y1:y2, x1:x2] = 0

        mask = np.zeros((self.size, self.size, 1), dtype=np.float32)
        mask[y1:y2, x1:x2] = 1
        mask = mask*2-1.0

        img = (img/127.5-1.0).astype(np.float32)

        item = {}
        item['image'] = img
        item['masked_image'] = mask_img
        item['mask'] = mask
        item['patch'] = patch
        return item


class ImageNetPatchInpaintTrain(ImageNetPatchInpaint):

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)

    def get_base(self) -> ImageNetBase:
        return ImageNetTrain(process_images=False)


class ImageNetPatchInpaintValidation(ImageNetPatchInpaint):

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)

    def get_base(self) -> ImageNetBase:
        return ImageNetValidation(process_images=False)


class ImageNetPatchInpaintTrainToy(ImageNetPatchInpaint):

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)

    def get_base(self) -> ImageNetBase:
        total = 1024 * 1024
        root_dir = Path(__file__).parent.parent.parent
        sample_path = root_dir / 'data' / 'super_resolution' / '0059.png'
        return [{
            'file_path_': sample_path
        } for _ in range(total)]


class ImageNetPatchInpaintValidationToy(ImageNetPatchInpaint):

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)

    def get_base(self) -> ImageNetBase:
        total = 64 * 1024
        root_dir = Path(__file__).parent.parent.parent
        sample_path = root_dir / 'data' / 'super_resolution' / '0059.png'
        return [{
            'file_path_': sample_path
        } for _ in range(total)]

