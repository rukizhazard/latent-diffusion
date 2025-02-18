import cv2
import fire
import json
import os
import pickle
import PIL

import albumentations as al
import numpy as np
import torchvision.transforms.functional as ttf

from functools import partial
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import List

from PIL import Image, ImageDraw, ImageFont

from ldm.modules.image_degradation import (
    degradation_fn_bsr,
    degradation_fn_bsr_light,
)


def get_flist_items(names: List[str]):
    root_dir = Path(os.environ['AVID_ROOT_DIR'])
    examples = []
    for folder_name in names:
        img_dir = root_dir / folder_name
        flist_path = img_dir / 'flist_ocr.pkl'
        if not flist_path.exists():
            continue
        with open(flist_path, 'rb') as f:
            flist = pickle.load(f)
        for ocr_name in flist:
            img_name = ocr_name[:-len('.ocr.json')]
            examples.append({
                'ocr_path': str(img_dir / ocr_name),
                'img_path': str(img_dir / img_name),
            })
    return examples


def merge_bbox(bbox_l: List[float], bbox_r: List[float]):
    xrelax, yrelax = 2, 2
    xmin = min(bbox_l[0::2] + bbox_r[0::2])
    xmax = max(bbox_r[0::2] + bbox_r[0::2])
    ymin = min(bbox_l[1::2] + bbox_r[1::2])
    ymax = max(bbox_r[1::2] + bbox_r[1::2])
    return [
        xmin - xrelax, ymin - yrelax,
        xmax + xrelax, ymin - yrelax,
        xmax + xrelax, ymax + yrelax,
        xmin - xrelax, ymax + yrelax,
    ]


def merge_polygon(poly_l: List[float], poly_r: List[float]):
    xtl, ytl, xbl, ybl = poly_l[0], poly_l[1], poly_l[-2], poly_l[-1]
    xtr, ytr, xbr, ybr = poly_r[2], poly_r[3], poly_r[4], poly_r[5]
    return [xtl, ytl, xtr, ytr, xbr, ybr, xbl, ybl]


def get_bbox_info(bbox: List[float]):
    xtl, ytl, xtr, ytr, xbl, ybl = bbox[0], bbox[1], bbox[2], bbox[3], bbox[-2], bbox[-1]
    h = np.sqrt((xtl-xbl)**2+(ytl-ybl)**2)
    w = np.sqrt((xtl-xtr)**2+(ytl-ytr)**2)
    return h, w


def render_text_in_rect(
    rect_size:int,
    pad:int,
    text:str,
    font:ImageFont.FreeTypeFont,
):
    if not text:
        return np.ones((rect_size, rect_size, 1), dtype=np.float32)

    image = Image.new('L', (rect_size, rect_size), 'white')
    draw = ImageDraw.Draw(image)
    merge_lines = lambda lines: '\n'.join([' '.join(line) for line in lines])
    lines = [[]]
    words = text.split()
    longest_word = max(words, key=len)
    xmin, ymin, xmax, ymax = draw.multiline_textbbox((pad, 0), longest_word, font=font)
    if xmax > rect_size:
        while xmax > rect_size:
            font = ImageFont.truetype(font.path, font.size - 1)
            xmin, ymin, xmax, ymax = draw.multiline_textbbox((pad, 0), longest_word, font=font)
    for word in words:
        lines[-1].append(word)
        xmin, ymin, xmax, ymax = draw.multiline_textbbox((pad, 0), merge_lines(lines), font=font)
        if xmax > rect_size:
            lines.append([lines[-1].pop()])
            xmin, ymin, xmax, ymax = draw.multiline_textbbox((pad, 0), merge_lines(lines), font=font)
            if ymax > rect_size:
                lines.pop()
                lines[-1][-1] += '...'
                while draw.multiline_textbbox((pad, 0), merge_lines(lines), font=font)[0] > rect_size:
                    lines[-1].pop()
                    if lines[-1]:
                        lines[-1][-1] += '...'
                    else:
                        lines[-1].append('...')
                break
    text_multi_line = merge_lines(lines)
    draw.multiline_text((pad, 0), text_multi_line, spacing=0, font=font)
    img_tsr = np.array(image).astype(np.uint8)
    img_tsr = (img_tsr/127.5-1.0).astype(np.float32)
    return img_tsr[..., np.newaxis]


class AvidInpaintSizeAware(Dataset):

    def __init__(
        self,
        size:int,
        names:List[str],
        min_font_size:int,
        max_font_size:int,
        cond_size:int,
        cond_font_size:int,
        pad:float=4,
        dropout:float=0,
        num_samples:int=0,
    ):
        '''
        Avid Inpaint Size Aware Dataset

        Performs following ops:
        1. open oct.json, randomly pick one word as anchor
        2. decide size randomly based on anchor constraint
        3. expand words as much as possible
        4. crop the surrounding region & gen masked crop
        '''
        super().__init__()
        self.base = self.get_base(names)
        if num_samples > 0:
            self.base = self.base[:num_samples]
        self.size = size
        self.cond_size = cond_size
        self.pad = pad
        self.min_font_size = min_font_size
        self.max_font_size = max_font_size
        self.dropout = dropout
        self.img_rescler = al.SmallestMaxSize(max_size=size, interpolation=cv2.INTER_AREA)
        font_path = os.environ['AVID_FONT_PATH']
        self.font = ImageFont.truetype(font_path, cond_font_size)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i:int):
        example = self.base[i]
        try:
            with open(example['ocr_path'], 'r', encoding='utf-8') as f:
                ocr_data = json.load(f)
            words = ocr_data['analyzeResult']['pages'][0]['words']
        except Exception as e:
            print(e)
            words = []

        # if empty page or failed read, return
        if len(words) == 0:
            return {
                'image': np.ones((self.size, self.size, 3), dtype=np.float32),
                'mask': np.zeros((self.size, self.size, 1), dtype=np.float32),
                'txt_image': np.ones((self.cond_size, self.cond_size, 1), dtype=np.float32),
                'text': '',
                'font_size': 0,
            }

        # pick anchor word
        offset2line = self.offset_to_line(ocr_data['analyzeResult']['pages'][0]['lines'])
        j = np.random.randint(0, len(words))
        anchor_word = words[j]
        h, w = get_bbox_info(anchor_word['polygon'])
        max_size = self.size - self.pad * 2
        scale = self.get_scale(h, w, max_size)
        line = self.merge_words(words, j, max_size / scale, offset2line)
        text, poly = line['content'], line['polygon']
        if np.random.rand() < self.dropout:
            text = ''
        bbox_w, bbox_h = max(poly[::2]) - min(poly[::2]), max(poly[1::2]) - min(poly[1::2])
        max_size_scl, pad_scl = max_size / scale, self.pad / scale
        offx = int(np.random.rand() * max(0, max_size_scl - bbox_w) + pad_scl)
        offy = int(np.random.rand() * max(0, max_size_scl - bbox_h) + pad_scl)
        size_scl = int(self.size / scale)
        try:
            image = Image.open(example['img_path']).convert('RGB')
            img = np.array(image).astype(np.uint8)
            mask = np.zeros_like(img)[...,:1]
            xmin, xmax, ymin, ymax = min(poly[::2]), max(poly[::2]), min(poly[1::2]), max(poly[1::2])
            mask[ymin:ymax+1, xmin:xmax+1] = 255
            x_start, y_start = max(0, xmin - offx), max(0, ymin - offy)
            crop = img[y_start:y_start+size_scl, x_start:x_start+size_scl, :]
            mask = mask[y_start:y_start+size_scl, x_start:x_start+size_scl, :]
            hc, wc = crop.shape[:2]
            if hc != wc:
                n_pad = max(hc, wc)
                pad_with = ((0, n_pad-hc), (0, n_pad-wc), (0, 0))
                crop = np.pad(crop, pad_with, 'constant', constant_values=255)
                mask = np.pad(mask, pad_with, 'constant', constant_values=0)
            out = self.img_rescler(image=crop, mask=mask)
            crop, mask = out['image'], (out['mask'] != 0).astype(np.float32)
            txt_image = render_text_in_rect(self.cond_size, self.pad, text, self.font)
        except Exception as e:
            print(e)
            crop = np.ones((self.size, self.size, 3), dtype=np.uint8)*255
            mask = np.zeros((self.size, self.size, 1), dtype=np.float32)
            text = ''
            txt_image = np.ones((self.cond_size, self.cond_size, 1), dtype=np.float32),
        return {
            'image': (crop/127.5-1.0).astype(np.float32),
            'mask': mask,
            'txt_image': txt_image,
            'text': text,
            'font_size': h * scale
        }

    def offset_to_line(self, lines:List):
        span = lines[-1]['spans'][-1]
        total_len = span['offset'] + span['length']
        offset2line = [-1 for _ in range(total_len)]
        for j, line in enumerate(lines):
            for span in line['spans']:
                for k in range(span['length']):
                    offset2line[span['offset'] + k] = j
        return offset2line

    def is_poly_oversize(self, poly:List, max_size:float):
        xtl, ytl, xtr, ytr = poly[:4]
        w = np.sqrt((xtl-xtr)**2+(ytl-ytr)**2)
        return w > max_size

    def merge_words(self, words:List, j:int, max_size:float, offset2line:List):
        line, poly = words[j], words[j]['polygon']
        k = j + 1
        num_words = len(words)
        while True:
            if k >= num_words:
                break
            j_ofst, k_ofst = words[j]['span']['offset'], words[k]['span']['offset']
            if offset2line[j_ofst] != offset2line[k_ofst]:
                break
            poly = merge_polygon(poly, words[k]['polygon'])
            if self.is_poly_oversize(poly, max_size):
                break
            line['content'] += f' {words[k]["content"]}'
            line['polygon'] = poly
            k += 1
        return line


    def get_scale(self, h:float, w:float, max_size:int):
        min_font_size, max_font_size = self.min_font_size, self.max_font_size
        max_font_size = min(max_font_size, max_size / w * h)
        min_font_size = min(max_font_size, min_font_size)
        if min_font_size <= h <= max_font_size:
            return 1
        font_size = min_font_size + np.random.rand() * (max_font_size - min_font_size)
        scale = font_size / h
        return scale


    def get_base(self, names: List[str]):
        '''
        Get examples
        '''
        return get_flist_items(names)


class AvidInpaint(Dataset):

    def __init__(
        self,
        size:int,
        min_pad_f:float=20,
        max_pad_f:float=60,
        limit:int=40,
        ratio:int=10,
        names:List[str]=None,
    ):
        '''
        Avid Inpaint Dataset

        Performs following ops in order:
        1. open ocr.json, randomly pick one line to be inpainted
        2. crop the surrounding region of the line
        3. generate masked crop
        '''
        super().__init__()
        self.base = self.get_base(names)
        self.size = size
        self.limit = limit
        self.ratio = ratio
        assert 0 < min_pad_f <= max_pad_f
        self.min_pad_f = min_pad_f
        self.max_pad_f = max_pad_f
        self.img_rescler = al.SmallestMaxSize(max_size=size, interpolation=cv2.INTER_AREA)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i:int):
        example = self.base[i]
        try:
            with open(example['ocr_path'], 'r', encoding='utf-8') as f:
                ocr_data = json.load(f)
            words = ocr_data['analyzeResult']['pages'][0]['words']
        except Exception as e:
            print(e)
            words = []
        if len(words) == 0:
            ret = {
                'image': np.ones((self.size, self.size, 3), dtype=np.uint8)*255,
                'mask': np.zeros((self.size, self.size, 1), dtype=np.float32),
                'text': '',
            }
        else:
            j = np.random.randint(0, len(words))
            line = self.merge_words(words, j)
            text, bbox = line['content'], line['polygon']
            xmin, xmax = min(bbox[0::2]), max(bbox[0::2])
            ymin, ymax = min(bbox[1::2]), max(bbox[1::2])
            x_span, y_span = xmax - xmin, ymax - ymin
            pad_span = int((self.min_pad_f + np.random.rand() * (self.max_pad_f - self.min_pad_f)))
            crop_size = x_span + pad_span
            l = min(int(pad_span * np.random.rand()), xmin)
            t = min(int(max(crop_size - y_span, 0) * np.random.rand()), ymin)
            x_start, y_start = xmin - l, ymin - t
            try:
                with Image.open(example['img_path']) as image:
                    if not image.mode == 'RGB':
                        image = image.convert('RGB')
                    img = np.array(image).astype(np.uint8)
                crop = img[y_start:y_start+crop_size, x_start:x_start+crop_size, :]
                h, w = crop.shape[:2]
                crop_size = max(h, w)
                crop = np.pad(
                    crop,
                    [(0,crop_size-h),(0,crop_size-w),(0,0)],
                    mode='constant',
                    constant_values=255)
                mask = np.zeros((crop_size, crop_size, 1), dtype=np.uint8)
                mask[t:t+ymax-ymin, l:l+xmax-xmin] = 255
                out = self.img_rescler(image=crop, mask=mask)
                crop, mask = out['image'], (out['mask'] != 0).astype(np.float32)
            except Exception as e:
                print(e)
                crop = np.ones((self.size, self.size, 3), dtype=np.uint8)*255
                mask = np.zeros((self.size, self.size, 1), dtype=np.float32)
                mask[:8, :8] = 1
                text = ''
            ret = {
                'image': crop,
                'mask': mask,
                'text': text,
            }
        ret['image'] = (ret['image']/127.5-1.0).astype(np.float32)
        return ret

    def merge_bbox(self, bbox_l: List, bbox_r: List):
        return merge_bbox(bbox_l, bbox_r)

    def merge_words(self, words: List, j: int):
        line = words[j]
        bbox = line['polygon']
        ymin, ymax = min(bbox[1::2]), max(bbox[1::2])
        while len(line['content']) < self.limit:
            if j + 1 < len(words):
                bbox_next = words[j + 1]['polygon']
                bbox_merge = self.merge_bbox(bbox, bbox_next)
                xspan_merge, yspan_merge = bbox_merge[2] - bbox_merge[0], bbox_merge[-1] - bbox_merge[1]
                y_avg = 0.5 * (min(bbox_next[1::2])+max(bbox_next[1::2]))
                if ymin <= y_avg <= ymax and xspan_merge <= self.ratio * yspan_merge:
                    line['content'] += f' {words[j + 1]["content"]}'
                    line['polygon'] = bbox_merge
                else:
                    break
                j += 1
            else:
                break
        return line

    def get_base(self, names: List[str]):
        '''
        Get examples
        '''
        return get_flist_items(names)


class AvidInpaintTrain(AvidInpaint):

    def __init__(self, **kwargs):
        if 'names' not in kwargs:
            kwargs.update({ 'names': ['Limit1'] })
        super().__init__(**kwargs)


class AvidInpaintValidation(AvidInpaint):

    def __init__(self, num_items:int=1024, **kwargs):
        if 'names' not in kwargs:
            kwargs.update({ 'names': ['Random'] })
        super().__init__(**kwargs)
        self.base = self.base[:num_items]


class AvidInpaintTxtImg(AvidInpaint):

    def __init__(
        self,
        font_size:int=24,
        img_height:int=32,
        dropout:float=0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        font_path = os.environ['AVID_FONT_PATH']
        self.font = ImageFont.truetype(font_path, font_size)
        self.img_height = img_height
        self.dropout = dropout

    def __getitem__(self, i:int):
        item = super().__getitem__(i)

        text = item['text']
        if np.random.random() < self.dropout:
            text = ''
        font_size = self.font.size
        height = self.img_height
        assert height >= font_size
        pad = (height - font_size) // 2
        width = self.font.getlength(text)+height-font_size
        width = int((width+height-1)//height*height)
        canvas = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(canvas)
        draw.text((pad, -pad), text, font=self.font, fill='black')
        item['txt_image'] = canvas
        return item


class AvidInpaintTxtImgTrain(AvidInpaintTxtImg):

    def __init__(self, **kwargs):
        if 'names' not in kwargs:
            kwargs.update({ 'names': ['Limit1'] })
        super().__init__(**kwargs)


class AvidInpaintTxtImgValidation(AvidInpaintTxtImg):

    def __init__(self, num_items:int=1024, **kwargs):
        if 'names' not in kwargs:
            kwargs.update({ 'names': ['Random'] })
        super().__init__(**kwargs)
        self.base = self.base[:num_items]


class AvidSuperRes(Dataset):

    def __init__(
        self,
        size:int=None,
        degradation:str=None,
        downscale_f:int=4,
        min_crop_f:float=0.5,
        max_crop_f:float=1.,
        names:List[str]=None,
    ):
        '''
        Avid Superresolution Dataset

        Performs following ops in order:
        1. crops image patch with size s as a random crop
        2. resizes crop to size with cv2.area_interpolation
        3. degrades resized crop with degradation_fn

        :param size: resizing to size after cropping
        :param degradation: degradation_fn, e.g. cv_bicubic or bsrgan_light
        :param downscale_f: downsample factor
        :param min_crop_f: determins crop size s, s = c * min_img_side_len
            where c is sampled from interval (min_crop_f, max_crop_f)
        :param max_crop_f: explained above
        '''
        super().__init__()
        self.base = self.get_base(names)
        assert size and (size / downscale_f).is_integer()
        self.size = size
        self.lr_size = int(size / downscale_f)
        assert 0 < min_crop_f <= max_crop_f <= 1
        self.min_crop_f = min_crop_f
        self.max_crop_f = max_crop_f
        self.img_rescler = al.SmallestMaxSize(max_size=size, interpolation=cv2.INTER_AREA)
        self.pil_interpolation = False
        if degradation == 'bsrgan':
            self.degradation_process = partial(degradation_fn_bsr, sf=downscale_f)
        elif degradation == 'bsrgan_light':
            self.degradation_process = partial(degradation_fn_bsr_light)
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
                self.degradation_process = partial(
                    ttf.resize, size=self.LR_size, interpolation=interpolation_fn)
            else:
                self.degradation_process = al.SmallestMaxSize(
                    max_size=self.lr_size, interpolation=interpolation_fn)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i:int):
        example = self.base[i]
        with Image.open(example['path']) as image:
            if not image.mode == 'RGB':
                image = image.convert('RGB')
            img = np.array(image).astype(np.uint8)
        min_side_len = min(img.shape[:2])
        crop_side_len = min_side_len * np.random.uniform(self.min_crop_f, self.max_crop_f, size=None)
        crop_side_len = int(crop_side_len)
        self.cropper = al.RandomCrop(height=crop_side_len, width=crop_side_len)
        img = self.cropper(image=img)['image']
        try:
            img = self.img_rescler(image=img)['image']
        except Exception as e:
            print(e)
            img = np.ones((self.size, self.size, 3), dtype=np.uint8)*255
        if self.pil_interpolation:
            image_pil = Image.fromarray(img)
            lr_image = self.degradation_process(image_pil)
            lr_image = np.array(lr_image).astype(np.uint8)
        else:
            lr_image = self.degradation_process(image=img)['image']
        return {
            'image': (img/127.5-1.0).astype(np.float32),
            'lr_image': (lr_image/127.5-1.0).astype(np.float32)
        }

    def get_base(self, names: List[str]):
        '''
        Get examples
        '''
        root_dir = Path(os.environ['AVID_ROOT_DIR'])
        examples = []
        for folder_name in names:
            img_dir = root_dir / folder_name
            flist_path = img_dir / 'flist.pkl'
            if not flist_path.exists():
                continue
            with open(flist_path, 'rb') as f:
                flist = pickle.load(f)
            for fname in flist:
                examples.append({ 'path': str(img_dir / fname) })
        return examples


class AvidSuperResTrain(AvidSuperRes):

    def __init__(self, **kwargs):
        if 'names' not in kwargs:
            kwargs.update({ 'names': ['Limit1'] })
        super().__init__(**kwargs)


class AvidSuperResValidation(AvidSuperRes):

    def __init__(self, num_items:int=1024, **kwargs):
        if 'names' not in kwargs:
            kwargs.update({ 'names': ['Random'] })
        super().__init__(**kwargs)
        self.base = self.base[:num_items]


def gen_file_list():
    avid_root_dir = Path(os.environ['AVID_ROOT_DIR'])
    for sub_dir in avid_root_dir.glob('*'):
        if not sub_dir.is_dir():
            continue
        print(f'process {sub_dir.name}')
        flist = []
        for path in tqdm(list(sub_dir.glob('*.jpg'))):
            flist.append(path.name)
        with open(sub_dir / 'flist.pkl', 'wb') as f:
            pickle.dump(flist, f)


def gen_ocr_list():
    avid_root_dir = Path(os.environ['AVID_ROOT_DIR'])
    for sub_dir in avid_root_dir.glob('*'):
        if not sub_dir.is_dir():
            continue
        print(f'process {sub_dir.name}')
        flist = []
        num_miss = 0
        for ocr_path in tqdm(list(sub_dir.glob('*.ocr.json'))):
            img_path = ocr_path.parent / ocr_path.name[:-len('.ocr.json')]
            if img_path.exists():
                flist.append(ocr_path.name)
            else:
                num_miss += 1
        with open(sub_dir / 'flist_ocr.pkl', 'wb') as f:
            pickle.dump(flist, f)
        print(f'total: {len(flist)} ocr files; miss: {num_miss}')


def filter_file_list():
    avid_root_dir = Path(os.environ['AVID_ROOT_DIR'])
    for sub_dir in avid_root_dir.glob('*'):
        if not sub_dir.is_dir():
            continue
        print(f'process {sub_dir.name}')
        for path in tqdm(list(sub_dir.glob('*.jpg'))):
            with Image.open(path) as image:
                h, w = image.size
                if h * 4 < w or w * 4 < h:
                    print(f'Delete {path.name} with shape {h} x {w}')
                    os.remove(str(path))


def test_inpaint_dataset():
    ds = AvidInpaint(128, names=['Limit100'])
    for i in range(len(ds)):
        example = ds[i]
        print(example['text'])
        image = ((example['image']+1)*127.5).astype(np.uint8)
        Image.fromarray(image).save('a.jpg')
        mask = ((example['mask'])*255).astype(np.uint8)
        mask = np.repeat(mask, 3, -1)
        Image.fromarray(mask).save('b.jpg')


def test_inpaint_img_dataset():
    ds = AvidInpaintTxtImg(size=128, names=['Random'])
    for i in range(len(ds)):
        example = ds[i]
        print(example['text'])
        example['txt_image'].save('a.jpg')


def test_inpaint_sw_dataset():
    ds = AvidInpaintSizeAware(
        size=128,
        names=['Random'],
        min_font_size=10,
        max_font_size=16,
        cond_size=224,
        cond_font_size=24,
    )
    for i in tqdm(range(len(ds))):
        example = ds[i]
        image = ((example['image']+1)*127.5).astype(np.uint8)
        Image.fromarray(image).save('a.jpg')
        mask = ((example['mask'])*255).astype(np.uint8)
        mask = np.repeat(mask, 3, -1)
        Image.fromarray(mask).save('b.jpg')
        txt_image = ((example['txt_image']+1)*127.5).astype(np.uint8)
        Image.fromarray(txt_image).save('c.jpg')


def clean_empty():
    root_dir = Path(os.environ['AVID_ROOT_DIR'])
    names = [path for path in root_dir.glob('*') if path.is_dir()]
    for folder_name in names:
        img_dir = root_dir / folder_name
        flist_path = img_dir / 'flist_ocr.pkl'
        if not flist_path.exists():
            continue
        with open(flist_path, 'rb') as f:
            flist = pickle.load(f)
        flist_new = []
        print(f'{folder_name} - old len: {len(flist)}')
        for ocr_name in tqdm(flist):
            ocr_path = img_dir / ocr_name
            with open(ocr_path, 'r', encoding='utf-8') as f:
                ocr_data = json.load(f)
            words = ocr_data['analyzeResult']['pages'][0]['words']
            if len(words) == 0:
                continue
            flist_new.append(ocr_name)
        print(f'{folder_name} - new len: {len(flist_new)}')
        with open(flist_path, 'wb') as f:
            pickle.dump(flist_new, f)


def main(mode:str):
    if mode == 'gen':
        gen_file_list()
    elif mode == 'filter':
        filter_file_list()
    elif mode == 'gen_ocr':
        gen_ocr_list()
    elif mode == 'test_inpaint':
        test_inpaint_dataset()
    elif mode == 'test_inpaint_img':
        test_inpaint_img_dataset()
    elif mode == 'test_inpaint_sw':
        test_inpaint_sw_dataset()
    elif mode == 'clean_empty':
        clean_empty()


if __name__ == '__main__':
    fire.Fire(main)