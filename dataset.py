import torch
import torch.nn as nn
import dicomsdl
import numpy as np
from glob import glob
import torch.nn.functional as F


def do_pad_to_square(image):
    l, h, w = image.shape
    if w > h:
        pad = w - h
        pad0 = pad // 2
        pad1 = pad - pad0
        image = F.pad(image, [0, 0, pad0, pad1], mode='constant', value=0)
    if w < h:
        pad = h - w
        pad0 = pad // 2
        pad1 = pad - pad0
        image = F.pad(image, [pad0, pad1, 0, 0], mode='constant', value=0)
    return image


def do_scale_to_size(image, spacing, max_size):
    dz, dy, dx = spacing
    l, s, s = image.shape # scale to max size
    if max_size != s:
        scale = max_size / s

        l = int(dz / dy * l * 0.5)  # we use sapcing dz,dy,dx = 2,1,1
        l = int(scale * l)
        h = int(scale * s)
        w = int(scale * s)

        image = F.interpolate(
            image.unsqueeze(0).unsqueeze(0),
            size=(l, h, w),
            mode='trilinear',
            align_corners=False,
        ).squeeze(0).squeeze(0)

    return image
         

def dicomsdl_to_numpy_image(ds, index=0):
    info = ds.getPixelDataInfo()
    if info['SamplesPerPixel'] != 1:
        raise RuntimeError('SamplesPerPixel != 1')  # number of separate planes in this image
    shape = [info['Rows'], info['Cols']]
    dtype = info['dtype']
    outarr = np.empty(shape, dtype=dtype)
    ds.copyFrameData(index, outarr)
    return outarr


def load_dicomsdl_dir(dcm_dir, slice_range=None):
    dcm_file = sorted(glob(f'{dcm_dir}/*.dcm'), key=lambda x: int(x.split('/')[-1].split('.')[0]))
     
    #fake some slice so that it won't cause error ....
    if len(dcm_file)==1:
        dcm = dicomsdl.open(dcm_file[0])
        pixel_array = dicomsdl_to_numpy_image(dcm) 
        pixel_array = pixel_array.astype(np.float32)
        image = np.stack([pixel_array]*16)
        dz,dy,dx = 1,1,1
        return image, (dz,dy,dx)
    
    
    #------------------------------------
    if slice_range is None: 
        slice_min = int(dcm_file[0].split('/')[-1].split('.')[0])
        slice_max = int(dcm_file[-1].split('/')[-1].split('.')[0])+1
        slice_range=(slice_min, slice_max)

    slice_min, slice_max = slice_range
    sz0, szN = None, None

    image = []
    for s in range(slice_min, slice_max):
        f = f'{dcm_dir}/{s}.dcm'

        #dcm = pydicom.read_file(f)
        #m = dcm.pixel_array
        #m = standardize_pixel_array(dcm)

        dcm = dicomsdl.open(f)
        pixel_array = dicomsdl_to_numpy_image(dcm)
        if dcm.PixelRepresentation == 1:
            bit_shift = dcm.BitsAllocated - dcm.BitsStored
            dtype = pixel_array.dtype
            pixel_array = (pixel_array << bit_shift).astype(dtype) >> bit_shift

        #processing
        pixel_array = pixel_array.astype(np.float32)
        pixel_array = dcm.RescaleSlope * pixel_array + dcm.RescaleIntercept
        xmin = dcm.WindowCenter-0.5-(dcm.WindowWidth-1)* 0.5
        xmax = dcm.WindowCenter-0.5+(dcm.WindowWidth-1)* 0.5
        norm = np.empty_like(pixel_array, dtype=np.uint8)
        dicomsdl.util.convert_to_uint8(pixel_array, norm, xmin, xmax)

        if dcm.PhotometricInterpretation == 'MONOCHROME1':
            norm = 255 - norm
        image.append(norm)

    if 1: #check inversion
        dcm0 = dicomsdl.open(f'{dcm_dir}/{slice_min}.dcm')
        dcmN = dicomsdl.open(f'{dcm_dir}/{slice_max-1}.dcm')
        sx0, sy0, sz0 = dcm0.ImagePositionPatient
        sxN, syN, szN = dcmN.ImagePositionPatient
        if szN > sz0:
            image=image[::-1]

        dx, dy = dcm0.PixelSpacing
        dz = np.abs((szN - sz0) / (slice_max - slice_min-1))

    image = np.stack(image)
    return image, (dz,dy,dx)


class RsnaDataset():
    def __init__(self, cfg, fold=0, mode='train'):
        super().__init__()
        self.df = pd.read_csv('data/fold/{}_fold_{}.csv'.format(mode, fold))
        self.mode = model
        self.batch_size = cfg.batch_size

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        image_info = self.df.iloc[idx]
        image_path = image_info['path']
        target = 
        
# image --> load_dicomsdl_dir: 839, 512, 512