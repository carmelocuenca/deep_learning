# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from scipy import misc

import nibabel as nib

XSIZE = 384
YSIZE = 384

from PIL import Image

#data = np.random.random((3, 2))
#img1 = Image.fromarray(data)
#img1.save('test.tiff')
#img2 = Image.open('test.tiff')
#
#pix = np.array(img2.getdata()).reshape(img2.size[1], img2.size[0])
#
#print(data -pix)
#print(data)
#print(pix)

bboxes = {}
def bbox(y):
    idx, idy, _ = np.where(y[:, :,:,0]!=0)
    
    xc, yc = 256, 256
    if idx.shape[0]!=0:
        print("limites", np.min(idx), np.max(idx), np.min(idy) , np.max(idy))
        xc = (np.min(idx) + np.max(idx))//2
        yc = (np.min(idy) + np.max(idy))//2
    else:
        print("soy cero")
    xmin = max(0, xc-XSIZE/2); xmax = min(512, xmin+XSIZE); xmin = xmax-XSIZE
    ymin = max(0, yc-YSIZE/2); ymax = min(512, ymin+YSIZE); ymin = ymax-YSIZE
    
    return int(xmin), int(xmax), int(ymin), int(ymax)

def normalize(X):
    mean = np.mean(X)
    std = np.std(X)
    return (X-mean)/std


def preprocessing(filename, X, y, limits=None):
    if limits==None:
        xmin, xmax, ymin, ymax = bbox(y)
        bboxes[filename ] = (xmin, xmax, ymin, ymax)
    else:
        print(limits)
        (xmin, xmax, ymin, ymax) = limits
    
    print(xmin, xmax, ymin, ymax, filename, X.shape, y.shape)
    assert(xmax-xmin==XSIZE and ymax-ymin==YSIZE )
    assert(X.shape[0]==512 and X.shape[1]==512 and X.shape[3]==1)
    assert(y.shape[0]==512 and y.shape[1]==512 and y.shape[3]==1)
    #return (normalize(X[xmin:xmax, ymin:ymax,...]), y[xmin:xmax, ymin:ymax,...])
    return (X[xmin:xmax, ymin:ymax,...], y[xmin:xmax, ymin:ymax,...])


import imageio
import glob
search_path = '/media/carmelocuenca/My Passport/Elio/automatic/'
output_path = '/media/carmelocuenca/My Passport/Elio/automatic/train/'
hdr_names = ['USC-0001.hdr', 'USC-0139.hdr', 'USC-0141.hdr',
             'USC-0149.hdr', 'USC-0164.hdr']
termination='_SNAKES3D.hdr'
#hdr_excludenames = ['USC-0001.hdr', 'USC-0139.hdr', 'USC-0141.hdr', 'USC-149.hdr', 'USC-0164.hdr',
#                    'USC-0164.hdr',  'USC-104.hdr']
#search_path = '/media/carmelocuenca/My Passport/Elio/manual/'
#output_path = '/media/carmelocuenca/My Passport/Elio/manual/train/'
#termination='_manual_segmentation_radiologist.hdr'
#hdr_names = ['USC-0001.hdr', 'USC-0139.hdr', 'USC-0141.hdr', 'USC-0149.hdr', 'USC-0164.hdr',
#'USC-0055.hdr',  'USC-0119.hdr',    'USC-0173.hdr',
#'USC-0053.hdr',  'USC-0100.hdr',    'USC-0164.hdr',  'USC-0104.hdr']

hdr_excludenames=''

idx = 0    
for filename in hdr_names:
  if filename not in hdr_excludenames:
      name = search_path + filename
      hdr = nib.load(name).get_data()
      mask = nib.load(name[0:-4] + termination).get_data()
      assert(hdr.shape == mask.shape) # Algo está mal y los ficheros no tienen la misma dimensión
      print("Serie %d Reading file [%s], total frames[%d]" % (idx, name, hdr.shape[2]) )
      #hdr, mask = preprocessing(filename, hdr.astype('float32'), mask, limits=bboxes[filename])
      #hdr, mask = preprocessing2(filename, hdr.astype('float32'), mask)
      hdr, mask = preprocessing(filename, hdr, mask, limits=bboxes[filename])
      #hdr, mask = preprocessing(filename, hdr, mask)
      print(filename, np.min(hdr), np.max(hdr), np.mean(hdr), np.std(hdr))
      
      assert(hdr.shape[0]==XSIZE and hdr.shape[1]==YSIZE and hdr.shape[3]==1)
      assert(hdr.shape ==  mask.shape)
      # print("shape: ", np.mean(data), np.std(data))
      for i in range(hdr.shape[2]):
        src = hdr[:, :, i, 0];
        #ssrc.dtype = np.uint16 # Something is wrong!!!
        seg = mask[:, :, i, 0]
        Image.fromarray(src).save(output_path + '%6.6d.tiff' % (idx + i))
        imageio.imwrite( output_path + '%6.6d_mask.tiff' % (idx + i), seg)
  idx += 10000

# Crop de las imágenes
#import os
#import glob
#from PIL import Image
#
#for file in set(glob.glob(output_path +'*.tiff')) - set(glob.glob(output_path + '*_mask.tiff')):
#  src = np.array(Image.open(file), np.int16)
#  seg = np.array(Image.open(os.path.splitext(file)[0]+'_mask.tiff'), np.uint8)
#  imageio.imwrite( file, src[xmin:xmax, ymin:ymax])
#  imageio.imwrite( os.path.splitext(file)[0]+'_mask.tiff', seg[xmin:xmax, ymin:ymax])
  
  
  
# Muestra las imágenes para comprobar que todo va bien

# Show de las images

i = 0

for file in set(glob.glob(output_path + '*.tiff')) - set(glob.glob(output_path + '/*_mask.tiff')):
    if i%100==0:
        img2 = np.array(Image.open(file))
        print(img2.shape, img2.dtype)
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(img2, cmap='gray')
        plt.show()
    i = i+1
  # reading image file

  # src = np.array(Image.open(file), np.float32)

  # seg = np.array(Image.open(os.path.splitext(file)[0]+'_mask.tiff'), bool)
  
  