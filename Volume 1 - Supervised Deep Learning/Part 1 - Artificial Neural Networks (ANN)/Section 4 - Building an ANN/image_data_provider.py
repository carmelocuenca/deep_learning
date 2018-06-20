import numpy as np
import imageio
import glob

class ImageDataProvider:
    """
    Generic data provider for images, supports gray scale and colored images.
    Assumes that the data images and label images are stored in the same folder
    and that the labels have a different file suffix 
    e.g. 'train/fish_1.tif' and 'train/fish_1_mask.tif'

    Usage:
    data_provider = ImageDataProvider("../fishes/train/*.tif")
        
    :param search_path: a glob search pattern to find all data and label images
    :param data_suffix: suffix pattern for the data images. Default '.tif'
    :param mask_suffix: suffix pattern for the label images. Default '_mask.tif'
    """
    
    def __init__(self, search_path, data_suffix=".tif",
                 mask_suffix='_mask.tif', shuffle_data=True):
        
        self.data_suffix = data_suffix
        self.mask_suffix = mask_suffix
        self.shuffle_data = shuffle_data
        
        self.data_files = self._find_data_files(search_path)
        
        if self.shuffle_data:
            np.random.shuffle(self.data_files)
        
        assert len(self.data_files) > 0, "No training files"
        print("Number of files used: %s" % len(self.data_files))
        
        img = imageio.imread(self.data_files[0])
        self.out_rows = img.shape[0]
        self.out_cols =  img.shape[1]
        
    def _find_data_files(self, search_path):
        all_files = sorted(glob.glob(search_path))
        return [name for name in all_files if self.data_suffix in name and not self.mask_suffix in name]
    
        
    def load_data(self):
        i = 0
        imgdatas = np.ndarray((len(self.data_files),self.out_rows,self.out_cols,1), dtype=np.int16)
        if self.mask_suffix!= None:
            imglabels = np.ndarray((len(self.data_files),self.out_rows,self.out_cols,1), dtype=np.uint8)
        else:
            imglabels = None
        for data_file in self.data_files:
            image_name = data_file
#            img = imageio.imread(image_name)
#            imgdatas[i] =img.reshape(*img.shape, 1)
            
            from PIL import Image
            img2 = Image.open(image_name)
            pix = np.array(img2.getdata()).reshape(img2.size[1], img2.size[0]).astype(np.int16)
            imgdatas[i] = pix.reshape(*pix.shape, 1)
            
            if self.mask_suffix!= None:
                label_name = image_name.replace(self.data_suffix, self.mask_suffix)
                label  = imageio.imread(label_name)
                label[label>0] = 1
                imglabels[i]  = label.reshape(*label.shape, 1)
            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, len(self.data_files)))
            
            i += 1
        print('loading done')
        
        print(imgdatas.shape, np.min(imgdatas), np.max(imgdatas), np.mean(imgdatas), np.std(imgdatas))
        
        if self.mask_suffix!= None:
            return imgdatas.astype('float32'), imglabels.astype('float32')
        else:
            return imgdatas.astype('float32'), None
