import os
import h5py
import json
import torch
from torch.utils.data import Dataset

class CaptionDataset(Dataset):

    def __init__(self, data_folder, data_name, split, transform=None):

        assert split in {'TRAIN', 'VALID', 'TEST'}

        self.split = split

        # Images
        self.image_file = h5py.File(os.path.join(data_folder, split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.images = self.image_file['images']

        # Number of captions per image
        self.captions_per_image = self.image_file.attrs['captions_per_image']

        # All the Captions
        with open(os.path.join(data_folder, split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)
        
        # Lengths of all the captions
        with open(os.path.join(data_folder, split + '_CAPTION_LENGTHS_' + data_name + '.json'), 'r') as j:
            self.caption_lengths = json.load(j)

        self.transform = transform

        self.dataset_size = len(self.captions)

    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx):
        
        image = torch.FloatTensor(self.images[idx//self.captions_per_image] / 255.0)
        if self.transform is not None:
            image = self.transform(image)

        caption = torch.LongTensor(self.captions[idx])
        caption_length = torch.LongTensor([self.caption_lengths[idx]])
        
        if self.split is 'TRAIN':
            return image, caption, caption_length
        else:
            all_captions = torch.LongTensor(
                self.captions[((idx//self.captions_per_image)*self.captions_per_image): \
                                ((idx//self.captions_per_image)*self.captions_per_image) + self.captions_per_image])
            
            return image, caption, caption_length, all_captions





