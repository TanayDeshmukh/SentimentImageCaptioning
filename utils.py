import os
import json
import h5py
import torch
import numpy as np
from tqdm import tqdm
from random import seed
from scipy.misc import imread, imresize
from collections import Counter
from random import sample, choice

def create_input_files(dataset, karpathy_json_path, image_folder, 
                        captions_per_image, min_word_freq, ouput_folder, max_len=100):
    
    with open(karpathy_json_path, 'r') as f:
        data = json.load(f)
    
    train_image_paths = []
    train_captions = []
    valid_image_paths = []
    valid_captions = []
    test_image_paths = []
    test_captions = []
    word_frequency = Counter()

    images = data['images']

    for i in tqdm(range(len(images))):
        image = images[i]
        captions = []
        for sentences in image['sentences']:
            word_frequency.update(sentences['tokens'])
            if len(sentences['tokens']) <= max_len:
                captions.append(sentences['tokens'])
        
        if len(captions) == 0:
            continue

        path = os.path.join(image_folder, image['filename'])

        if image['split'] in {'train', 'restval'}:
            train_image_paths.append(path)
            train_captions.append(captions)
        
        elif image['split'] in {'val'}:
            valid_image_paths.append(path)
            valid_captions.append(captions)

        elif image['split'] in {'test'}:
            test_image_paths.append(path)
            test_captions.append(captions)

    assert len(train_captions) == len(train_image_paths)
    assert len(valid_captions) == len(valid_image_paths)
    assert len(test_captions) == len(test_image_paths)

    words = [w for w in word_frequency.keys() if word_frequency[w] >= min_word_freq]
    word_map = {k:v+1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map)+1
    word_map['<start>'] = len(word_map)+1
    word_map['<end>'] = len(word_map)+1
    word_map['<pad>'] = 0

    
    # Base name for all the optput files
    base_file_name = dataset + '_' + str(captions_per_image) + '_captions_per_image_' + str(min_word_freq) + '_minimum_word_frequency'

    #saving wordmap to a json file
    with open(os.path.join(ouput_folder, 'WORDMAP_'+base_file_name+'.json'), 'w') as f:
        json.dump(word_map, f)
    
    seed(123)
    for image_paths, image_captions, split in [(train_image_paths, train_captions, 'TRAIN'),
                                                (valid_image_paths, valid_captions, 'VALID'),
                                                (test_image_paths, test_captions, 'TEST')]:
        
        with h5py.File(os.path.join(ouput_folder, split +'_IMAGES_'+base_file_name+'.hdf5'), 'a') as h:

            h.attrs['captions_per_image'] = captions_per_image
            images = h.create_dataset('images', (len(image_paths), 3, 256, 256), dtype='uint8')
            
            print('Reading %s images and captions.'%split)

            encoded_captions = []
            caption_lengths = []

            for i, path in tqdm(enumerate(image_paths)):

                # Sample captions
                if len(image_captions[i]) < captions_per_image:
                    captions = image_captions[i] + [choice(image_captions[i]) for _ in range(captions_per_image - len(image_captions))]
                else:
                    captions = sample(image_captions[i], k=captions_per_image)

                # Sanity check
                assert len(captions) == captions_per_image

                # Read Images
                img = imread(image_paths[i])
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                # img = imresize(img, (256, 256))
                img = img.transpose(2, 0, 1)

                # Sanity check
                assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255

                # Save image to hdf5 file
                images[i] = img

                for j, c in enumerate(captions):
                    encoded_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + \
                                [word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))
                    c_len = len(c) + 2

                    encoded_captions.append(encoded_c)
                    caption_lengths.append(c_len)
                
            # Sanity check
            assert images.shape[0] * captions_per_image == len(encoded_captions) == len(caption_lengths)

            # Save Encoded Captions and Caption Lengths in json files
            with open(os.path.join(ouput_folder, split + '_CAPTIONS_' + base_file_name + '.json'), 'w') as j:
                json.dump(encoded_captions, j)
            
            with open(os.path.join(ouput_folder, split + '_CAPTION_LENGTHS_' + base_file_name + '.json'), 'w') as j:
                json.dump(caption_lengths, j)


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def accuracy(output, targets, k):
    '''
    Computes top k accuracies

    :param output: output from the model
    :param targets: true labels
    :return: top-k accuracies
    '''

    batch_size = targets.size(0)
    _, indices = output.topk(k, 1, True, True)
    correct = indices.eq(targets.view(-1, 1).expand_as(indices))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)

def adjust_learning_rate(optimizer, decay_factor):
    '''
    decay learning rate by a given factor
    '''

    print('\nDecaying Learning rate.')
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))

def save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder,
                        encoder_optimizer, decoder_optimizer, recent_bleu4_score, is_best_score):
    
    '''
    Save model checkpoint

    '''
    
    state = {
        'epoch' : epoch,
        'epochs_since_improvement' : epochs_since_improvement,
        'bleu4' : recent_bleu4_score,
        'encoder_state_dict' : encoder.state_dict(),
        'decoder_state_dict' : decoder.state_dict(),
        'encoder_optimizer' : encoder_optimizer,
        'decoder_optimizer' : decoder_optimizer,
    }
    file_name = './Result/checkpoint_' + data_name + '.pth.tar'
    torch.save(state, file_name)
    if is_best_score:
        file_name = './Result/BEST_checkpoint_' + data_name + '.pth.tar'
        torch.save(state, file_name)

class AverageMeter(object):
    '''
    To keep track of most recent average, sum and count of a metric
    '''

    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum/self.count

class TrainingParameters(object):

    def __init__(self, start_epoch, epochs, epochs_since_improvement, batch_size, workers, encoder_lr, decoder_lr,
                    grad_clip, alpha_c, best_bleu4, print_freq, fine_tune_encoder, checkpoint):
        self.start_epoch = start_epoch
        self.epochs = epochs
        self.epochs_since_improvement = epochs_since_improvement
        self.batch_size = batch_size
        self.workers = workers
        self.encoder_lr = encoder_lr
        self.decoder_lr = decoder_lr
        self.grad_clip = grad_clip
        self.alpha_c = alpha_c
        self.best_bleu4 = best_bleu4
        self.print_freq = print_freq
        self.fine_tune_encoder = fine_tune_encoder
        self.checkpoint = checkpoint
