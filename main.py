import os
import json
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from utils import TrainingParameters, save_checkpoint, adjust_learning_rate
from trainer import train, validate
from models import Encoder, Decoder
from dataset import CaptionDataset

# Data parameters
data_folder = '/netscratch/deshmukh/Datasets/data_show-attend-tell'  # output folder used in generate_input_files
data_name = 'flickr8k_5_captions_per_image_5_minimum_word_frequency'  # base name shared by data files

# Model parameters
embedding_dimension = 512  # dimension of word embeddings
hidden_dimension = 512  # dimension of decoder RNN
attention_dimension = 512
context_vector_dimension = 512
dropout = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

print('Device : ', device)

def main():

    print('Training parameters Initialized')
    training_parameters = TrainingParameters( start_epoch = 0,
                                            epochs = 120,  # number of epochs to train for
                                            epochs_since_improvement = 0,  # Epochs since improvement in BLEU score
                                            batch_size = 32,
                                            workers = 1,  # for data-loading; right now, only 1 works with h5py
                                            fine_tune_encoder = True,  # fine-tune encoder
                                            encoder_lr = 1e-4,  # learning rate for encoder, if fine-tuning is used
                                            decoder_lr = 4e-4,  # learning rate for decoder
                                            grad_clip = 5.0,  # clip gradients at an absolute value of
                                            alpha_c = 1.0,  # regularization parameter for 'doubly stochastic attention'
                                            best_bleu4 = 0.0,  # BLEU-4 score right now
                                            print_freq = 100,  # print training/validation stats every __ batches
                                            checkpoint =  './Result/BEST_checkpoint_flickr8k_5_captions_per_image_5_minimum_word_frequency.pth.tar' # path to checkpoint, None if none
                                            # checkpoint = None
                                          )

    print('Loading Word-Map')
    word_map_file = os.path.join(data_folder,'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    print('Creating Model')

    if training_parameters.checkpoint is None:
        encoder = Encoder()
        encoder.fine_tune(training_parameters.fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p : p.requires_grad, encoder.parameters()),
                                                lr=training_parameters.encoder_lr) if training_parameters.fine_tune_encoder else None
        
        decoder = Decoder(attention_dimension = attention_dimension,
                            embedding_dimension = embedding_dimension,
                            hidden_dimension = hidden_dimension,
                            vocab_size = len(word_map),
                            device = device,
                            context_vector_dimension = context_vector_dimension,               
                            dropout = dropout)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p : p.requires_grad, decoder.parameters()),
                                                lr=training_parameters.decoder_lr)

    else:
        checkpoint = torch.load(training_parameters.checkpoint)
        training_parameters.start_epoch = checkpoint['epoch'] + 1
        training_parameters.epochs_since_improvement = checkpoint['epochs_since_improvement']
        training_parameters.best_bleu4 = checkpoint['bleu4']

        encoder = Encoder()
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        encoder_optimizer = checkpoint['encoder_optimizer']

        decoder = Decoder(attention_dimension = attention_dimension,
                            embedding_dimension = embedding_dimension,
                            hidden_dimension = hidden_dimension,
                            vocab_size = len(word_map),
                            device = device,
                            context_vector_dimension = context_vector_dimension,
                            dropout = dropout)
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        decoder_optimizer = checkpoint['decoder_optimizer']

        if training_parameters.fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(training_parameters.fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p : p.requires_grad, encoder.parameters()),
                                                lr=training_parameters.encoder_lr)

    encoder.to(device)
    decoder.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
        
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    print('Creating Data Loaders')
    train_dataloader = torch.utils.data.DataLoader(
                                    CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize])),
                                    batch_size=training_parameters.batch_size, shuffle=True)
    
    validation_dataloader = torch.utils.data.DataLoader(
                                    CaptionDataset(data_folder, data_name, 'VALID', transform=transforms.Compose([normalize])),
                                    batch_size=training_parameters.batch_size, shuffle=True, pin_memory=True)

    for epoch in range(training_parameters.start_epoch, training_parameters.epochs):

        if training_parameters.epochs_since_improvement == 20:
            break
            
        if training_parameters.epochs_since_improvement > 0  and training_parameters.epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if training_parameters.fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        train(train_loader = train_dataloader,
              encoder = encoder,
              decoder = decoder,
              criterion = criterion,
              encoder_optimizer = encoder_optimizer,
              decoder_optimizer = decoder_optimizer,
              epoch = epoch,
              device = device,
              training_parameters = training_parameters)

        break

        recent_bleu4_score = validate(validation_loader = validation_dataloader,
                                    encoder = encoder,
                                    decoder = decoder,
                                    criterion = criterion,
                                    word_map = word_map,
                                    device = device,
                                    training_parameters = training_parameters)

        is_best_score = recent_bleu4_score > training_parameters.best_bleu4
        training_parameters.best_bleu4 = max(recent_bleu4_score, training_parameters.best_bleu4)
        if not is_best_score:
            training_parameters.epochs_since_improvement += 1
            print('\nEpochs since last improvement : %d\n' % (training_parameters.epochs_since_improvement))
        else:
            training_parameters.epochs_since_improvement = 0
        
        save_checkpoint(data_name, epoch, training_parameters.epochs_since_improvement, encoder, decoder,
                        encoder_optimizer, decoder_optimizer, recent_bleu4_score, is_best_score)       

if __name__ =='__main__':
    main()