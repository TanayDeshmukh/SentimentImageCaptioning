import time
import torch
from nltk.translate.bleu_score import corpus_bleu
from torch.nn.utils.rnn import pack_padded_sequence
from utils import AverageMeter, clip_gradient, accuracy

def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch, device, training_parameters):

    print('\nTraining')

    encoder.train()
    decoder.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top_5_accuracies = AverageMeter()

    start = time.time()

    for idx, (images, captions, caption_lengths) in enumerate(train_loader):
        data_time.update(time.time()-start)

        images = images.to(device)
        captions = captions.to(device)
        caption_lengths = caption_lengths.to(device)

        images = encoder(images)
        outputs, sorted_captions, decode_lengths, alphas, sorted_indices = decoder(images, captions, caption_lengths)

        # Remove <start>
        target_captions = sorted_captions[:, 1:]

        # Remove <pad>
        outputs, _,_,_ = pack_padded_sequence(outputs, lengths=decode_lengths, batch_first=True)
        targets, _,_,_ = pack_padded_sequence(target_captions, lengths=decode_lengths, batch_first=True)

        # Calculating loss
        loss = criterion(outputs, targets)

        # Doubly stochastic attention regularization
        loss += training_parameters.alpha_c * ((1 - alphas.sum(dim=1)) ** 2).mean()

        # Back propogation
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if training_parameters.grad_clip is not None:
            clip_gradient(decoder_optimizer, training_parameters.grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, training_parameters.grad_clip)
        
        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Update metrics
        top5 = accuracy(outputs, targets, k=5)
        losses.update(loss.item(), sum(decode_lengths))
        top_5_accuracies.update(top5, sum(decode_lengths))
        batch_time.update(time.time() -  start)

        start = time.time()

        if idx % training_parameters.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                    'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data load time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Top 5 accuracies {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, idx, len(train_loader),
                                                                            batch_time = batch_time,
                                                                            data_time = data_time,
                                                                            loss = losses,
                                                                            top5 = top_5_accuracies))


def validate(validation_loader, encoder, decoder, criterion, word_map, device, training_parameters):

    print('\nValidating')

    decoder.eval()
    if encoder is not None:
        encoder.eval()
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top_5_accuracies = AverageMeter()

    start = time.time()

    true_captions = list()
    predicted_captions = list()

    with torch.no_grad():

        for idx, (images, captions, caption_lengths, all_captions) in enumerate(validation_loader):

            images = images.to(device)
            captions = captions.to(device)
            caption_lengths = caption_lengths.to(device)

            images = encoder(images)
            outputs, sorted_captions, decode_lengths, alphas, sorted_indices = decoder(images, captions, caption_lengths)

            # Remove <start>
            target_captions = sorted_captions[:, 1:]

            # Remove <pad>
            outputs_copy = outputs.clone()
            outputs, _,_,_ = pack_padded_sequence(outputs, lengths=decode_lengths, batch_first=True)
            targets, _,_,_ = pack_padded_sequence(target_captions, lengths=decode_lengths, batch_first=True)

            # Calculating loss
            loss = criterion(outputs, targets)

            # Doubly stochastic attention regularization
            loss += training_parameters.alpha_c * ((1 - alphas.sum(dim=1)) ** 2).mean()

            # Update metrics
            top5 = accuracy(outputs, targets, k=5)
            losses.update(loss.item(), sum(decode_lengths))
            top_5_accuracies.update(top5, sum(decode_lengths))
            batch_time.update(time.time() -  start)

            start = time.time()

            if idx % training_parameters.print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                        'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Top 5 accuracies {top5.val:.3f} ({top5.avg:.3f})'.format(idx, len(validation_loader),
                                                                                batch_time = batch_time,
                                                                                loss = losses,
                                                                                top5 = top_5_accuracies))

            # True captions
            all_captions = all_captions[sorted_indices]
            for i in range(all_captions.size(0)): # loop over batch size
                image_caps = all_captions[i].tolist()
                image_captions = list(
                    map( lambda caption: [word for word in caption if word not in {word_map['<start>'], word_map['<pad>']}],
                                image_caps))
                true_captions.append(image_captions)
        
            # Predicted captions
            _, predictions = torch.max(outputs_copy, dim=2)
            predictions = predictions.tolist()
            temp_predictions = list()
            for j, _ in enumerate(predictions):
                temp_predictions.append(predictions[j][:decode_lengths[j]])
            predictions = temp_predictions
            predicted_captions.extend(predictions)

            assert len(predicted_captions) == len(true_captions)

        bleu4 = corpus_bleu(true_captions, predicted_captions)

        print(
            '\n Loss : {loss.avg:.3f}, Top-5-Accuracies : {top5.val:.3f}, BLEU-4 score : {bleu}'.format(loss=losses,
                                                                                                        top5 = top_5_accuracies,
                                                                                                        bleu = bleu4))

        return bleu4




