import sys
import torch
import torchvision
from torch import nn

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    
    def __init__(self, encoded_image_size = 14):
        super(Encoder, self).__init__()
        self.encoded_image_size = encoded_image_size

        resnet = torchvision.models.resnet101(pretrained=True)

        # print(resnet)
        # sys.exit()

        # Removing linear and pooling layers
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()

    def fine_tune(self, fine_tune=True):
        '''
        :param fine_tune : Allow gradient computation for convolutional blocks 2 to 4 in encoder
        '''
        
        for p in self.resnet.parameters():
            p.requires_grad = False

        for c in list(self.resnet.children())[:5]:
            for p in c.parameters():
                p.requires_grad = fine_tune

    def forward(self, images):

        '''
        :param images : this will be a tensor of dimension (batch size, 3, image_size, image_size)
        :return : encoded images
        '''

        out = self.resnet(images) # out -> (batch_size, 3, image_size/32, image_size/32)
        out = self.adaptive_pool(out) # out -> (batch_size, 3, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1) # out -> (batch_size, encoded_image_size, encoded_image_size, 3)

        return out

class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dimension, hidden_dimension, attention_dimension):
        """
        :param encoder_dimension: feature size of encoded images
        :param hidden_dimension: size of decoder's RNN
        :param attention_dimension: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dimension, attention_dimension)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(hidden_dimension, attention_dimension)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dimension, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_output, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_output)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)

        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_output * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class Decoder(nn.Module):
    
    def __init__(self, attention_dimension, embedding_dimension, hidden_dimension, 
                    vocab_size, device, context_vector_dimension, encoder_dimension=2048, dropout = 0.5,):
        '''
        :param embedding_dimension : embedding size
        :param hidden_dimension : size of the decoder's RNN
        :param vocab_size : size of the vocabulary
        :param encoder_dimension : feature size of encoded image
        :param dropout : dropout
        '''

        super(Decoder, self).__init__()

        self.embedding_dimension = embedding_dimension
        self.hidden_dimension = hidden_dimension
        self.vocab_size = vocab_size
        self.encoder_dimension = encoder_dimension
        self.context_vector_dimension = context_vector_dimension 
        self.dropout = dropout
        self.device = device

        self.attention = Attention(self.encoder_dimension, self.hidden_dimension, attention_dimension)
        
        # Decoding LSTMCell
        self.decode_step = nn.LSTMCell(input_size = self.embedding_dimension + self.encoder_dimension, hidden_size = self.hidden_dimension, bias = True) 

        # Linear layer to find scores across vocabulary
        self.fc = nn.Linear(in_features = self.hidden_dimension, out_features = self.vocab_size) 

        # Embedding layer
        self.embedding = nn.Embedding(num_embeddings = self.vocab_size, embedding_dim = self.embedding_dimension)

        # Activation
        self.softmax = nn.Softmax(dim=1)
        
        # Linear layers to find initial hidden and cell state of LSTMCell
        self.init_h = nn.Linear(encoder_dimension, hidden_dimension)
        self.init_c = nn.Linear(encoder_dimension, hidden_dimension)

        # linear layer to create a sigmoid-activated gate
        self.f_beta = nn.Linear(hidden_dimension, encoder_dimension) 
        self.sigmoid = nn.Sigmoid()

        # Dropout
        self.dropout = nn.Dropout(p=self.dropout)

        # Initialize some layers with uniform distribution
        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune = True):
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_output):
        '''
        Create initial hidden and cell state for dcoder's LSTMCell
        
        :param encoer_output : encoded images, a tensor of dimension(batch_size, num_pixels, encoder_dimension)
        :return : hidden_state, cell_state
        '''

        mean_encoder_output = encoder_output.mean(dim=1)
        h = self.init_h(mean_encoder_output) #(batch_size, hidden_dimension)
        c = self.init_c(mean_encoder_output)

        return h, c

    def forward(self, encoder_output, encoded_captions, caption_lengths, context = None):
        '''
        :param encoder_output : encoded image, a tensor of size (batch size, encoded_image_size, encoded_image_size, encoder_dimension)
        :param encoded_captions : encoded captions, a tensor of size (batch_size, max_caption_length)
        :param caption_lengths : caption lengths, a tensor of size (batch size, 1)
        :param context : context is 1 if target is a positive caption, 0 if otherwise( ie. negative caotion)
        :return : socres for vocabulary, sorted encoded captions, decode length, weights, sort indices 
        '''

        batch_size = encoder_output.size(0)

        # Flatten image
        encoder_output = encoder_output.view(batch_size, -1, self.encoder_dimension)
        number_of_pixels = encoder_output.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_indices = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_output = encoder_output[sort_indices]
        encoded_captions = encoded_captions[sort_indices]
        
        # Initialize LSTM states
        hidden_state, cell_state = self.init_hidden_state(encoder_output)
        
        embeded_captions = self.embedding(encoded_captions)

        decode_lengths = (caption_lengths - 1).tolist()

        outputs = torch.empty((batch_size, max(decode_lengths), self.vocab_size)).to(self.device)
        alphas = torch.zeros((batch_size, max(decode_lengths), number_of_pixels)).to(self.device)

        # Initialize context vector
        context_vextor = torch.tensor([[context] * self.context_vector_dimension] * batch_size)
        context_vextor = context_vextor.to(self.device)

        print(context_vextor.shape)

        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_output[ : batch_size_t],
                                                                hidden_state[ : batch_size_t])

            print(embeded_captions[: batch_size_t, t, :].shape, attention_weighted_encoding.shape)
            return 1, 1, 1, 1, 1
            
            gate = self.sigmoid(self.f_beta(hidden_state[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding

            hidden_state, cell_state = self.decode_step(
                                        torch.cat([embeded_captions[: batch_size_t, t, :], attention_weighted_encoding], dim=1),
                                        (hidden_state[:batch_size_t], cell_state[:batch_size_t]))

            prediction = self.fc(self.dropout(hidden_state))
            outputs[:batch_size_t, t, :] = prediction
            alphas[:batch_size_t, t, :] = alpha

        return outputs, encoded_captions, decode_lengths, alphas, sort_indices
