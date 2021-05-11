import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn import utils

from models.discriminators.resblocks import Block
from models.discriminators.resblocks import OptimizedBlock


class SNResNetProjectionDiscriminator(nn.Module):

    def __init__(self, num_features=64, num_classes=0, activation=F.relu, transform_space=None):
        super(SNResNetProjectionDiscriminator, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.activation = activation

        self.block1 = OptimizedBlock(3, num_features)
        self.block2 = Block(num_features, num_features * 2,
                            activation=activation, downsample=True)
        self.block3 = Block(num_features * 2, num_features * 4,
                            activation=activation, downsample=True)
        self.block4 = Block(num_features * 4, num_features * 8,
                            activation=activation, downsample=True)
        self.block5 = Block(num_features * 8, num_features * 16,
                            activation=activation, downsample=True)
        self.l6 = utils.spectral_norm(nn.Linear(num_features * 16, 1))

        if transform_space is not None:
            # self.transformer = nn.Transformer(d_model=1024,nhead=4,num_encoder_layers=4).encoder
            encoder_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=8)
            transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
            self.transformer = transformer_encoder

            if transform_space == 'batch':
                self.transform_space = 'batch'
            elif transform_space == 'embeddings':
                self.transform_space = 'embeddings'
            else:
                print ("You specified a transform space but did not provide an appropriate one. Exiting")
                exit()
        else:
            #transform_space is None
            self.transform_space = None

        if num_classes > 0:
            self.l_y = utils.spectral_norm(nn.Embedding(num_classes, num_features * 16))

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.l6.weight.data)
        optional_l_y = getattr(self, 'l_y', None)
        if optional_l_y is not None:
            init.xavier_uniform_(optional_l_y.weight.data)

    def forward(self, x, y=None):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))
        output = self.l6(h)

        if y is not None:

            if self.transform_space == 'embeddings':

                emb_weight = self.l_y.weight.clone() #NxD
                emb_weight = emb_weight.unsqueeze(0) #1xNxD
                emb_weight = emb_weight.permute(1,0,2)

                transformed_emb_weights = self.transformer(emb_weight) #Nx1xD
                transformed_emb_weights = transformed_emb_weights.squeeze(1) #NxD
                
                # print (y.shape) # (BxN)
                # exit()
                # self.l_y.weight = transformed_emb_weights

                # y_feats = self.l_y(y)
                y_feats = torch.nn.functional.embedding(y, transformed_emb_weights)
                output += torch.sum(y_feats * h, dim=1, keepdim=True)

            elif self.transform_space == 'batch':
                y_feats = self.l_y(y)
                y_feats = y_feats.unsqueeze(0)
                y_feats = self.transformer(y_feats).squeeze(0)
                output += torch.sum(y_feats * h, dim=1, keepdim=True)

            else:
                y_feats = self.l_y(y)
                output += torch.sum(y_feats * h, dim=1, keepdim=True)

        return output


class SNResNetConcatDiscriminator(nn.Module):

    def __init__(self, num_features, num_classes, activation=F.relu,
                 dim_emb=128):
        super(SNResNetConcatDiscriminator, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.dim_emb = dim_emb
        self.activation = activation

        self.block1 = OptimizedBlock(3, num_features)
        self.block2 = Block(num_features, num_features * 2,
                            activation=activation, downsample=True)
        self.block3 = Block(num_features * 2, num_features * 4,
                            activation=activation, downsample=True)
        if num_classes > 0:
            self.l_y = utils.spectral_norm(nn.Embedding(num_classes, dim_emb))
        self.block4 = Block(num_features * 4 + dim_emb, num_features * 8,
                            activation=activation, downsample=True)
        self.block5 = Block(num_features * 8, num_features * 16,
                            activation=activation, downsample=True)
        self.l6 = utils.spectral_norm(nn.Linear(num_features * 16, 1))

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.l6.weight.data)
        if hasattr(self, 'l_y'):
            init.xavier_uniform_(self.l_y.weight.data)

    def forward(self, x, y=None):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        if y is not None:
            emb = self.l_y(y).unsqueeze(-1).unsqueeze(-1)
            emb = emb.expand(emb.size(0), emb.size(1), h.size(2), h.size(3))
            h = torch.cat((h, emb), dim=1)
        h = self.block4(h)
        h = self.block5(h)
        h = torch.sum(self.activation(h), dim=(2, 3))
        return self.l6(h)
