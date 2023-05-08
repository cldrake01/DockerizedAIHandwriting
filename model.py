import torch
import torch.nn as nn


class HTR(nn.Module):
    """
    A handwritten text recognition model that uses a convolutional neural network and a transformer-based architecture.

    Args:
        hidden_dims (int): The number of output channels in the convolutional layer.
        numheads (int): The number of heads in the multi-head attention layers of the encoder and decoder.
        numencodelayers (int): The number of layers in the encoder.
        numdecodelayers (int): The number of layers in the decoder.
        maxlinelen (int): The maximum length of the input sequence.

    Attributes:
        resnet (torchvision.models.ResNet): The ResNet backbone.
        convolutional_layer (torch.nn.Conv2d): The convolutional layer applied to the ResNet output.
        encode_layers (torch.nn.TransformerEncoderLayer): The transformer encoder layer used in the encoder.
        encoder (torch.nn.TransformerEncoder): The transformer encoder.
        decode_layers (torch.nn.TransformerEncoderLayer): The transformer encoder layer used in the decoder.
        decoder (torch.nn.TransformerEncoder): The transformer decoder.
        fc (torch.nn.Linear): The fully connected layer that converts the output of the decoder to the final output.
        fc2 (torch.nn.Linear): The fully connected layer that is applied to the flattened output of the convolutional layer.
        fc3 (torch.nn.Linear): The fully connected layer used in the encoder and decoder LSTMs.
        encode (torch.nn.LSTM): The LSTM used in the encoder.
        decode (torch.nn.LSTM): The LSTM used in the decoder.
        pool (torch.nn.AdaptiveMaxPool2d): The adaptive max pooling layer applied to the output of the convolutional layer.
        maxlinelen (int): The maximum length of the input sequence.
    """

    def __init__(self, hidden_dims, numheads, numencodelayers, numdecodelayers, maxlinelen) -> None:
        """
       Initializes the HTR model.

       Args:
           hidden_dims (int): The number of output channels in the convolutional layer.
           numheads (int): The number of heads in the multi-head attention layers of the encoder and decoder.
           numencodelayers (int): The number of layers in the encoder.
           numdecodelayers (int): The number of layers in the decoder.
           maxlinelen (int): The maximum length of the input sequence.
       """

        super().__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        del self.resnet.fc
        self.convolutional_layer = nn.Conv2d(2048, hidden_dims, 1)

        self.encode_layers = nn.TransformerEncoderLayer(hidden_dims, numheads)
        self.encoder = nn.TransformerEncoder(self.encode_layers, numencodelayers)

        self.decode_layers = nn.TransformerEncoderLayer(hidden_dims, numheads)
        self.decoder = nn.TransformerEncoder(self.decode_layers, numdecodelayers)

        self.fc = nn.Linear(256, 257)

        self.fc2 = nn.Linear(256 * 2 * maxlinelen, 256)
        self.fc3 = nn.Linear(512, 256)

        self.encode = nn.LSTM(256, 256, bidirectional=True)
        self.decode = nn.LSTM(256, 256, bidirectional=True)

        self.pool = nn.AdaptiveMaxPool2d(10)

        self.maxlinelen = maxlinelen

    def throughresnet(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        return x

    def forward(self, input_param):
        x = self.throughresnet(input_param)

        x = self.pool(x)

        # x = self.encoder(x)
        # x = self.decoder(x)
        # x = self.fc(x)
        x = x.flatten(1, 3)
        x = self.fc2(x)
        x = x[None, :]

        x, hidden = self.encode(x)
        x = self.fc3(x)

        x, hidden = self.decode(x, hidden)
        x = self.fc3(x)

        out = torch.zeros((self.maxlinelen, 257))
        b = self.fc(x)
        out[0] = b

        for i in range(1, self.maxlinelen):
            x, hidden = self.decode(x, hidden)
            x = self.fc3(x)

            out[i] = self.fc(x)

        return out
