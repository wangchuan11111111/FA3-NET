import torch
import torch.nn as nn

from methods.utils.conformer.modules import Transpose
from methods.utils.model_utilities import (ResNet, AGCA, AGCARepeater, DWResNet, ASPModule)
from methods.utils.conformer.encoder import ConformerBlocks


class ConvConformer(nn.Module):
    def __init__(self, cfg, dataset):
        super().__init__()
        self.cfg = cfg
        self.num_classes = dataset.num_classes
        self.conv_block1 = nn.Sequential(
            ResNet(in_channel=7, out_channel=32),
            nn.AvgPool2d(kernel_size=(1, 2)),
        )
        self.conv_block2 = nn.Sequential(
            ResNet(in_channel=32, out_channel=64),
            nn.AvgPool2d(kernel_size=(1, 2)),
        )
        self.conv_block3 = nn.Sequential(
            ResNet(in_channel=64, out_channel=128),
            nn.AvgPool2d(kernel_size=(1, 2)),
        )
        self.conv_block4 = nn.Sequential(
            ResNet(in_channel=128, out_channel=256),
            nn.AvgPool2d(kernel_size=(2, 2)),
        )

        self.conformer_block = ConformerBlocks(encoder_dim=256, num_layers=2, num_attention_heads=4)

        self.fc_sed_track1 = nn.Linear(256, self.num_classes, bias=True)
        self.fc_sed_track2 = nn.Linear(256, self.num_classes, bias=True)
        self.fc_sed_track3 = nn.Linear(256, self.num_classes, bias=True)
        self.fc_doa_track1 = nn.Linear(256, 3, bias=True)
        self.fc_doa_track2 = nn.Linear(256, 3, bias=True)
        self.fc_doa_track3 = nn.Linear(256, 3, bias=True)
        self.final_act_sed = nn.Sequential()  # nn.Sigmoid()
        self.final_act_doa = nn.Tanh()

    def forward(self, x: torch.Tensor):
        """
        x: spectrogram, (batch_size, num_channels, num_frames, num_freqBins)
        """

        # Conv
        ###conv1
        x_conv1 = self.conv_block1(x)

        ###conv2
        x_conv2 = self.conv_block2(x_conv1)

        ###conv3
        x_conv3 = self.conv_block3(x_conv2)

        ###conv4
        x_conv4 = self.conv_block4(x_conv3)

        x_conv4 = x_conv4.mean(dim=3)  # (N, C, T)
        x_conv4 = x_conv4.permute(0, 2, 1)  # (N, T, C)

        x_conformer = self.conformer_block(x_conv4)
        x_conformer = self.attentionpooling(x_conformer)
        # x_conformer = self.pooling(x_conformer.transpose(1, 2)).transpose(1, 2)

        # # fc
        x_sed_1 = self.final_act_sed(self.fc_sed_track1(x_conformer))
        x_sed_2 = self.final_act_sed(self.fc_sed_track2(x_conformer))
        x_sed_3 = self.final_act_sed(self.fc_sed_track3(x_conformer))
        x_sed = torch.stack((x_sed_1, x_sed_2, x_sed_3), 2)
        x_doa_1 = self.final_act_doa(self.fc_doa_track1(x_conformer))
        x_doa_2 = self.final_act_doa(self.fc_doa_track2(x_conformer))
        x_doa_3 = self.final_act_doa(self.fc_doa_track3(x_conformer))
        x_doa = torch.stack((x_doa_1, x_doa_2, x_doa_3), 2)
        output = {
            'sed': x_sed,
            'doa': x_doa,
        }

        return output
