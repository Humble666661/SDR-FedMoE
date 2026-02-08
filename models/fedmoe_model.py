import math
import torch
import torch.nn as nn


def init_layer(layer):
    if layer.weight.ndimension() == 4:
        n_out, n_in, height, width = layer.weight.size()
        n = n_in * height * width
    elif layer.weight.ndimension() == 3:
        n_out, n_in, height = layer.weight.size()
        n = n_in * height
    else:
        n_out, n = layer.weight.size()

    std = math.sqrt(2.0 / n)
    scale = std * math.sqrt(3.0)
    layer.weight.data.uniform_(-scale, scale)
    if layer.bias is not None:
        layer.bias.data.fill_(0.0)


def init_bn(bn):
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


class ConvBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, pad, normalisation, dil=1):
        super().__init__()
        self.norm = normalisation
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel,
            stride=stride,
            padding=pad,
            dilation=dil,
        )
        if self.norm == "bn":
            self.bn1 = nn.BatchNorm1d(out_channels)
        elif self.norm == "wn":
            self.conv1 = nn.utils.weight_norm(self.conv1, name="weight")
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        if hasattr(self, "bn1"):
            init_bn(self.bn1)

    def forward(self, x):
        if self.norm == "bn":
            x = self.relu(self.bn1(self.conv1(x)))
        else:
            x = self.relu(self.conv1(x))
        return x


class FullyConnected(nn.Module):
    def __init__(self, in_channels, out_channels, activation, normalisation, att=None):
        super().__init__()
        self.att = att
        self.norm = normalisation
        self.fc = nn.Linear(in_features=in_channels, out_features=out_channels)

        if activation == "sigmoid":
            self.act = nn.Sigmoid()
            self.norm = None
        elif activation == "softmax":
            self.act = nn.Softmax(dim=-1)
            self.norm = None
        elif activation == "global":
            self.act = None
            self.norm = None
        else:
            self.act = nn.ReLU()
            if self.norm == "bn":
                self.bnf = nn.BatchNorm1d(out_channels)
            elif self.norm == "wn":
                self.wnf = nn.utils.weight_norm(self.fc, name="weight")

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc)
        if self.norm == "bn":
            init_bn(self.bnf)

    def forward(self, x):
        if self.norm is not None:
            if self.norm == "bn":
                x = self.act(self.bnf(self.fc(x)))
            else:
                x = self.act(self.wnf(x))
        else:
            if self.act:
                x = self.act(self.fc(x))
            else:
                x = self.fc(x)
        return x


class Expert(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=512, output_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class GatingNetwork(nn.Module):
    def __init__(self, input_dim=256, num_experts=6):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_experts)
        self.bn = nn.BatchNorm1d(num_experts)

    def forward(self, x):
        logits = self.fc(x)
        logits = self.bn(logits)
        weights = torch.nn.functional.softmax(logits, dim=-1)
        return weights


class GatingNetwork_Head(nn.Module):
    def __init__(self, input_dim=256, num_experts=2):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        logits = self.fc(x)
        weights = torch.nn.functional.softmax(logits, dim=-1)
        return weights


class SDR_CNN_FedMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv_1 = ConvBlock1d(80, 128, 7, 1, 3, "bn")
        self.Pool_1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        self.Conv_2 = ConvBlock1d(128, 256, 7, 1, 3, "bn")
        self.Pool_2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        self.LSTM = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
        )

        self.num_experts_shared = 6
        self.num_experts_personalized = 6
        self.gate = GatingNetwork(
            input_dim=256,
            num_experts=self.num_experts_shared + self.num_experts_personalized,
        )

        self.shared_experts = nn.ModuleList([Expert() for _ in range(self.num_experts_shared)])
        self.personalized_experts = nn.ModuleList([Expert() for _ in range(self.num_experts_personalized)])

        self.Drop = nn.Dropout(0.4)
        self.Drop_shared = nn.Dropout(0.4)

        self.FC_1 = FullyConnected(256, 128, "relu", None)
        self.FC_2 = FullyConnected(128, 64, "relu", None)
        self.FC_3 = FullyConnected(64, 1, "global", None)

        self.FC_1_shared = FullyConnected(256, 128, "relu", None)
        self.FC_2_shared = FullyConnected(128, 64, "relu", None)
        self.FC_3_shared = FullyConnected(64, 1, "global", None)

        self.Gating = GatingNetwork_Head()

    def forward(self, x):
        B, F, W = x.shape
        x_personalized = self.Conv_1(x)
        x_personalized = self.Pool_1(x_personalized)
        x_personalized = self.Conv_2(x_personalized)
        x_personalized = self.Pool_2(x_personalized)
        x_personalized = torch.transpose(x_personalized, 1, 2)
        x_personalized, _ = self.LSTM(x_personalized)
        x_personalized = x_personalized[:, -1, :].reshape(B, -1)

        gating = self.gate(x_personalized)
        shared_outputs = [expert(x_personalized) for expert in self.shared_experts]
        personalized_outputs = [expert(x_personalized) for expert in self.personalized_experts]
        experts_outputs = shared_outputs + personalized_outputs

        moe_output = 0
        for i in range(self.num_experts_shared + self.num_experts_personalized):
            moe_output += torch.reshape(gating[:, i], [B, 1]).expand_as(experts_outputs[i]) * experts_outputs[i]

        x_shared = self.FC_1_shared(moe_output)
        x_shared = self.FC_2_shared(x_shared)
        x_shared = self.Drop_shared(x_shared)
        x_shared = self.FC_3_shared(x_shared)

        gating_w = self.Gating(moe_output)

        x_personalized = self.FC_1(moe_output)
        x_personalized = self.FC_2(x_personalized)
        x_personalized = self.Drop(x_personalized)
        x_personalized = self.FC_3(x_personalized)

        w_shared = gating_w[:, 0].unsqueeze(-1)
        w_personalized = gating_w[:, 1].unsqueeze(-1)
        x_fusion = w_shared * x_shared + w_personalized * x_personalized
        x_fusion = torch.sigmoid(x_fusion)
        return x_fusion
