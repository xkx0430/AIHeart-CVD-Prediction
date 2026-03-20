import torch
import torch.nn as nn


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class RAW_MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(RAW_MLP, self).__init__()
        layers = list()
        sizes = [input_size] + hidden_sizes
        for input_size, output_size in zip(sizes[:-1], sizes[1:]):
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.Tanhshrink())
            layers.append(nn.LayerNorm(output_size))
            layers.append(nn.Dropout(0.1))
        self.layers = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=1.0)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.layers(x)
        return x

class DUAL_MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(DUAL_MLP, self).__init__()
        hz_layers, ts_layers = list(), list()
        sizes = [input_size] + hidden_sizes
        for _size, size_ in zip(sizes[:-1], sizes[1:]):
            hz_layers.append(nn.Linear(_size, size_))
            hz_layers.append(nn.Tanhshrink())
            hz_layers.append(nn.LayerNorm(size_))
            hz_layers.append(nn.Dropout(0.1))
            ts_layers.append(nn.Linear(_size, size_))
            ts_layers.append(nn.Tanhshrink())
            ts_layers.append(nn.LayerNorm(size_))
            ts_layers.append(nn.Dropout(0.1))
        self.hz_layers = nn.Sequential(*hz_layers)
        self.ts_layers = nn.Sequential(*ts_layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=1.0)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        hz = self.hz_layers(x)
        ts = self.ts_layers(x)
        return torch.hstack((hz, ts))
