import torch.nn as nn
import torch
from torchvision import models
from utils import save_net,load_net
from torchsummary import summary
from typing import Optional, List, Any


class CSRNet(nn.Module):
    def __init__(
        self, 
        frontend_feat: List[Any],
        backend_feat: List[int],
        load_weights: Optional[bool]=False
    ) -> None:
        super(CSRNet, self).__init__()

        self.seen = 0

        # Define layer, number mean Conv and 'M' mean maxPooling
        self.frontend_feat = frontend_feat
        self.backend_feat  = backend_feat

        # Make layer from previous infomations
        self.frontend = self.make_layers(self.frontend_feat)
        self.backend = self.make_layers(self.backend_feat,in_channels = 512,dilation = True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        # Init weights
        if not load_weights:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
            for i in range(len(self.frontend.state_dict().items())):
                key = list(self.frontend.state_dict().items())[i][0]
                self.frontend.state_dict()[key].data[:] = list(mod.state_dict().items())[i][1].data[:]


    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    
    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
    def make_layers(
        self, 
        cfg: List[Any], 
        in_channels: Optional[int] = 3,
        batch_norm:Optional[bool]=False,
        dilation:Optional[bool]=False
    ) -> nn.Sequential:
        if dilation:
            d_rate = 2
        else:
            d_rate = 1
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)


if __name__ == '__main__':
    model = CSRNet(FRONTEND_FEAT, BACKEND_FEAT)
    summary(model, (3, 64, 64))