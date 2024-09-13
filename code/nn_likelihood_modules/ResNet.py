from ResNet_component import Bottleneck, BasicBlock, nn

"""
A typial resnet implementation for RGB images
"""
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, model_name, num_classes=10):
        super(ResNet, self).__init__()
        # Self-denied parameters for activation levels registration and other display
        self.model_name = model_name + '_' + str(num_blocks)
        self.regist_actLevel = False
        # ResNet initialization
        self.in_planes = 64
        # Hand made number of hidden layers for the activation levels extraction
        self.nb_hidden_layers = 2
        self.hidden_start_size = 512

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # This input size is specific with AdaptiveAvgPool2d(1,1): Pooling all features after convolution to 1 value
        """
        ResNet original code

        Classification layer:
        self.linear = nn.Linear(512*block.expansion, num_classes)

        Corresponding forward function
        def forward(self, x):
            if self.regist_actLevel:
                actLevel = []
                out = self.relu1(self.bn1(self.conv1(x)))
                out = self.layer1(out)
                out = self.layer2(out)
                out = self.layer3(out)
                out = self.layer4(out)
                out = self.avgpool(out)
                out = out.view(out.size(0), -1)
                # Registration of the activation levels before the last prediction layer
                actLevel.append(out)
                out = self.linear(out)
                return out, actLevel
            else:
                out = self.relu1(self.bn1(self.conv1(x)))
                out = self.layer1(out)
                out = self.layer2(out)
                out = self.layer3(out)
                out = self.layer4(out)
                out = self.avgpool(out)
                out = out.view(out.size(0), -1)
                out = self.linear(out)
                return out
        """
        ## Modified version to have multiple hidden layers
        # Fully Connect
        temp_fully_connect_size = self.hidden_start_size
        linear_hidden_list = []
        linear_hidden_list.append(nn.Sequential(
            nn.Linear(512*block.expansion, temp_fully_connect_size),
            nn.ReLU(),
        ))
        for _ in range(self.nb_hidden_layers-1):
            linear_hidden_list.append(nn.Sequential(
                nn.Linear(temp_fully_connect_size, int(temp_fully_connect_size/2)),
                nn.ReLU(),
            ))
            temp_fully_connect_size = int(temp_fully_connect_size/2)
        self.linear_hiddens = nn.ModuleList(linear_hidden_list)
        self.linear_out = nn.Linear(temp_fully_connect_size, num_classes)
        
        # Relu and average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu1 = nn.ReLU()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def activate_registration(self):
        self.regist_actLevel = True
        
    def deactivate_registration(self):
        self.regist_actLevel = False

    def forward(self, x):
        if self.regist_actLevel:
            actLevel = []
            out = self.relu1(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            for i in range(0, self.nb_hidden_layers):
                out = self.linear_hiddens[i](out)
                # Registration of the activation levels before the last prediction layer
                actLevel.append(out)
            out = self.linear_out(out)
            return out, actLevel
        else:
            out = self.relu1(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            for i in range(0, self.nb_hidden_layers):
                out = self.linear_hiddens[i](out)
            out = self.linear_out(out)
            return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2], 'ResNet18')


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3], 'ResNet34')


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3], 'ResNet50')
