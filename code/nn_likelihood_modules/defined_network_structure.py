"""
Defined neural network structure to be tested
"""

from basic_network_structure import Conv_Block, Conv_Block_with_batch_norm, Basic_Conv2d, Basic_Conv2d_with_batch_norm, Basic_Maxpool2d
from common_imports import *

"""
Structure with symmetric input size
"""

"""
Basic configurable convolutional neural network
"""
# CNN Model by the passed configuration to constructor (with the configuration of the hidden layers)
class CNNConfig_New(nn.Module):
    def __init__(self, depth_image, image_size, output_size, nb_conv_block, nb_hidden_layers, conv_k_size,
                 conv_out_channels_start_size, fully_connect_start_size, batch_norm=False, regist_actLevel=False):
        super().__init__()
        self.model_name = 'CNNConfig' + '_' + str(nb_conv_block) + '_' + str(conv_k_size) + '_' + str(batch_norm) + '_' + str(conv_out_channels_start_size)
        self.depth_image = depth_image
        self.image_size = image_size
        self.nb_hidden_layers = nb_hidden_layers
        self.conv_k_size = conv_k_size
        self.regist_actLevel = regist_actLevel
        # Convolutional Network
        nb_out_channels = conv_out_channels_start_size
        conv_modules = []
        if batch_norm :
            conv_modules.append(Conv_Block_with_batch_norm(depth_image,conv_out_channels_start_size,
                                                           *(conv_k_size,1,int(conv_k_size/2)),*(2,2,0)))
            for _ in range(nb_conv_block-1):
                conv_modules.append(Conv_Block_with_batch_norm(nb_out_channels,nb_out_channels*2,
                                                               *(conv_k_size,1,int(conv_k_size/2)),*(2,2,0)))
                nb_out_channels *= 2
            conv_modules.append(Basic_Conv2d_with_batch_norm(nb_out_channels, nb_out_channels*2,
                                                             *(conv_k_size,1,int(conv_k_size/2))))
        else :
            conv_modules.append(Conv_Block(depth_image,conv_out_channels_start_size,
                                           *(conv_k_size,1,int(conv_k_size/2)),*(2,2,0)))
            for _ in range(nb_conv_block-1):
                conv_modules.append(Conv_Block(nb_out_channels,nb_out_channels*2,
                                               *(conv_k_size,1,int(conv_k_size/2)),*(2,2,0)))
                nb_out_channels *= 2
            conv_modules.append(Basic_Conv2d(nb_out_channels, nb_out_channels*2,
                                             *(conv_k_size,1,int(conv_k_size/2))))
        self.conv_net = nn.Sequential(*conv_modules)
        # Flatten Layer
        self.flatten = nn.Flatten()
        # Fully Connect
        temp_fully_connect_size = fully_connect_start_size
        temp_hidden_list = []
        temp_hidden_list.append(nn.Sequential(
            nn.Linear(self.get_linear_input_size(depth_image, image_size), temp_fully_connect_size),
            nn.ReLU(),
        ))
        for _ in range(nb_hidden_layers-1):
            temp_hidden_list.append(nn.Sequential(
                nn.Linear(temp_fully_connect_size, int(temp_fully_connect_size/2)),
                nn.ReLU(),
            ))
            temp_fully_connect_size = int(temp_fully_connect_size/2)
        self.hidden_layers = nn.ModuleList(temp_hidden_list)
        
        self.finalLinear = nn.Sequential(
            nn.Linear(temp_fully_connect_size, output_size),
            nn.LogSoftmax(dim=1),
        )
        
    def activate_registration(self):
        self.regist_actLevel = True
        
    def deactivate_registration(self):
        self.regist_actLevel = False

    def get_linear_input_size(self, depth_image, image_size):
        rand_input = Variable(torch.rand(1, depth_image, image_size, image_size))
        rand_output = self.conv_net(rand_input)
        linear_input_size = rand_output.view(1,-1).size(1)
        return linear_input_size

    
    def forward(self, x):
        if self.regist_actLevel:
            actLevel = []
            x_reshaped = x.view(-1, self.depth_image, self.image_size, self.image_size)
            x_conv = self.conv_net(x_reshaped)
            x_flatten = self.flatten(x_conv)
            x_hidden_layer = self.hidden_layers[0](x_flatten)
            # Register the activation level after this layer
            actLevel.append(x_hidden_layer)
            for i in range(1, self.nb_hidden_layers):
                x_hidden_layer = self.hidden_layers[i](x_hidden_layer)
                # Register the activation level after this layer
                actLevel.append(x_hidden_layer)
            x_final = self.finalLinear(x_hidden_layer)
            return x_final, actLevel
        else:
            x_reshaped = x.view(-1, self.depth_image, self.image_size, self.image_size)
            x_conv = self.conv_net(x_reshaped)
            x_flatten = self.flatten(x_conv)
            x_hidden_layer = self.hidden_layers[0](x_flatten)
            for i in range(1, self.nb_hidden_layers):
                x_hidden_layer = self.hidden_layers[i](x_hidden_layer)
            x_final = self.finalLinear(x_hidden_layer)
            return x_final

"""
Inception Modules
"""
class Advanced_Stack_block(nn.Module):
    def __init__(self, nb_in_channels, nb_out_channels, nb_conv, kernel_size):
        # Kernel size must be odd number
        super().__init__()
        reduced_size = int(nb_out_channels/4)
        if reduced_size == 0:
            reduced_size = 1
        self.nb_conv = nb_conv
        self.conv_1x1 = Basic_Conv2d_with_batch_norm(nb_in_channels, nb_out_channels, *(1,1,0))

        self.stack_conv = nn.ModuleList([Basic_Conv2d_with_batch_norm(nb_out_channels, nb_out_channels, *(kernel_size,1,int((kernel_size-1)/2))) for i in range(nb_conv)])
    
    def forward(self, x):
        outputs = []
        output_1x1 = self.conv_1x1(x)
        outputs.append(output_1x1)
        output_temp = output_1x1
        for i in range(self.nb_conv):
            output_temp = self.stack_conv[i](output_temp)
            outputs.append(output_temp)

        return torch.cat(outputs,1) #1 parce que c'est depth concat

class Advanced_Inception_stack_3x3_block(nn.Module):
    def __init__(self, nb_in_channels, nb_out_channels, nb_conv_3x3):
        super().__init__()
        self.stack_conv_branch = Advanced_Stack_block(nb_in_channels, nb_out_channels, nb_conv_3x3, 3)
        self.reduc_conv_1x1 = Basic_Conv2d_with_batch_norm((nb_conv_3x3+2)*nb_out_channels, nb_in_channels, *(1,1,0))
        
        self.max_pool = Basic_Maxpool2d(*(3,1,1))
        self.max_pool_conv = Basic_Conv2d_with_batch_norm(nb_in_channels, nb_out_channels, *(1,1,0))
        
        self.last_ReLU_activate = nn.ReLU()
    
    def forward(self, x):
        outputs = []
        
        output_stack_branch = self.stack_conv_branch(x)
        outputs.append(output_stack_branch)

        output_pool = self.max_pool(x)
        output_pool = self.max_pool_conv(output_pool)
        outputs.append(output_pool)

        reduc_input = torch.cat(outputs,1)
        reduc_output = self.reduc_conv_1x1(reduc_input)

        final_output = self.last_ReLU_activate(x + reduc_output)

        return final_output

class Advanced_Inception_stack_3x3(nn.Module):
    def __init__(self, depth_image, image_size, output_size, nb_incep_block, nb_conv_3x3, nb_hidden_layers,
                 fully_connect_start_size, regist_actLevel=False):
        super().__init__()
        self.model_name = 'Inception_' + str(nb_incep_block) + '_' + str(nb_conv_3x3)
        self.depth_image = depth_image
        self.image_size = image_size
        self.nb_hidden_layers = nb_hidden_layers
        self.regist_actLevel = regist_actLevel
        conv_modules = []
        nb_output_channels = 32
        conv_modules.append(Basic_Conv2d_with_batch_norm(depth_image, nb_output_channels, *(1,1,0)))
        for i in range(nb_incep_block):
            conv_modules.append(Advanced_Inception_stack_3x3_block(nb_output_channels,nb_output_channels * 2,nb_conv_3x3))

        self.conv_net = nn.Sequential(*conv_modules)
        self.avg_conv_before_fc_net = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            Basic_Conv2d_with_batch_norm(nb_output_channels, nb_output_channels, *(1,1,0))
        )
        self.flatten = nn.Flatten()
        # Fully Connect
        temp_fully_connect_size = fully_connect_start_size
        temp_hidden_list = []
        temp_hidden_list.append(nn.Sequential(
            nn.Linear(self.get_linear_input_size(depth_image, image_size), temp_fully_connect_size),
            nn.ReLU(),
        ))
        for _ in range(nb_hidden_layers-1):
            temp_hidden_list.append(nn.Sequential(
                nn.Linear(temp_fully_connect_size, int(temp_fully_connect_size/2)),
                nn.ReLU(),
            ))
            temp_fully_connect_size = int(temp_fully_connect_size/2)
        self.hidden_layers = nn.ModuleList(temp_hidden_list)
        
        self.finalLinear = nn.Sequential(
            nn.Linear(temp_fully_connect_size, output_size),
            nn.LogSoftmax(dim=1),
        )
        
    def activate_registration(self):
        self.regist_actLevel = True
        
    def deactivate_registration(self):
        self.regist_actLevel = False

    def get_linear_input_size(self, depth_image, image_size):
        rand_input = Variable(torch.rand(1, depth_image, image_size, image_size))
        rand_output = self.avg_conv_before_fc_net(self.conv_net(rand_input))
        linear_input_size = rand_output.view(1,-1).size(1)
        return linear_input_size

    
    def forward(self, x):
        if self.regist_actLevel:
            actLevel = []
            x_reshaped = x.view(-1, self.depth_image, self.image_size, self.image_size)
            x_conv = self.conv_net(x_reshaped)
            x_avg_conv = self.avg_conv_before_fc_net(x_conv)
            x_flatten = self.flatten(x_avg_conv)
            x_hidden_layer = self.hidden_layers[0](x_flatten)
            # Register the activation level after this layer
            actLevel.append(x_hidden_layer)
            for i in range(1, self.nb_hidden_layers):
                x_hidden_layer = self.hidden_layers[i](x_hidden_layer)
                # Register the activation level after this layer
                actLevel.append(x_hidden_layer)
            x_final = self.finalLinear(x_hidden_layer)
            return x_final, actLevel
        else:
            x_reshaped = x.view(-1, self.depth_image, self.image_size, self.image_size)
            x_conv = self.conv_net(x_reshaped)
            x_avg_conv = self.avg_conv_before_fc_net(x_conv)
            x_flatten = self.flatten(x_avg_conv)
            x_hidden_layer = self.hidden_layers[0](x_flatten)
            for i in range(1, self.nb_hidden_layers):
                x_hidden_layer = self.hidden_layers[i](x_hidden_layer)
            x_final = self.finalLinear(x_hidden_layer)
            return x_final


"""
Structure with asymmetric input size
"""

"""
Basic configurable convolutional neural network
"""
# CNN Model by the passed configuration to constructor (with the configuration of the hidden layers)
class CNNConfig_New_WH(nn.Module):
    def __init__(self, depth_image, image_h_size, image_w_size, output_size, nb_conv_block, nb_hidden_layers, conv_k_size,
                 conv_out_channels_start_size, fully_connect_start_size, batch_norm=False, padding=False, regist_actLevel=False):
        super().__init__()
        self.model_name = 'CNNConfig' + '_' + str(nb_conv_block) + '_' + str(conv_k_size) + '_' + str(batch_norm) + '_' + str(conv_out_channels_start_size)
        self.depth_image = depth_image
        self.image_h_size = image_h_size
        self.image_w_size = image_w_size
        self.nb_hidden_layers = nb_hidden_layers
        self.conv_k_size = conv_k_size
        self.regist_actLevel = regist_actLevel
        # Convolutional Network
        nb_out_channels = conv_out_channels_start_size
        conv_modules = []
        if batch_norm :
            if padding:
                conv_modules.append(Conv_Block_with_batch_norm(depth_image,conv_out_channels_start_size,
                                                            *(conv_k_size,1,int(conv_k_size/2)),*(2,2,0)))
                for _ in range(nb_conv_block-1):
                    conv_modules.append(Conv_Block_with_batch_norm(nb_out_channels,nb_out_channels*2,
                                                                *(conv_k_size,1,int(conv_k_size/2)),*(2,2,0)))
                    nb_out_channels *= 2
                conv_modules.append(Basic_Conv2d_with_batch_norm(nb_out_channels, nb_out_channels*2,
                                                                *(conv_k_size,1,int(conv_k_size/2))))
            else:
                conv_modules.append(Conv_Block_with_batch_norm(depth_image,conv_out_channels_start_size,
                                                            *(conv_k_size,1,0),*(2,2,0)))
                for _ in range(nb_conv_block-1):
                    conv_modules.append(Conv_Block_with_batch_norm(nb_out_channels,nb_out_channels*2,
                                                                *(conv_k_size,1,0),*(2,2,0)))
                    nb_out_channels *= 2
                conv_modules.append(Basic_Conv2d_with_batch_norm(nb_out_channels, nb_out_channels*2,
                                                                *(conv_k_size,1,0)))
        else :
            if padding:
                conv_modules.append(Conv_Block(depth_image,conv_out_channels_start_size,
                                            *(conv_k_size,1,int(conv_k_size/2)),*(2,2,0)))
                for _ in range(nb_conv_block-1):
                    conv_modules.append(Conv_Block(nb_out_channels,nb_out_channels*2,
                                                *(conv_k_size,1,int(conv_k_size/2)),*(2,2,0)))
                    nb_out_channels *= 2
                conv_modules.append(Basic_Conv2d(nb_out_channels, nb_out_channels*2,
                                                *(conv_k_size,1,int(conv_k_size/2))))
            else:
                conv_modules.append(Conv_Block(depth_image,conv_out_channels_start_size,
                                            *(conv_k_size,1,0),*(2,2,0)))
                for _ in range(nb_conv_block-1):
                    conv_modules.append(Conv_Block(nb_out_channels,nb_out_channels*2,
                                                *(conv_k_size,1,0),*(2,2,0)))
                    nb_out_channels *= 2
                conv_modules.append(Basic_Conv2d(nb_out_channels, nb_out_channels*2,
                                                *(conv_k_size,1,0)))
        self.conv_net = nn.Sequential(*conv_modules)
        # Flatten Layer
        self.flatten = nn.Flatten()
        # Fully Connect
        temp_fully_connect_size = fully_connect_start_size
        temp_hidden_list = []
        temp_hidden_list.append(nn.Sequential(
            nn.Linear(self.get_linear_input_size(depth_image, image_h_size, image_w_size), temp_fully_connect_size),
            nn.ReLU(),
        ))
        for _ in range(nb_hidden_layers-1):
            temp_hidden_list.append(nn.Sequential(
                nn.Linear(temp_fully_connect_size, int(temp_fully_connect_size/2)),
                nn.ReLU(),
            ))
            temp_fully_connect_size = int(temp_fully_connect_size/2)
        self.hidden_layers = nn.ModuleList(temp_hidden_list)
        
        self.finalLinear = nn.Sequential(
            nn.Linear(temp_fully_connect_size, output_size),
            nn.LogSoftmax(dim=1),
        )
        
    def activate_registration(self):
        self.regist_actLevel = True
        
    def deactivate_registration(self):
        self.regist_actLevel = False

    def get_linear_input_size(self, depth_image, image_h_size, image_w_size):
        rand_input = Variable(torch.rand(1, depth_image, image_h_size, image_w_size))
        rand_output = self.conv_net(rand_input)
        linear_input_size = rand_output.view(1,-1).size(1)
        return linear_input_size

    
    def forward(self, x):
        if self.regist_actLevel:
            actLevel = []
            x_reshaped = x.view(-1, self.depth_image, self.image_h_size, self.image_w_size)
            x_conv = self.conv_net(x_reshaped)
            x_flatten = self.flatten(x_conv)
            x_hidden_layer = self.hidden_layers[0](x_flatten)
            # Register the activation level after this layer
            actLevel.append(x_hidden_layer)
            for i in range(1, self.nb_hidden_layers):
                x_hidden_layer = self.hidden_layers[i](x_hidden_layer)
                # Register the activation level after this layer
                actLevel.append(x_hidden_layer)
            x_final = self.finalLinear(x_hidden_layer)
            return x_final, actLevel
        else:
            x_reshaped = x.view(-1, self.depth_image, self.image_h_size, self.image_w_size)
            x_conv = self.conv_net(x_reshaped)
            x_flatten = self.flatten(x_conv)
            x_hidden_layer = self.hidden_layers[0](x_flatten)
            for i in range(1, self.nb_hidden_layers):
                x_hidden_layer = self.hidden_layers[i](x_hidden_layer)
            x_final = self.finalLinear(x_hidden_layer)
            return x_final

"""
Inception
"""
class Advanced_Inception_stack_3x3_WH(nn.Module):
    def __init__(self, depth_image,  image_h_size, image_w_size, output_size, nb_incep_block, nb_conv_3x3, nb_hidden_layers,
                 fully_connect_start_size, regist_actLevel=False):
        super().__init__()
        self.model_name = 'Inception_' + str(nb_incep_block) + '_' + str(nb_conv_3x3)
        self.depth_image = depth_image
        self.image_h_size = image_h_size
        self.image_w_size = image_w_size
        self.nb_hidden_layers = nb_hidden_layers
        self.regist_actLevel = regist_actLevel
        conv_modules = []
        nb_output_channels = 32
        conv_modules.append(Basic_Conv2d_with_batch_norm(depth_image, nb_output_channels, *(1,1,0)))
        for i in range(nb_incep_block):
            conv_modules.append(Advanced_Inception_stack_3x3_block(nb_output_channels,nb_output_channels * 2,nb_conv_3x3))

        self.conv_net = nn.Sequential(*conv_modules)
        self.avg_conv_before_fc_net = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            Basic_Conv2d_with_batch_norm(nb_output_channels, nb_output_channels, *(1,1,0))
        )
        self.flatten = nn.Flatten()
        # Fully Connect
        temp_fully_connect_size = fully_connect_start_size
        temp_hidden_list = []
        temp_hidden_list.append(nn.Sequential(
            nn.Linear(self.get_linear_input_size(depth_image, image_h_size, image_w_size), temp_fully_connect_size),
            nn.ReLU(),
        ))
        for _ in range(nb_hidden_layers-1):
            temp_hidden_list.append(nn.Sequential(
                nn.Linear(temp_fully_connect_size, int(temp_fully_connect_size/2)),
                nn.ReLU(),
            ))
            temp_fully_connect_size = int(temp_fully_connect_size/2)
        self.hidden_layers = nn.ModuleList(temp_hidden_list)
        
        self.finalLinear = nn.Sequential(
            nn.Linear(temp_fully_connect_size, output_size),
            nn.LogSoftmax(dim=1),
        )
        
    def activate_registration(self):
        self.regist_actLevel = True
        
    def deactivate_registration(self):
        self.regist_actLevel = False

    def get_linear_input_size(self, depth_image, image_h_size, image_w_size):
        rand_input = Variable(torch.rand(1, depth_image, image_h_size, image_w_size))
        rand_output = self.avg_conv_before_fc_net(self.conv_net(rand_input))
        linear_input_size = rand_output.view(1,-1).size(1)
        return linear_input_size

    
    def forward(self, x):
        if self.regist_actLevel:
            actLevel = []
            x_reshaped = x.view(-1, self.depth_image, self.image_h_size, self.image_w_size)
            x_conv = self.conv_net(x_reshaped)
            x_avg_conv = self.avg_conv_before_fc_net(x_conv)
            x_flatten = self.flatten(x_avg_conv)
            x_hidden_layer = self.hidden_layers[0](x_flatten)
            # Register the activation level after this layer
            actLevel.append(x_hidden_layer)
            for i in range(1, self.nb_hidden_layers):
                x_hidden_layer = self.hidden_layers[i](x_hidden_layer)
                # Register the activation level after this layer
                actLevel.append(x_hidden_layer)
            x_final = self.finalLinear(x_hidden_layer)
            return x_final, actLevel
        else:
            x_reshaped = x.view(-1, self.depth_image, self.image_h_size, self.image_w_size)
            x_conv = self.conv_net(x_reshaped)
            x_avg_conv = self.avg_conv_before_fc_net(x_conv)
            x_flatten = self.flatten(x_avg_conv)
            x_hidden_layer = self.hidden_layers[0](x_flatten)
            for i in range(1, self.nb_hidden_layers):
                x_hidden_layer = self.hidden_layers[i](x_hidden_layer)
            x_final = self.finalLinear(x_hidden_layer)
            return x_final

"""
Simple fully-connected neural network structure
"""
class Config_FCNN_Rect(nn.Module):
    def __init__(self, input_size, hidden_size, nb_hidden_layers, nb_classes, batch_norm=False, mish_activation=False):
        """
        Configurable Fully-connected neural network
        """
        super().__init__()
        # Model attributes
        self.model_name = 'FCNN_' + str(hidden_size) + '_' + str(nb_hidden_layers)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nb_hidden_layers = nb_hidden_layers
        self.nb_classes = nb_classes
        """
        Added attribute for extracting activation levels
        """
        self.regist_actLevel = False
        """
        """
        if batch_norm:
            self.model_name = self.model_name + '_batch_norm'  
        # Fully Connect
        temp_hidden_list = []
        if batch_norm:
            if mish_activation:
                    temp_hidden_list.append(nn.Sequential(
                        nn.Linear(input_size, hidden_size),
                        nn.BatchNorm1d(hidden_size, eps=0.00001),
                        nn.Mish(),
                    ))
            else:
                temp_hidden_list.append(nn.Sequential(
                        nn.Linear(input_size, hidden_size),
                        nn.BatchNorm1d(hidden_size, eps=0.00001),
                        nn.ReLU(),
                    ))
        else:
            if mish_activation:
                temp_hidden_list.append(nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.Mish(),
                ))
            else:
                temp_hidden_list.append(nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                ))
        for _ in range(nb_hidden_layers-1):
            if batch_norm:
                if mish_activation:
                    temp_hidden_list.append(nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        nn.BatchNorm1d(hidden_size, eps=0.00001),
                        nn.Mish(),
                    ))
                else:
                    temp_hidden_list.append(nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        nn.BatchNorm1d(hidden_size, eps=0.00001),
                        nn.ReLU(),
                    ))
            else:
                if mish_activation:
                    temp_hidden_list.append(nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        nn.Mish(),
                    ))
                else:
                    temp_hidden_list.append(nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                    ))
        self.hidden_layers = nn.ModuleList(temp_hidden_list)
        
        self.finalLinear = nn.Sequential(
            nn.Linear(hidden_size, nb_classes),
            nn.LogSoftmax(dim=1),
        )

    """
    Added function to extract activation levels
    """
    def activate_registration(self):
        self.regist_actLevel = True
        
    def deactivate_registration(self):
        self.regist_actLevel = False
    """
    """
    
    def forward(self, x):
        """
        Code to modify in order to extract activation levels
        """
        if self.regist_actLevel:
            actLevel = []
            x_hidden_layer = x.view(-1, self.input_size)
            for i in range(0, self.nb_hidden_layers):
                x_hidden_layer = self.hidden_layers[i](x_hidden_layer)
                # Register the activation levels 
                actLevel.append(x_hidden_layer)

            x_final = self.finalLinear(x_hidden_layer)
            return x_final, actLevel
        else:
            x_hidden_layer = x.view(-1, self.input_size)
            for i in range(0, self.nb_hidden_layers):
                x_hidden_layer = self.hidden_layers[i](x_hidden_layer)
            x_final = self.finalLinear(x_hidden_layer)
            return x_final
        
    
class Config_FCNN(nn.Module):
    def __init__(self, input_size, fully_connect_start_size, nb_hidden_layers, nb_classes, batch_norm=False):
        """
        Configurable Fully-connected neural network
        """
        super().__init__()
        # Model attributes
        self.model_name = 'FCNN_' + str(fully_connect_start_size) + '_' + str(nb_hidden_layers)
        self.input_size = input_size
        self.fully_connect_start_size = fully_connect_start_size
        self.nb_hidden_layers = nb_hidden_layers
        self.nb_classes = nb_classes
        if batch_norm:
            self.model_name = self.model_name + '_batch_norm'  
        # Fully Connect
        temp_fully_connect_size = fully_connect_start_size
        temp_hidden_list = []
        if batch_norm:
            temp_hidden_list.append(nn.Sequential(
                nn.Linear(input_size, temp_fully_connect_size),
                nn.BatchNorm1d(temp_fully_connect_size, eps=0.00001),
                nn.ReLU(),
            ))
        else:
            temp_hidden_list.append(nn.Sequential(
                nn.Linear(input_size, temp_fully_connect_size),
                nn.ReLU(),
            ))
        for _ in range(nb_hidden_layers-1):
            if batch_norm:
                temp_hidden_list.append(nn.Sequential(
                    nn.Linear(temp_fully_connect_size, int(temp_fully_connect_size/2)),
                    nn.BatchNorm1d(int(temp_fully_connect_size/2), eps=0.00001),
                    nn.ReLU(),
                ))
            else:
                temp_hidden_list.append(nn.Sequential(
                    nn.Linear(temp_fully_connect_size, int(temp_fully_connect_size/2)),
                    nn.ReLU(),
                ))
            temp_fully_connect_size = int(temp_fully_connect_size/2)
        self.hidden_layers = nn.ModuleList(temp_hidden_list)
        
        self.finalLinear = nn.Sequential(
            nn.Linear(temp_fully_connect_size, nb_classes),
            nn.LogSoftmax(dim=1),
        )
    
    def forward(self, x):
        x_hidden_layer = x.view(-1, self.input_size)
        for i in range(0, self.nb_hidden_layers):
            x_hidden_layer = self.hidden_layers[i](x_hidden_layer)
        x_final = self.finalLinear(x_hidden_layer)
        return x_final
    
class FCNN_SL(nn.Module):
    def __init__(self, input_size, nb_classes):
        """
        Single layered fully-connected neural network
        """
        super().__init__()
        # Model attributes
        self.model_name = 'FCNN_Single_Layer'
        self.input_size = input_size
        self.nb_classes = nb_classes
        # Fully Connect
        self.SLLinear = nn.Sequential(
            nn.Linear(input_size, nb_classes),
            nn.LogSoftmax(dim=1),
        )
    
    def forward(self, x):
        x_reshaped = x.view(-1, self.input_size)
        x_final = self.SLLinear(x_reshaped)
        return x_final