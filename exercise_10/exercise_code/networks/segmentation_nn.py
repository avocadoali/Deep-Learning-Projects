"""SegmentationNN"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23, hp=None):
        super().__init__()
        self.hp = hp
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        self.num_classes = num_classes
        self.padding = self.hp['padding']
        self.stride_down= self.hp['stride_down']
        self.stride_up= self.hp['stride_up']
    
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        # Downsampling layers
        down_block = []
        down_block.append(self.down_block(3, 64 ))
        down_block.append(self.down_block(64, 128))
        down_block.append(self.down_block(128, 256))
        self.down_layers = nn.ModuleList(down_block)
 
        # Upsampling layers
        up_block = []
        up_block.append(self.up_block(256, 128))
        up_block.append(self.up_block(128, 64))
        self.up_layers = nn.ModuleList(up_block)

        # Output LayeJr
        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4 , stride=self.stride_up, padding=self.padding),
            nn.Conv2d(32, self.num_classes, kernel_size=1), 
        )

    def down_block(self, input_channel, output_channel):
        block = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=3 , stride=self.stride_down, padding=self.padding),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(), 
            nn.Conv2d(output_channel, output_channel, kernel_size=3 , stride=self.stride_down, padding=self.padding),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(), 
            nn.Conv2d(output_channel, output_channel, kernel_size=3 , stride=self.stride_down, padding=self.padding),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(), 
            nn.MaxPool2d(2, 2)
        )

        return block

    def up_block(self, input_channel, output_channel):
        block = nn.Sequential(
            nn.ConvTranspose2d(input_channel, output_channel, kernel_size=4 , stride=self.stride_up, padding=self.padding),
            nn.ReLU(), 
            nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=1, padding = self.padding),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(), 
            nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=1, padding = self.padding),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(), 
        )
        return block

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directlyprint(device).

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        skip_connections = []

        for down_layer in self.down_layers:
            x = down_layer(x)
            skip_connections.append(x)


        for up_layer in self.up_layers:
            skip_connection = skip_connections.pop()
            # print()
            # print(x.shape)
            # print(skip_connection.shape)
            x_add = skip_connection + x
            # print(x_add.shape)
            x = up_layer(x_add)
            # x = up_layer(x)


        x = self.output_layer(x)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    # @property
    # def is_cuda(self):
    #     """
    #     Check if model parameters are allocated on the GPU.
    #     """
    #     return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

        
class DummySegmentationModel(nn.Module):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()

if __name__ == "__main__":
    from torchinfo import summary
    summary(SegmentationNN(), (1, 3, 240, 240), device="cpu")