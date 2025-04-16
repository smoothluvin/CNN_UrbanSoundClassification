from torch import nn

class CNNNetwork(nn.module):

    # Constructor
    def __init__(self):
        super().__init__()
        # Creating 4 conv blocks -> flatten layer -> linear layer -> softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                # In channels will take in grey-scale images
                in_channels = 1,
                out_channels = 16,
                kernel_size = 3,
                stride = 1,
                padding = 2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        ),
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                # The in channels for each subsequent layer will be equal to the output of the previous layer
                in_channels = 16,
                out_channels = 32,
                kernel_size = 3,
                stride = 1,
                padding = 2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        ),
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                # The in channels for each subsequent layer will be equal to the output of the previous layer
                in_channels = 32,
                out_channels = 64,
                kernel_size = 3,
                stride = 1,
                padding = 2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        ),
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                # The in channels for each subsequent layer will be equal to the output of the previous layer
                in_channels = 64,
                out_channels = 128,
                kernel_size = 3,
                stride = 1,
                padding = 2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        )
        self.flatten = nn.Flatten()
        
        # (num_inputs = previous_layer_inputs * frequency axis * time axis, num_outputs (All the different classes))
        self.linear = nn.Linear(128 * 5 * 4, 10)

        # Softmax to normalize the result across all the different classes
        self.softmax = nn.Softmax(dim=1)