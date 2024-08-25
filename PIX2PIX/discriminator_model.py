import torch
import torch.nn as nn

# CONVOLUTIONAL NEURAL NETWORK BLOCK
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            # CONVOLUTIONAL LAYER WITH REFLECTION PADDING FOR BETTER EDGE HANDLING
            nn.Conv2d(in_channels, out_channels, 4, stride, bias=False, padding_mode="reflect"),
            # BATCH NORMALIZATION FOR STABLE TRAINING
            nn.BatchNorm2d(out_channels),
            # LEAKY RELU ACTIVATION TO ALLOW SMALL NEGATIVE VALUES
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        # APPLY THE CONVOLUTIONAL SEQUENCE TO THE INPUT
        return self.conv(x)

# DISCRIMINATOR MODEL
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        # INITIAL LAYER TO PROCESS CONCATENATED INPUT
        self.initial = nn.Sequential(
            # CONVOLUTIONAL LAYER FOR INITIAL FEATURE EXTRACTION
            nn.Conv2d(in_channels * 2, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            # LEAKY RELU ACTIVATION
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            # ADD CNN BLOCKS WITH INCREASING FEATURE SIZES
            layers.append(
                CNNBlock(in_channels, feature, stride=1 if feature == features[-1] else 2),
            )
            in_channels = feature

        # FINAL CONVOLUTIONAL LAYER TO PRODUCE SINGLE-CHANNEL OUTPUT
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"))

        # COMBINE ALL LAYERS INTO A SEQUENTIAL MODEL
        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        # CONCATENATE INPUT IMAGES ALONG THE CHANNEL DIMENSION
        x = torch.cat([x, y], dim=1)
        # PASS THROUGH INITIAL LAYER
        x = self.initial(x)
        # PASS THROUGH MAIN MODEL
        x = self.model(x)
        return x

# TEST FUNCTION TO VERIFY MODEL ARCHITECTURE
def test():
    # CREATE RANDOM INPUT TENSORS
    x = torch.randn((1, 3, 256, 256))
    y = torch.randn((1, 3, 256, 256))

    # INSTANTIATE THE DISCRIMINATOR MODEL
    model = Discriminator()

    # FORWARD PASS THROUGH THE MODEL
    preds = model(x, y)
    # PRINT THE SHAPE OF THE OUTPUT
    print(preds.shape)

# EXECUTE TEST FUNCTION IF SCRIPT IS RUN DIRECTLY
if __name__ == "__main__":
    test()