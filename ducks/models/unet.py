from diffusers import UNet1DModel

def get_unnet1dmodel(
    sample_size=1024,
    in_channel=11,
    out_channel=11,
    layers_per_block=2,
    block_out_channels=(32, 128, 128, 256),
    device="cuda"
):
    """
    Return a UNet 1D model.
    """
    model = UNet1DModel(
        sample_size=sample_size,  # the target batch size
        in_channels=in_channel,  # the number of input channels,
        out_channels=out_channel,  # the number of output channels
        layers_per_block=layers_per_block,  # how many ResNet layers to use per UNet block
        block_out_channels=(32, 128, 128, 256),  # More channels -> more parameters
        down_block_types=(
            "DownBlock1D",  # a regular ResNet downsampling block
            "DownBlock1D",
            "AttnDownBlock1D",  # a ResNet downsampling block with spatial self-attention
            "AttnDownBlock1D",
        ),
        up_block_types=(
            "AttnUpBlock1D",
            "AttnUpBlock1D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock1D",
            "UpBlock1D",  # a regular ResNet upsampling block
        ),
    )
    model = model.to(device)
    return model