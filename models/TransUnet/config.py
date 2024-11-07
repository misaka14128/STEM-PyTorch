import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.patches_size = 16
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.classifier = 'seg'
    config.representation_size = None
    config.resnet_pretrained_path = None
    config.pretrained_path = './models/TransUnet/pre-trained/imagenet21k_ViT-B_16.npz'

    config.decoder_channels = (256, 128, 64, 16)
    # config.n_classes = 4
    # config.activation = 'softmax'
    return config
