import numpy as np
import torch
import torch.nn as nn

# ...

class MixStyleResCausalModel(nn.Module):
    def __init__(self, model_name='resnet18', pretrained=False, num_classes=2, prob=0.2, ms_class=MixStyle, ms_layers=["layer1", "layer2"], mix="crosssample"):
        super().__init__()
        self.feature_extractor = ResNet(
            block=BasicBlock,
            layers=[2, 2, 2, 2],
            ms_class=ms_class,
            ms_layers=ms_layers,
            mix=mix
        )
        if pretrained:
            print('------- Loading pretrained weights ----------')
            init_pretrained_weights(self.feature_extractor, model_urls[model_name])

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = Classifier(in_channels=512, num_classes=num_classes)

        self.RandomReplace = RandomReplace(p=prob)
        self.dropout = RandomZero(p=prob)
        self.channel_shuffle = ChannelShuffleCustom(groups=16)

    def forward(self, input, labels=None, cf=['cs', 'dropout', 'replace'], norm=True):
        """
        Forward pass:
        - Returns logits
        - If training and labels provided, also returns counterfactual logits
        """
        feature = self.feature_extractor(input, labels)  # B x 512 x 7 x 7
        cls_feature = self.avgpool(feature).view(feature.size(0), -1)
        cls = self.classifier(cls_feature)

        if (not self.training) or cf is None or labels is None:
            return cls

        used_m = np.random.choice(cf)
        if used_m == 'dropout':
            cf_output = self.dropout(feature.clone())
        elif used_m == 'cs':
            cf_output = self.channel_shuffle(feature)
        elif used_m == 'replace':
            cf_output = self.RandomReplace(feature.clone())
        else:
            raise ValueError(f'Unknown counterfactual operation: {used_m}')

        cf_output = self.avgpool(cf_output)
        if norm:
            cf_output = norm_feature(cf_output, p=2, dim=1)
        cf_output = cf_output.view(cf_output.size(0), -1)
        cls_cf_out = self.classifier(cf_output)
        return cls, cls_cf_out
