import torch.nn as nn
import torch.nn.functional as F

class DETR(nn.Module):
    def __init__(self, backbone, positional_embedding, transformer, num_classes, num_predict):
        super(DETR, self).__init__()
        self.backbone = backbone
        self.positional_embedding = positional_embedding
        self.transformer = transformer
        hidden_dim = transformer.d_model
        backbone_out_channels = backbone.out_channels
        self.num_classes = num_classes
        self.query_embed = nn.Embedding(num_predict, hidden_dim)  # query에 대한 positional embedding
        self.input_proj = nn.Conv2d(backbone_out_channels, hidden_dim, kernel_size=1)

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, x):
        feature = self.backbone(x)
        pos_embed = self.positional_embedding(x)

        hs = self.transformer(self.input_proj(feature), self.query_embed.weight, pos_embed)[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

        return out


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x