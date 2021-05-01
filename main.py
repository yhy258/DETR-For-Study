import torch
import torch.nn as nn
from transformer import Transformer
import torchvision
from positional_embedding import PositionalEmbedding
from detr import DETR
import torch.nn.functional as F
from matcher import HungarianMatcher, box_cxcywh_to_xyxy, generalized_box_iou
import config
from tqdm.notebook import tqdm

"""
    Warning -> trainloader 설정 해줘야함.
    dataset -> images, targets : {'labels' : ~ , 'boxes': ~ }
    이 때 box는 center coordinates, width, height [0,1]
"""

class ResNetFeatures(nn.Module):
    def __init__(self, model):
        super(ResNetFeatures, self).__init__()

        self.seq1 = nn.Sequential(model.conv1,
                                  model.bn1,
                                  model.relu,
                                  model.maxpool,
                                  model.layer1,
                                  model.layer2,
                                  model.layer3,
                                  model.layer4)

        self.out_channels = 2048

    def forward(self, x):
        x = self.seq1(x)

        return x

transformer = Transformer(d_model=config.ModelConfig.d_model, nhead=config.ModelConfig.n_head, num_encoder_layers=config.ModelConfig.num_encoder_layers,
                 num_decoder_layers=config.ModelConfig.num_decoder_layers, dim_feedforward=config.ModelConfig.dim_feedforward, dropout=config.ModelConfig.dropout)


# Backbone 선언
model = torchvision.models.resnet50(pretrained=True)
backbone = ResNetFeatures(model)

# BN Freeze Function
def BN_Freeze(model):
    for name, module in model.named_modules():
        if name == 'bn1' :
            module.requires_grad_ = False
        if name == 'layer1' or name == 'layer2' or name == 'layer3' or name == 'layer4':
            for child_name, child_module in module.named_modules():
                if (len(child_name) > 1) and child_name[2:4] == 'bn':
                    child_module.requires_grad_ = False
    return model

backbone = BN_Freeze(backbone)

# Positional Embedding 선언

positional_embedding = PositionalEmbedding(config.ModelConfig.d_model)

# DETR 선언

detr = DETR(backbone=backbone, positional_embedding=positional_embedding, transformer=transformer, num_classes=config.TrainConfig.classes_num ,num_predict=config.TrainConfig.predict_num)
param_dicts = [
        {"params": [p for n, p in detr.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in detr.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": 1e-3
        },
    ]
optimizer = torch.optim.AdamW(detr.parameters(), lr=1e-3)


USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')



matcher = HungarianMatcher()


def _get_src_permutation_idx(indices):  # matcher에 맞게 index 조정 함수.
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx

# Train loader, epochs, box_nums, num_classes 설정 안되어있음.
for epoch in range(config.TrainConfig.epochs):
    print("Epoch {}/{}".format(epoch+1, config.TrainConfig.epochs))
    for images, targets in tqdm(train_loader):  # targets -> image_id : , annotations :  -> 'labels' :, 'boxes'로 줘야할 듯.
        images = images.to(DEVICE)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        outputs = detr(images)  # out = {'pred_logits':  'pred_boxes':}

        # Loss
        indices = matcher(outputs, targets)
        idx = _get_src_permutation_idx(indices)

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])  # targets의 인덱스를 바꿔줌.
        target_classes = torch.full(outputs['pred_logits'].shape[:2], num_classes=100,
                                    dtype=torch.int64, device=DEVICE)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(outputs['pred_logits'].transpose(1, 2), target_classes)  # class loss

        idx = _get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        num_boxes = config.TrainConfig.predict_num

        loss_bbox = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)))
        loss_giou = loss_giou.sum() / num_boxes

        losses = loss_ce + loss_bbox + loss_giou

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    print("This Iter Loss : {}".format(losses.item()))