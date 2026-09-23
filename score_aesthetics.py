"""Score already-generated images with the LAION aesthetic predictor used in the paper.

predictor: improved-aesthetic-predictor / sac+logos+ava1-l14-linearMSE.pth
           (MLP head on L2-normalised CLIP ViT-L/14 image embeddings)

  python3 score_aesthetics.py <dir1> [<dir2> ...]
"""
import glob
import os
import sys

import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

ASSETS = '/root/bayes-tmp/afa-assets'
CKPT = f'{ASSETS}/models/aesthetic/sac+logos+ava1-l14-linearMSE.pth'
CLIP = f'{ASSETS}/models/clip-vit-large-patch14'


class MLP(nn.Module):
    def __init__(self, input_size=768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024), nn.Dropout(0.2),
            nn.Linear(1024, 128), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.Dropout(0.1),
            nn.Linear(64, 16), nn.Linear(16, 1))

    def forward(self, x):
        return self.layers(x)


def main():
    head = MLP().eval()
    head.load_state_dict(torch.load(CKPT, map_location='cpu'))
    clip = CLIPModel.from_pretrained(CLIP).eval()
    proc = CLIPProcessor.from_pretrained(CLIP)
    print('model loaded\n')
    print(f'{"variant":<28} {"n":>3} {"mean":>7} {"min":>7} {"max":>7}')
    rows = []
    for d in sys.argv[1:]:
        files = sorted(glob.glob(os.path.join(d, '*.png')))
        if not files:
            continue
        scores = []
        for f in files:
            im = Image.open(f).convert('RGB')
            with torch.no_grad():
                emb = clip.get_image_features(**proc(images=im, return_tensors='pt'))
                emb = emb / emb.norm(dim=-1, keepdim=True)
                scores.append(head(emb.float()).item())
        name = '/'.join(d.rstrip('/').split('/')[-2:])
        rows.append((name, sum(scores) / len(scores)))
        print(f'{name:<28} {len(scores):>3} {sum(scores) / len(scores):>7.3f} '
              f'{min(scores):>7.3f} {max(scores):>7.3f}', flush=True)


if __name__ == '__main__':
    main()
