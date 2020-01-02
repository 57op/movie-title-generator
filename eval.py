import sys
import torch
import sentencepiece as spm

from pathlib import Path
from model import Model

# inference on cpu
device = torch.device('cpu')

sp = spm.SentencePieceProcessor()
sp.Load('models/bpe.model')
special_chars = {
  '<pad>': len(sp),
  '</s>': len(sp) + 1
}

d_model = 128
dim_feedforward = 512
nhead = 4

model = Model(
  special_chars['<pad>'],
  vocabulary_size=len(sp) + len(special_chars),
  d_model=d_model,
  nhead=nhead,
  dim_feedforward=dim_feedforward).to(device)
model.load_state_dict(
  torch.load(
    'models/colab_epoch1120.torch',
    map_location=device))
model.eval()

if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('rating', type=float)
  parser.add_argument('--samples', type=int, default=1)

  args = parser.parse_args()

  for i in range(args.samples):
    print(model.predict(args.rating, sp, special_chars, temperature=0.8))
