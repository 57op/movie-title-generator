import sys
import math

import torch
import sentencepiece as spm

from pathlib import Path
from model import Model

device = torch.device('cpu') # 'cuda' if torch.cuda.is_available() else 'cpu')
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
model.load_state_dict(torch.load(next(Path('models').glob('*.torch'))))
model.eval()
print(model.predict(float(sys.argv[1]), sp, special_chars, temperature=0.8))
