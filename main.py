import os

import sentencepiece as spm
import torch
import torch.nn as nn

from data_loader import DataLoader
from model import Model
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sp = spm.SentencePieceProcessor()
sp.Load('data/bpe.model')
special_chars = {
  '<pad>': len(sp),
  '</s>': len(sp) + 1
}

dl = DataLoader('data/dataset.json')

stopped_epoch = 945
epochs = 5000
batch_size = 128

d_model = 128
dim_feedforward = 512
nhead = 4


factor = 2
warmup = 4000
steps = stopped_epoch * 1138 # 0

model = Model(
  special_chars['<pad>'],
  vocabulary_size=len(sp) + len(special_chars),
  d_model=d_model,
  nhead=nhead,
  dim_feedforward=dim_feedforward).to(device)
model.load_state_dict(torch.load(next(Path('models').glob('*.torch'))))

optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
criterion = nn.CrossEntropyLoss(ignore_index=special_chars['<pad>'])

previous_model_name = None
log_msg = ''


for e in range(stopped_epoch, epochs):
  previous_loss = 0

  for b, input in enumerate(dl.gen_batches(batch_size, sp, special_chars)):
    source = torch.tensor(input, device=device)
    key_padding_mask = source == special_chars['<pad>']
    source = source.transpose(0, 1)

    output = model(source[:-1, :], key_padding_mask=key_padding_mask[:, :-1]) # tgt_seq_len [d] × batch_size [N] × vocabulary_size [C]
    loss = criterion(
      # N × C × d
      output.transpose(0, 1).transpose(1, 2),
      # N × d
      source[1:, :].transpose(0, 1))
    
    steps += 1
    lrate = factor * ((d_model ** (-0.5)) * min(steps ** (-0.5), steps * (warmup ** (-1.5))))

    for p in optimizer.param_groups:
      p['lr'] = lrate

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    avg_loss = (previous_loss * b + loss.item()) / (b + 1)
    previous_loss = avg_loss
    
    print('\r{0}'.format(' ' * len(log_msg)), end='')
    log_msg = '\r[Epoch {e}/{epochs}][Batch {b}][LR {lrate:e}] Loss = {l}'\
      .format(e=e + 1, epochs=epochs, b=b, lrate=lrate, l=avg_loss)
    print(log_msg, end='')
    
    del source, key_padding_mask, output, loss
    # torch.cuda.empty_cache()
      
  if e % 5 == 4:
    if previous_model_name:
      os.unlink(previous_model_name)

    model_name = 'models/epoch{e:04d}.torch'.format(e=e + 1)
    torch.save(model.state_dict(), model_name)
    previous_model_name = model_name
