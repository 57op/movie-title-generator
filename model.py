import math
import torch
import torch.nn as nn

# from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
  def __init__(self, d_model, dropout=0.1, max_len=5000):
    super().__init__()
    self.dropout = nn.Dropout(p=dropout)

    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0).transpose(0, 1)
    self.register_buffer('pe', pe)

  def forward(self, x):
    x = x + self.pe[:x.size(0), :]
    return self.dropout(x)

class Model(nn.Module):
  def __init__(self, padding_idx, vocabulary_size=0, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
    super().__init__()
    self.vocabulary_size = vocabulary_size
    self.d_model = d_model
    
    self.rating_embedding = nn.Embedding(101, d_model) # 0.0 - 10.0 [step 0.1]
    self.token_embedding = nn.Embedding(self.vocabulary_size, d_model, padding_idx=padding_idx)
    self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)
    
    encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout)
    # forward: src, mask=None, src_key_padding_mask=None
    self.encoder = nn.TransformerEncoder(encoder_layer, 6)
    self.fc = nn.Linear(d_model, self.vocabulary_size, bias=False)
    
  # input is [seq_length, batch_size]
  def forward(self, input, key_padding_mask=None):
    src = self.token_embedding(input)
    # src is [seq_length, batch_size, d_model]
    
    src[0, :, :] = self.rating_embedding(input[0, :])
    # the first item in the sequent isn't a token, but rating
    
    # add position informations
    src = self.positional_encoding(src)
    
    mask = nn.Transformer.generate_square_subsequent_mask(None, src.size(0)).to(input.device)
    return self.fc(self.encoder(src, mask=mask, src_key_padding_mask=key_padding_mask))


  def predict(self, rating, sp, special_chars, temperature=1.0, device=None, maxiter=256):
    assert isinstance(rating, float) and 0.0 <= rating <= 10.0, 'rating must be a float within 0-10 range'
    
    # sentencepiece automatically includes <unk>, <s>, </s>
    # if i had known it before-hand (my bad, sry)
    # i would have used them instead of adding <pad> and </s> manually to vocabulary
    black_list = [sp.unk_id(), sp.bos_id(), sp.eos_id()]
    tgt = [int(rating / 10 * 101)]

    while tgt[-1] != special_chars['</s>'] and len(tgt) < maxiter:
      tgt_e = torch.tensor(tgt, device=device).unsqueeze(1)
      output = self(tgt_e)
      
      # https://towardsdatascience.com/how-to-sample-from-language-models-682bceb97277
      # Lower temperatures make the model increasingly confident in its top choices, while temperatures greater than 1 decrease confidence
      output = output[-1, 0, :] / temperature # apply temperature
    
      # make it impossible to draw blacklisted tokens
      for token_id in black_list:
        output[token_id] = -math.inf
    
      output = torch.softmax(output, dim=-1)
      tgt.append(torch.multinomial(output, 1).item())
      
    return sp.DecodeIds(tgt[1:-1])
