import json

class DataLoader:
  def __init__(self, dataset):
    with open(dataset) as fh:
      self.movies = json.load(fh)
  
  def gen_batches(self, bs, sp, special_chars):
    for i in range(0, len(self.movies), bs):
      targets = [] # <s> title <sep> plot </s> <pad>
      max_tgt = 0
      
      for movie in self.movies[i : i + bs]:
        tgt = [int(movie['rating'] / 10 * 101)] + sp.EncodeAsIds(movie['title']) + [special_chars['</s>']]
        max_tgt = max(max_tgt, len(tgt))
        targets.append(tgt)
        
      # pad to the right
      for target in targets:
        if len(target) < max_tgt:
          target.extend([special_chars['<pad>']] * (max_tgt - len(target)))
          
      yield targets
  
  def size(self, sp):
    max_encoded_title = 0
    max_encoded_plot = 0
    
    for movie in self.movies:
      max_encoded_title = max(max_encoded_title, len(sp.EncodeAsIds(movie['title'])))
      max_encoded_plot = max(max_encoded_plot, len(sp.EncodeAsIds(movie['plot'])))
      
    return max_encoded_title, max_encoded_plot
    
if __name__ == '__main__':
  import sentencepiece as spm
  sp = spm.SentencePieceProcessor()
  sp.Load('data/bpe.model')
  
  dl = DataLoader('data/dataset.json')

  for b in dl.gen_batches(32, sp, {'</s>': -1, '<pad>': -2}):
    print(b)
    exit()
