class Vocabulary:
    def __init__(self):
        self.char_to_idx = {
            '<pad>': 0,
            '<sos>': 1,
            '<eos>': 2,
            ' ': 3,
            "'": 4,
        }
        
        # Add a-z (5 to 30)
        for i, c in enumerate('abcdefghijklmnopqrstuvwxyz'):
            self.char_to_idx[c] = i + 5
        
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)  # 31
    
    def encode(self, text):
        text = text.lower()
        indices = [1]  # <sos>
        for c in text:
            if c in self.char_to_idx:
                indices.append(self.char_to_idx[c])
        indices.append(2)  # <eos>
        return indices
    
    def decode(self, indices):
        chars = []
        for i in indices:
            if i > 2:  # skip pad, sos, eos
                chars.append(self.idx_to_char[i])
        return ''.join(chars)