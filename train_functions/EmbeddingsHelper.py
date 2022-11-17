from torchtext.data import get_tokenizer
from torchtext.vocab import GloVe

max_words = 25
embed_len = 50

tokenizer = get_tokenizer("basic_english")
global_vectors = GloVe(name = "6B", dim = embed_len)

# https://coderzcolumn.com/tutorials/artificial-intelligence/how-to-use-glove-embeddings-with-pytorch

def vectorize_batch(X):
    # separate the question into individual tokens (words)
    X = [tokenizer(x) for x in X]
    # make all sentences have the same number of tokens, pad with empty string or cut as needed
    X = [tokens+[""] * (max_words-len(tokens))  if len(tokens)<max_words else tokens[:max_words] for tokens in X]
    # note that this shape will require batch_first = true for the lstm, so we will transpose it at the end
    X_tensor = torch.zeros(len(X), max_words, embed_len)
    for i, tokens in enumerate(X):
        X_tensor[i] = global_vectors.get_vecs_by_tokens(tokens)
    # with the transpose, we can have batch_first = false for the lstm
    return torch.transpose(X_tensor, 0, 1)
