# Adapted from Robert Guthrie

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from tqdm import tqdm

torch.manual_seed(1)


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# In solving the problem of adding new words, I realized that adding new words would
# cause index errors because of the way model() works with tensor sizes
# but adding the expected words during training would also defeat the purpose of
# recognizing "new" words
# Hence, this suggestion of dynamically resizing the model
# was from claude https://claude.ai/chat/13bd27b9-7692-4703-8aa3-78a35e855d66
def add_new_word(word):
    if word not in word_to_ix:
        word_to_ix[word] = len(word_to_ix)
        # Expand the embedding layer by one row
        old_weights = model.word_embeddings.weight.data
        new_weights = torch.cat([
            old_weights,
            torch.randn(1, old_weights.shape[1])  # random init for new word
        ])
        model.word_embeddings = nn.Embedding.from_pretrained(new_weights, freeze=False)

def test(words, targets):
    for word in words:
        add_new_word(word)
    inputs = prepare_sequence(words, word_to_ix)
    tag_scores = model(inputs)

    pos = torch.argmax(tag_scores, dim=1)

    # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
    # for word i. The predicted tag is the maximum scoring tag.
    # Here, we can see the predicted sequence below is 0 1 2 0 1
    # since 0 is index of the maximum value of row 1,
    # 1 is the index of maximum value of row 2, etc.
    # Which is DET NOUN VERB DET NOUN, the correct sequence!
    #print(pos)

    pred = [ix_to_tag[p.item()] for p in pos]

    # Header
    # used gemini to get quick formatting code
    print(f"\n{'Word':<10} | {'Predicted':<10} | {'Actual':<10}")
    print("-" * 35)

    # Rows
    for w, p, a in zip(words, pred, targets):
        print(f"{w:<10} | {p:<10} | {a:<10}")

# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

training_data = [
    # Tags are: DET - determiner; NN - noun; V - verb
    # For example, the word "The" is a determiner
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"]),
    ("The big dog ate the red apple".split(), ["DET", "ADJ", "NN", "V", "DET", "ADJ", "NN"]),
    ("Everybody read that awesome book".split(), ["NN", "V", "DET", "ADJ", "NN"])
]

word_to_ix = {}
# For each words-list (sentence) and tags-list in each tuple of training_data
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix: # word has not been assigned an index yet
            word_to_ix[word] = len(word_to_ix) # Assign each word with a unique index
            
    
#print(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2, "ADJ": 3}  # Assign each tag with a unique index
ix_to_tag = {v: k for k, v in tag_to_ix.items()}
# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6


if __name__ == "__main__":

    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    # See what the scores are before training
    # Note that element i,j of the output is the score for tag j for word i.
    # Here we don't need to train, so the code is wrapped in torch.no_grad()

    #with torch.no_grad():
        #example = 2
        #test(training_data[example][0], training_data[example][1])


    for epoch in tqdm(range(300)):  # again, normally you would NOT do 300 epochs, it is toy data
        for sentence, tags in training_data:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix)

            # Step 3. Run our forward pass.
            tag_scores = model(sentence_in)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()

    # See what the scores are after training
    with torch.no_grad():
        example = 0
        test(training_data[example][0], training_data[example][1])
        
        
