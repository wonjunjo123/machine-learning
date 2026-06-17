# Adapted from Robert Guthrie

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

from tqdm import tqdm

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
# We will use Shakespeare Sonnet 2
'''test_sentence = """when forty winters shall besiege thy brow
and dig deep trenches in thy beautys field
thy youths proud livery so gazed on now
will be a totterd weed of small worth held
then being asked where all thy beauty lies
where all the treasure of thy lusty days
to say within thine own deep sunken eyes
were an all eating shame and thriftless praise
how much more praise deserved thy beautys use
if thou couldst answer this fair child of mine
shall sum my count and make my old excuse
proving his beauty by succession thine
this were to be new made when thou art old
and see thy blood warm when thou feelst it cold""".split()'''

adjectives = ("adorable", "clueless", "dirty", "odd", "stupid")
nouns = ("puppy", "car", "rabbit", "girl", "monkey")
verbs = ("runs", "hits", "jumps", "drives", "barfs") 
adverbs = ("crazily", "dutifully", "foolishly", "merrily", "occasionally")

test_sentence = []

for adj in adjectives:
    for noun in nouns:
        for verb in verbs:
            for adverb in adverbs:
                sentence = adj + " " + noun + " " + verb + " " + adverb
                test_sentence.extend(sentence.split())
                



# we should tokenize the input, but we will ignore that for now
# build a list of tuples.
# Each tuple is ([ word_i-CONTEXT_SIZE, ..., word_i-1 ], target word)
ngrams = [
    (
        [test_sentence[i - j - 1] for j in range(CONTEXT_SIZE)],
        test_sentence[i]
    )
    for i in range(CONTEXT_SIZE, len(test_sentence))
]
# Print the first 3, just so you can see what they look like.
#print(ngrams[:3])

vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}


class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

if __name__ == "__main__":
    losses = []
    loss_function = nn.NLLLoss()
    model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    EPOCHS = 10

    for epoch in tqdm(range(EPOCHS)):
        total_loss = 0
        for context, target in ngrams:

            # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
            # into integer indices and wrap them in tensors)
            context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)

            # Step 2. Recall that torch *accumulates* gradients. Before passing in a
            # new instance, you need to zero out the gradients from the old
            # instance
            model.zero_grad()

            # Step 3. Run the forward pass, getting log probabilities over next
            # words
            log_probs = model(context_idxs)

            # Step 4. Compute your loss function. (Again, Torch wants the target
            # word wrapped in a tensor)
            loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))

            # Step 5. Do the backward pass and update the gradient
            loss.backward()
            optimizer.step()

            # Get the Python number from a 1-element Tensor by calling tensor.item()
            total_loss += loss.item()
        losses.append(total_loss)
    #print(losses)  # The loss decreased every iteration over the training data!

    data = [] # will keep track of embeddings for hierarchical cluster analysis
    # Formatting code from https://chatgpt.com/c/69e1e7af-4640-83ea-bb17-909f30035a93
    for word in sorted(vocab):
        x = (model.embeddings.weight[word_to_ix[word]]).tolist()
        data.append(x)
        print(f"{word} [{ ' '.join(f'{v:+.3f}' for v in x) }]")

    data = np.array(data)

    # used https://claude.ai/chat/007f43f3-34a7-48af-832b-34e5243f0868 to help with syntax details
    linkage_data = linkage(data, method='ward', metric='euclidean')
    plt.figure(figsize=(10,15))
    dendrogram(linkage_data, labels=sorted(vocab), orientation='left')
    plt.show()
    
    












