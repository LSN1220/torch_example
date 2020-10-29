import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

data = [("me gusta comer en la cafeteria".split(), "SPANISH"),
        ("Give it to me".split(), "ENGLISH"),
        ("No creo que sea una buena idea".split(), "SPANISH"),
        ("No it is not a good idea to get lost at sea".split(), "ENGLISH")]
test_data = [("Yo creo que si".split(), "SPANISH"),
             ("it is lost on me".split(), "ENGLISH")]
word_to_ix = {}

for sent, _ in data + test_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)
# {'me': 0, 'gusta': 1, 'comer': 2, 'en': 3,
# 'la': 4, 'cafeteria': 5, 'Give': 6, 'it': 7,
# 'to': 8, 'No': 9, 'creo': 10, 'que': 11,
# 'sea': 12, 'una': 13, 'buena': 14, 'idea': 15,
# 'is': 16, 'not': 17, 'a': 18, 'good': 19,
# 'get': 20, 'lost': 21, 'at': 22, 'Yo': 23,
# 'si': 24, 'on': 25}

VOCAB_SIZE = len(word_to_ix)
NUM_LABELS = 2


class BoWClassifier(nn.Module):
    def __init__(self, num_labels, vocab_size):
        super(BoWClassifier, self).__init__()
        self.linear = nn.Linear(vocab_size, num_labels)

    def forward(self, bow_vec):
        return F.log_softmax(self.linear(bow_vec), dim=1)


def make_bow_vector(sentence, word_to_ix):
    vec = torch.zeros(len(word_to_ix))
    for word in sentence:
        ix = word_to_ix[word]
        vec[ix] += 1
    return vec.view(1, -1)


def make_target(label, label_to_ix):
    return torch.LongTensor([label_to_ix[label]])


model = BoWClassifier(NUM_LABELS, VOCAB_SIZE)

for param in model.parameters():
    print(param)

with torch.no_grad():
    sample = data[0]
    bow_vector = make_bow_vector(sample[0], word_to_ix)
    log_probs = model(bow_vector)
    print(log_probs)

label_to_ix = {"SPANISH": 0, "ENGLISH": 1}

with torch.no_grad():
    for instance, label in test_data:
        bow_vector = make_bow_vector(instance, word_to_ix)
        log_probs = model(bow_vector)
        print(log_probs)
print(next(model.parameters())[:, word_to_ix['creo']])

loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    for instance, label in data:
        model.zero_grad()
        bow_vector = make_bow_vector(instance, word_to_ix)
        target = make_target(label, label_to_ix)
        log_probs = model(bow_vector)
        loss = loss_function(log_probs, target)
        loss.backward()
        optimizer.step()
with torch.no_grad():
    for instance, label in test_data:
        bow_vector = make_bow_vector(instance, word_to_ix)
        log_probs = model(bow_vector)
        print(log_probs)
print(next(model.parameters())[:, word_to_ix['creo']])