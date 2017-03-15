from torchtext import data
from torchtext import datasets

TEXT = data.Field()
LABEL = data.Field(sequential=False, use_vocab=False, preprocessing=(lambda s: int(s)))
train, test = datasets.IMDB.splits(TEXT, LABEL, root='../data/')
train_iter, test_iter = data.BucketIterator.splits( (train, test), batch_size=32, sort_key=lambda x: len(x.text), device=-1)

print(train.fields)
print(len(train))
print(len(test))
print(vars(train[0]))
print(vars(test[0]))

TEXT.build_vocab(train)
#LABEL.build_vocab(train)

print(len(TEXT.vocab))

train_iter, test_iter = data.BucketIterator.splits(
    (train, test), batch_size=32, sort_key=lambda x: len(x.text), device=-1)

print(TEXT.vocab.freqs.most_common(100))


batch = next(iter(train_iter))
print(batch.text)
print(batch.label)
