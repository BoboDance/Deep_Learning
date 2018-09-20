import zipfile
import collections
import nltk

import tensorflow as tf

# sources:
# - https://gist.github.com/yxtay/a94d971955d901c4690129580a4eafb9


def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        text = tf.compat.as_str(f.read(f.namelist()[0])).split()

    print('Text has length of {}'.format(len(text)))
    print(nltk.Text(text))
    return nltk.Text(text)


def build_dataset(text, vocab_size):
    # replace words occurring less frequent than vocab size with UNK
    fdist = nltk.FreqDist(text)
    vocab = fdist.most_common(vocab_size)

    text = [word if word in vocab else 'UNK' for word in text.words()]

    # text.most
    # dictionary = dict()
    # for word, _ in count:
    #     dictionary[word] = len(dictionary)
    # data = list()
    # unk_count = 0
    # for word in words:
    #     if word in dictionary:
    #         index = dictionary[word]
    #     else:
    #         index = 0  # dictionary['UNK']
    #         unk_count += 1
    #     data.append(index)
    # count[0][1] = unk_count
    # reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    # return data, count, dictionary, reverse_dictionary


file = "text8.zip"
text = read_data(file)

# define new vocab size
vocabulary_size = 50000

data, count, dictionary, reverse_dictionary = build_dataset(words)
del words  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0
