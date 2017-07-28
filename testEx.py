import collections
vocabulary_size = 50000

def build_dataset(words): #words in corpus , list
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  # 因为使用了most_common,所以count中的word，是按照word在文本中出现的次数从大到小排列的
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)   # assign id to word
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK'] UNK:unknown
      unk_count += 1
    data.append(index) #translate word to id
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reverse_dictionary #data:ids count:list of [word,num] dictionary:word->id ,

if __name__ == '__main__':
    data, count, dictionary, reverse_dictionary = build_dataset("student Teacher")
    print(data)
    print(count)
    print(dictionary)
    print(reverse_dictionary)