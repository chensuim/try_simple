import json

import re
from nltk.corpus import stopwords
from bert_serving.client import BertClient

bc_client = BertClient(show_server_config=False)

bracket_p = re.compile(r"<.*?>|\(.*?\)")
chi_p = re.compile(r"[\u4E00-\u9FFF]")
not_char_p = re.compile(r"[^\s\w]")
choices_p = re.compile(r"[ABCDE]:")
_p = re.compile(r"_*")
stop_words = set(stopwords.words('english'))


def remove_stop_words(s):
    words = s.split()
    ans = list()
    for word in words:
        if word not in stop_words:
            ans.append(word)
    return ' '.join(ans)


def clean_keep_stop(s):
    s = re.sub(_p, '', s)
    s = re.sub(choices_p, ' ', s)
    s = re.sub(bracket_p, ' ', s)
    s = re.sub(not_char_p, '', s)
    s = s.replace("u2014", "")
    return ' '.join(map(lambda x: x.lower(), s.split()))


def clean(s):
    s = re.sub(_p, '', s)
    s = re.sub(choices_p, ' ', s)
    s = re.sub(bracket_p, ' ', s)
    s = re.sub(not_char_p, '', s)
    s = s.replace("u2014", "")
    return remove_stop_words(' '.join(map(lambda x: x.lower(), s.split())))


def get_embedding(text):
    features = bc_client.encode([text])
    return features[0]


def main(fn, on):
    with open(fn, 'r') as f, open(on, 'w') as w:
        all_labels = set()
        clean_data = list()
        i = 0
        for line in f.readlines():
            if i & 0xff == 0:
                print(i)
            i += 1
            line = json.loads(line)
            text = line['text']
            labels = line['labels']
            text = clean(text)
            if text:
                vec = get_embedding(text)
                clean_data.append([vec, labels])
                all_labels.update(labels)
        num_classes = len(all_labels)
        all_labels = list(all_labels)
        idx = dict(zip(all_labels, range(num_classes)))
        for data in clean_data:
            data[1] = list(map(lambda x: '__label__' + str(idx[x]), data[1]))
            w.write(' '.join([str(x) for x in data[0]]))
            w.write(' ')
            w.write(' '.join(data[1]))
            w.write('\n')
        print('num_classes is %s' % num_classes)
        return 0


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv[1], sys.argv[2]))


