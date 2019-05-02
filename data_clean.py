'''
@author: Laugh
@date: 2019/05/02 13:48
'''

def preprocess_file(Config):
    # 语料文本内容
    files_content = ''
    with open(Config.poetry_file, 'r',encoding='UTF-8') as f:
        for line in f:
            x = line.strip() + "]"
            x = x.split(":")[1]
            if len(x) <= 5 :
                continue
            if x[5] == '，':
                files_content += x
            

    words = sorted(list(files_content))
    counted_words = {}
    for word in words:
        if word in counted_words:
            counted_words[word] += 1
        else:
            counted_words[word] = 1

    # 去掉低频的字
    erase = []
    for key in counted_words:
        if counted_words[key] <= 2:
            erase.append(key)
    for key in erase:
        del counted_words[key]
    wordPairs = sorted(counted_words.items(), key=lambda x: -x[1])

    words, _ = zip(*wordPairs)
    words += (" ",)
    # word到id的映射
    word2num = dict((c, i) for i, c in enumerate(words))
    num2word = dict((i, c) for i, c in enumerate(words))
    return word2num, num2word, words, files_content