'''
update: 2019-05-10
@Author: Laugh
'''
import os
import argparse
import numpy as np
from translate import translation
from keras.models import load_model
import warnings; warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Generate Poetry')
parser.add_argument('--word', type=str, default='风火山林', required=False,
                    help='word')
parser.add_argument('--num', type=int, default=7, required=False,
                    help='num')
                    
args = parser.parse_args()
word = args.word
num = args.num

class DataGenerator:
    def __init__(self, path, single_word=False, remove_words=[]):
        '''
        Poetry Data Generator.
        
        # Arguments
            path: the data file path
            single_word: used single word or used vocabulary by jieba. False is default.
            remove_words: list of remove words. ex: '。', '，'
        '''
        with open(path, 'r',encoding='UTF-8') as f:
            self.text=f.read()
        
        if remove_words:
            print("remove the words:", remove_words)
            self.text = [t for t in self.text if t not in remove_words]
        
        if single_word:
            self.vocab = sorted(set(self.text))
            self.vocab_to_int = {c: i for i, c in enumerate(self.vocab)}
            self.int_to_vocab = dict(enumerate(self.vocab))
            self.encoded = np.array([self.vocab_to_int[c] for c in self.text], dtype=np.int32)
        else:
            import jieba
            seg_list = jieba.lcut(self.text, cut_all=False)
            self.vocab = sorted(set(seg_list))
            self.vocab_to_int = {c: i for i, c in enumerate(self.vocab)}
            self.int_to_vocab = dict(enumerate(self.vocab))
            self.encoded = np.array([self.vocab_to_int[c] for c in seg_list if c not in remove_word], dtype=np.int32)
    

    def window_transform_text(self, window_size, step_size):
        '''
        To sperate data to input / output by window size and step size.
        # Arguments
            window_size: input sentence size, word numbers each stentence.
            step_size: shift size for next sentence along all text.
        # return
            inputs: input data list. It is a two-dimensional list (, window_size).
            outputs: output data list. It is a two-dimensional list (, 1).
        '''
        total_len = len(self.text)
        data_size = int(np.ceil((total_len - window_size) / step_size))
        x_start, x_end = 0, (total_len - window_size)
        y_start, y_end = window_size, total_len
        print("x_start:{} x_end:{} y_start:{} y_end:{} total_len:{} data_size:{}".format(x_start, x_end, y_start, y_end, total_len, data_size))
        # containers for input/output pairs
        inputs = []
        outputs = []
        for i in range(x_start, x_end, step_size):
            inputs.append(self.text[i : (i + window_size)])
        for i in range(y_start, y_end, step_size):
            outputs.append(self.text[i])
        return inputs,outputs
    
    def predict_next_chars(self, model, input_chars, num_to_predict, window_size, one_hot=False):
        '''
        To predict next char.
        # Arguments
            model: Instance of `Model`
            input_chars: input chars list
            num_to_predict: number of chars to predict
            window_size: input sentence size, word numbers each stentence.
            one_hot: used one hot vector or not. False is default.
        # return
            predicted_chars: input_chars + [predict chars]
        '''
        inputs = input_chars[:]
        # create output
        predicted_chars = ''.join(inputs)
        # number of vocabulary
        num_vocab = len(self.vocab)
        for i in range(num_to_predict):
            if one_hot:
                # convert this round's predicted characters to numerical input
                x_test = np.zeros((1, window_size, num_vocab))
                for t, char in enumerate(inputs):
                    x_test[0, t, self.vocab_to_int[char]] = 1.
            else:
                x_test = np.zeros((1, window_size, 1))
                for t, char in enumerate(inputs):
                    x_test[0, t, 0] = self.vocab_to_int[char]

            # make this round's prediction
            test_predict = model.predict(x_test,verbose = 0)[0]
            r = self.pick_top_n(test_predict, num_vocab)
            d = self.int_to_vocab[r] 

            # update predicted_chars and input
            predicted_chars += d
            inputs += d
            inputs = inputs[1:]
        return predicted_chars
      
    def pick_top_n(self, preds, vocab_size, top_n=5):
        '''
        To pick up one of top n.
        '''
        p = np.squeeze(preds)
        p[np.argsort(p)[:-top_n]] = 0
        p = p / np.sum(p)
        c = np.random.choice(vocab_size, 1, p=p)[0]
        return c

def gen_poetry(model, prefix_ary, poetry_type=5):
    poetry_sentence_len = poetry_type - 1
    poetry = []
    for char in prefix_ary:
        predict_input = gen.predict_next_chars(model, [char], poetry_sentence_len, window_size, one_hot=one_hot)
        poetry.append(predict_input)
    return poetry

if __name__ == '__main__':
    os.chdir('F:/Jupyter/Laugh/demo/data')
    window_size = 1
    step_size = 1
    one_hot = True
    hidden_units = 512
    remove_words = ['\n', '。', '，', '！', ' ', '?']

    gen = DataGenerator('poetry.txt', single_word=True, remove_words=remove_words)
    features = (len(gen.vocab) if one_hot else 1) # 1 for vocab_for_int, len(gen.vocab) for one hot encoding
    output_dim = len(gen.vocab)
    predict_modell = load_model('model_1000.h5')

    prefix_ary = [i for i in word]
    poetry = gen_poetry(predict_modell, prefix_ary, num)
    print(*poetry, sep='\n')
    sentens = poetry

    # translate
    sen1 = sentens[0] + ',' + sentens[1] + '.'
    translation(sen1)
    sen2 = sentens[2] + ',' + sentens[3] + '.'
    translation(sen2)