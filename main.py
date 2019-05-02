'''
@author: Laugh
@date: 2019/05/02 13:48
'''
import os
import argparse
import numpy as np
from keras import layers
from keras.optimizers import Adam
from data_clean import preprocess_file
from keras.models import Sequential, load_model
import warnings; warnings.filterwarnings("ignore")
from keras.callbacks import ModelCheckpoint, LambdaCallback

parser = argparse.ArgumentParser(description='Generate Poetry')
parser.add_argument('--word', type=str, default='月', required=False,
                    help='word')
parser.add_argument('--epochs', type=int, default=500, required=False,
                    help='epochs')
parser.add_argument("--learning_rate", type=int, default=0.001, required=False,
                    help='learning_rate')

args = parser.parse_args()
word = args.word
epochs = args.epochs
learning_rate = args.learning_rate

class Config(object):
    poetry_file = 'data/poetry.txt'
    weight_file = 'models/poetry_model.h5'

    max_len = 6 # 绝句长度包含标点
    epochs = epochs
    learning_rate = learning_rate
    
class GeneratePoetry(object):
    def __init__(self, config):
        self.model = None
        self.do_train = True
        self.loaded_model = False
        self.config = config
        
        # 文件预处理
        self.word2num, self.num2word, self.words, self.files_content = preprocess_file(self.config)
        self.poems = self.files_content.split(']')  # 诗的 list
        self.poems_num = len(self.poems)  # 诗的总数量
        
        # 如果模型文件存在则直接加载模型，否则开始训练
        if os.path.exists(self.config.weight_file):
            self.model = load_model(self.config.weight_file)
            optimizer = Adam(lr=self.config.learning_rate)
            self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        else:
            self.train()
        self.do_train = False
        self.loaded_model = True
    
    # 建立模型
    def build_model(self):
        self.model = Sequential()
        self.model.add(layers.LSTM(512, return_sequences=True, input_shape=(6, len(words))))
        self.model.add(layers.Dropout(0.6))
        self.model.add(layers.LSTM(256))
        self.model.add(layers.Dropout(0.6))
        self.model.add(layers.Dense(len(words), activation='softmax'))
        optimizer = Adam(lr=self.config.learning_rate)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    def word2numF(self, x): return self.word2num.get(x, len(self.words) - 1)
    
    def sample(self, preds):
        exp_preds = np.asarray(preds).astype('float64')
        preds = exp_preds / np.sum(exp_preds)
        pro = np.random.choice(range(len(preds)), 1, p=preds)
        return int(pro.squeeze())
        
    # 根据给出的首个文字，生成五言绝句
    def predict_first(self, char):
        if not self.loaded_model:
            return
        index = np.random.randint(0, self.poems_num)
        sentence = self.poems[index][1-self.config.max_len:] + char
        generate = str(char)
        generate += self._preds(sentence, length=23)
        return generate

    # 内部方法，输入 max_len 长度字符串，返回 length 长度的预测值字符串
    def _preds(self, sentence, length=23):
        sentence = sentence[:self.config.max_len]
        generate = ''
        for i in range(length):
            pred = self._pred(sentence)
            generate += pred
            sentence = sentence[1:]+pred
        return generate

    # 内部方法，根据一串输入，返回单个预测字符
    def _pred(self, sentence):
        sentence = sentence[-self.config.max_len:]
        x_pred = np.zeros((1, self.config.max_len, len(self.words)))
        for t, char in enumerate(sentence):
            x_pred[0, t, self.word2numF(char)] = 1.
        preds = self.model.predict(x_pred, verbose=0)[0]
        next_index = self.sample(preds)
        next_char = self.num2word[next_index]
        
        return next_char
        
     # 生成数据   
    def data_generator(self):
        i = 0
        while 1:
            x = self.files_content[i: i + self.config.max_len]
            y = self.files_content[i + self.config.max_len]

            if ']' in x or ']' in y:
                i += 1
                continue
                
            y_vec = np.zeros(
                shape=(1, len(self.words)),
                dtype=np.bool
            )
            y_vec[0, self.word2numF(y)] = 1.0

            x_vec = np.zeros(
                shape=(1, self.config.max_len, len(self.words)),
                dtype=np.bool
            )

            for t, char in enumerate(x):
                 x_vec[0, t, self.word2numF(char)] = 1.0
            yield x_vec, y_vec
            i += 1
    
    # 训练过程中，每 5 个 epoch 打印出当前的学习情况
    def generate_sample_result(self, logs):
        if (self.config.epochs + 1) % 5 == 0:
            # 随机获取某首诗词中的首字用于测试  
            char = self.poems[np.random.randint(len(self.poems))][0]
            generate = predict_first(char)
            print("+++++++++++++ 随机测试: {} +++++++++++++".format(char))
            print(generate)
            
    # 训练模型
    def train(self):
        if not self.model:
            self.build_model()
            
        self.model.fit_generator(
            generator=self.data_generator(),
            verbose=True,
            steps_per_epoch=32,
            epochs=self.config.epochs,
            callbacks=[
                ModelCheckpoint('poetry_model.h5'),
                LambdaCallback(on_epoch_end=self.generate_sample_result)]
        )
        
if __name__ == '__main__':
    config = Config
    model = GeneratePoetry(config)
    generate = model.predict_first(word)
    print(generate)