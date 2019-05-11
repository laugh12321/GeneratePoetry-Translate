# Generate Poetry & Translate
use RNN and LSTM generate poetry by Keras, and translate poetry.

用Keras实现RNN+LSTM的模型自动编写古诗, 并将生成的古诗翻译成英文.

<b><font color='red'>Note: </font></b>  `main.py` 中已经省略的模型的构建过程，采用的是已经训练好的模型 `data/model_1000.h5`（迭代了1000次）, `data/poetry.txt` 是生成诗词所必须的语料文件.

## Usage

```
git clone https://github.com/laugh12321/GeneratePoetry-Translate.git

python main.py --word 风火山林 --num 7

风雨洞幽险驿雨
火维地上青山何
山白云雨中军司
林雨夕殿里雁又
Wind and rain Tunnel, dangerous post-rain, fire on the ground where the green hills.
Military commander in mountain white clouds and rain, wild goose in forest rain evening place.
```

<b><font color='red'>Note: </font></b> you must run it with GPU!!!