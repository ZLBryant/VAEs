# VAEs

python+pytorch
对VAE和CVAE进行实现，后续可能会实现其他的VAE

#### Dataset
MNIST

#### 运行
VAE，进行三种形式的生成图片：随机生成图片、根据输入图片生成相似图片、对两个数字进行插值实现渐变（按默认参数，具体设置详见代码）：
``` 
训练VAE：python test.py --VAE --train
用VAE生成图片：python test.py --VAE --test
```
CVAE，进行两种形式的生成图片：根据输入数字生成相似图片、对两个数字进行插值实现渐变（按默认参数，具体设置详见代码）：
``` text
训练VAE：python test.py --CVAE --train
用VAE生成图片：python test.py --CVAE --test
```

#### 结果
生成结果详见output目录下的内容，其实个人感觉代码中CVAE进行插值的做法有问题，是直接对两个数字的独热编码进行插值，但这两个是离散的编码，能进行插值吗，VAE就是针对AE中两个独立的编码之间的渐变问题，这里的方法是有悖于VAE中的想法的，有更好的想法还请指教。
