#!/usr/bin/env python
"""Chainer example: train a multi-layer perceptron on MNIST

This is a minimal example to write a feed-forward net.
It requires scikit-learn to load MNIST dataset.

"""
import argparse

import numpy as np
import matplotlib.pyplot as plt
import six

import chainer
from chainer import Link, Chain, ChainList
from chainer import computational_graph as c
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import optimizers

import easydict

import data

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
# 使用するGPUの数
args = easydict.EasyDict({
    # GPU数
    "gpu":0,
    # 確率的勾配降下法で学習させる際の１回分のバッチサイズ
    "batchsize":100,
    # 学習の繰り返し回数
    "n_epoch":20,
    # 中間層の数(高精度：1000,低精度：100)
    "n_units":100})

if args.gpu > 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu > 0 else np

# Prepare dataset
# パラメータデータセット
print('load MNIST dataset')
mnist = data.load_mnist_data()
mnist['data'] = mnist['data'].astype(np.float32)
mnist['data'] /= 255    # 0-1のデータに変換
mnist['target'] = mnist['target'].astype(np.int32)

# 手書き数字データを描画する関数
def draw_digit(data):
    size = 28
    plt.figure(figsize=(2.5, 3))

    X, Y = np.meshgrid(range(size),range(size))
    Z = data.reshape(size,size)     # convert from vector to 28x28 matrix
    Z = Z[::-1,:]                   # flip vertical
    plt.xlim(0,27)
    plt.ylim(0,27)
    plt.pcolor(X, Y, Z)
    plt.gray()
    plt.tick_params(labelbottom="off")
    plt.tick_params(labelleft="off")

# サンプル表示
# draw_digit(mnist['data'][12345])
# plt.show()
# plt.close()

# 学習用データを N個、検証用データを残りの個数と設定
N = 60000
x_train, x_test = np.split(mnist['data'],   [N])
y_train, y_test = np.split(mnist['target'], [N])
N_test = y_test.size

# Prepare multi-layer perceptron model
# 多層パーセプトロンモデルの設定
# 入力 784次元、出力 10次元
model = chainer.Chain(l1=L.Linear(784, args.n_units),           # 入力層：28×28の画像
                      l2=L.Linear(args.n_units, args.n_units),  # 隠れ層
                      l3=L.Linear(args.n_units, 10))            # 出力層：0〜9の判定
if args.gpu > 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

# Neural net architecture
# ニューラルネットの構造
def forward(x_data, y_data):
    x, t = chainer.Variable(x_data), chainer.Variable(y_data)
    h1 = F.dropout(F.relu(model.l1(x)))     # 活性化関数はReLUを使用
    h2 = F.dropout(F.relu(model.l2(h1)))    # 活性化関数はReLUを使用
    # 論文"Dropout: A Simple Way to Prevent Neural Networks from Overfitting"
    # http://www.cs.toronto.edu/%7Ersalakhu/papers/srivastava14a.pdf
    # により提唱されている方法で、ランダムで中間層のニューロンをないものとして扱うことで
    # 過学習を防ぐことが知られている
    y = model.l3(h2)
    # 多クラス分類なので誤差関数としてソフトマックス関数の
    # 交差エントロピー関数を用いて、誤差を導出する
    # なお、回帰モデルであれば、二乗和誤差を用いる
    return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

# Setup optimizer
# 最適化セットアップ
# 深層学習の勾配法には様々な種類がある(有名どころは、SGD,Momentum,AdaGrad,Adam)
# SGD(Stochastic Gradient Descent:確率的勾配降下法)→chainerではη=0.01,α=0.9
# Momentum SGD→chainerではη=0.01,α=0.9
# AdaGrad→chainerではε=1e-8,η=0.001
# RMSprop→chainerではα=0.99,ε=1e-8,η=0.01
# AdaDelta→chainerではρ=0.95,ε=1e-6
# Adam→chainerではα=0.001,β1=0.9,β2=0.999,ε=1e-8
# 参考：https://qiita.com/tokkuman/items/1944c00415d129ca0ee9
opt = optimizers.Adam()   # 今回は最適化手法にAdamを使用する
opt.setup(model)          # 最適化対象をsetupに渡す

# Learning loop
for epoch in six.moves.range(1, args.n_epoch + 1):
    print('epoch', epoch)

    # training
    # N個の順番をランダムに並び替える
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    l1_W = []
    l2_W = []
    l3_W = []

    # 0〜Nまでのデータをバッチサイズごとに使って学習
    for i in six.moves.range(0, N, args.batchsize):

        x_batch = xp.asarray(x_train[perm[i:i + args.batchsize]])
        y_batch = xp.asarray(y_train[perm[i:i + args.batchsize]])

        model.zerograds()                       # 勾配を初期化
        loss, acc = forward(x_batch, y_batch)   # 順伝播させて誤差と精度を算出
        loss.backward()                         # 誤差逆伝播で勾配を計算
        opt.update()                            # 学習モデルに適用する

        train_loss.append(loss.data)
        train_acc.append(acc.data)
        sum_loss += float(loss.data) * len(y_batch)
        sum_accuracy += float(acc.data) * len(y_batch)

    # 訓練データの誤差と、正解精度を表示
    print('train mean loss={}, accuracy={}'.format(
        sum_loss / N, sum_accuracy / N))

    # evaluation
    # テストデータで誤差と、正解精度を算出し汎化性能を確認
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, N_test, args.batchsize):
        x_batch = xp.asarray(x_test[i:i + args.batchsize])
        y_batch = xp.asarray(y_test[i:i + args.batchsize])

        # 順伝播させて誤差と精度を算出
        loss, acc = forward(x_batch, y_batch)

        test_loss.append(loss.data)
        test_acc.append(acc.data)
        sum_loss += float(loss.data) * len(y_batch)
        sum_accuracy += float(acc.data) * len(y_batch)

    # テストデータでの誤差と、正解精度を表示
    print('test  mean loss={}, accuracy={}'.format(
        sum_loss / N_test, sum_accuracy / N_test))

    # 学習したパラメーターを保存
    l1_W.append(model.l1.W)
    l2_W.append(model.l2.W)
    l3_W.append(model.l3.W)

    # # 精度と誤差をグラフ描画
    # plt.figure(figsize=(8,6))
    # plt.plot(range(len(train_acc)), train_acc)
    # plt.plot(range(len(test_acc)), test_acc)
    # plt.legend(["train_acc","test_acc"],loc=4)
    # plt.title("Accuracy of digit recognition.")
    # plt.plot()

# 精度と誤差をグラフ描画
plt.figure(figsize=(8,6))

plt.subplot(221)
plt.plot(range(len(train_acc)), train_acc, "-r")
plt.legend(["train_acc"],loc=4)
plt.title("Accuracy of digit recognition of train_acc.")

plt.subplot(222)
plt.plot(range(len(test_acc)), test_acc, "-b")
plt.legend(["test_acc"],loc=4)
plt.title("Accuracy of digit recognition of test_acc.")

plt.subplot(223)
plt.plot(range(len(train_loss)), train_loss, "-r")
plt.legend(["train_loss"],loc=4)
plt.title("Loss of digit recognition of train_loss.")

plt.subplot(224)
plt.plot(range(len(test_loss)), test_loss, "-b")
plt.legend(["test_loss"],loc=4)
plt.title("Loss of digit recognition of test_loss.")

plt.show()
plt.close()
# matplotlibのスタイルを変える
plt.style.use('default')
# plt.style.use('bmh')
# plt.style.use('dark_background')
# plt.style.use('fivethirtyeight')
# plt.style.use('ggplot')
# plt.style.use('grayscale')
# plt.style.use('seaborn-bright')
# plt.style.use('seaborn-colorblind')
# plt.style.use('seaborn-dark')
# plt.style.use('seaborn-dark-palette')
# plt.style.use('seaborn-darkgrid')
# plt.style.use('seaborn-deep')
# plt.style.use('seaborn-muted')
# plt.style.use('seaborn-pastel')
# plt.style.use('seaborn-ticks')
# plt.style.use('seaborn-white')
# plt.style.use('seaborn-whitegrid')

def draw_digit3(data, n, ans, recog):
    size = 28
    plt.subplot(10, 10, n)
    Z = data.reshape(size,size)   # convert from vector to 28x28 matrix
    Z = Z[::-1,:]             # flip vertical
    plt.xlim(0,27)
    plt.ylim(0,27)
    plt.pcolor(Z)
    plt.title("ans=%d, recog=%d"%(ans,recog), size=8)
    plt.gray()
    plt.tick_params(labelbottom="off")
    plt.tick_params(labelleft="off")

# 答え合わせ
# 識別した100個の数字を表示
plt.figure(figsize=(15,15))
cnt = 0
for idx in np.random.permutation(N)[:100]:

    xxx = x_train[idx].astype(np.float32)
    h1 = F.dropout(F.relu(model.l1(chainer.Variable(xxx.reshape(1,784)))))
    h2 = F.dropout(F.relu(model.l2(h1)))
    y  = model.l3(h2)
    cnt += 1
    draw_digit3(x_train[idx], cnt, y_train[idx], np.argmax(y.data))
plt.show()
plt.close()
