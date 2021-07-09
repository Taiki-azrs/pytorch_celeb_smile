import torch
import torch.nn as nn
import torch.nn.functional as F
class Model(nn.Module):
    def __init__(self):
        # スーパークラス（Module クラス）の初期化メソッドを実行 
        super().__init__() 

        self.c0 = nn.Conv2d(in_channels=3,    # 入力は3チャネル
                            out_channels=16,  # 出力は16チャネル
                            kernel_size=3,    # カーネルサイズは3*3
                            stride=2,         # 1pix飛ばしでカーネルを移動
                            padding=1)        # 画像の外側1pixを埋める

        self.c1 = nn.Conv2d(in_channels=16,   # 入力は16チャネル
                            out_channels=32,  # 出力は32チャネル
                            kernel_size=3,    # カーネルサイズは3*3
                            stride=2,         # 1pix飛ばしでカーネルを移動
                            padding=1)        # 画像の外側1pixを埋める

        self.c2 = nn.Conv2d(in_channels=32,   # 入力は32チャネル
                            out_channels=64,  # 出力は64チャネル
                            kernel_size=3,    # カーネルサイズは3*3
                            stride=2,         # 1pix飛ばしでカーネルを移動
                            padding=1)        # 画像の外側1pixを埋める          

        self.bn0 = nn.BatchNorm2d(num_features=16)   # c0用のバッチ正則化
        self.bn1 = nn.BatchNorm2d(num_features=32)   # c1用のバッチ正則化
        self.bn2 = nn.BatchNorm2d(num_features=64)   # c2用のバッチ正則化

        self.fc = nn.Linear(in_features=64 * 28 * 28,   # 入力サイズ
                            out_features=4)             # 各クラスに対応する4次元のベクトルに変換

    def __call__(self, x): # 入力から出力を計算するメソッドを定義
        h = F.relu(self.bn0(self.c0(x)))
        h = F.relu(self.bn1(self.c1(h)))
        h = F.relu(self.bn2(self.c2(h)))  
        h = h.view(-1, 64 * 28 * 28)
        y = self.fc(h)     # 全結合層
        return y
