# YOLOv5でマスク着用判定

## 概要

物体認識アルゴリズムYOLOv5を使用して人物がマスクを着用しているかしていないか判定する。

### 使用環境・データ

Google colaboratory 

Mask Wearing Dataset

### 大まかな手順

1. 作業フォルダ準備
2. YOLOv5インストール・データセットダウンロード
3. 訓練データ学習
4. テストデータ推論

## 作業フォルダの準備

1. Google Driveの任意の場所に作業するフォルダを作成。ここでは例としてMy drive直下に「YOLOv5_mask」というフォルダを作成して説明する。
2. 「YOLOv5」にgithubからダウンロードした「YOLOv5.ipynb」を設置する。
3. roboflowの「Mask Wearing Dataset」をダウンロード。フォーマットは「YOLO v5 Pytorch」また、「download zip computer」を選択。
4. ダウンロードしたzipファイルを解凍（Mask Wearing~という名前のフォルダ）
5. 解凍したフォルダから、「train」「valid」の２つのフォルダを「YOLOv5_mask」にアップロードする

#### この時点のフォルダ構成

- YOLOv5_mask内に
  - train
  - valid
  - YOLOv5.ipynbの３つが存在

確認できたらYOLOv5.ipynbを起動する

### GPU設定

割り当てられるGPUを確認。

```python
!nvidia-smi
```

Tesla T4 / Tesla K80 / Tesla P100

のいずれかが表示される（稀にV100が出る場合がある）

計算速度が一番速いP100が出るまでランタイムをリセットする。

### googleドライブのマウント

出力に表示されるリンクをクリック→アクセスを許可→表示されたコードを「Enter your authorization code:」のインプット欄にペーストしてEnterキーを押す。

```python
from google.colab import drive
drive.mount('/content/drive')
```

「YOLOv5_mask」に移動

```python
import os
os.chdir("/content/drive/My Drive/YOLOv5_mask")
```

### YOLOv5の設定

```python
!git clone https://github.com/ultralytics/yolov5  # githubからclone
!pip install -qr yolov5/requirements.txt  # 必要なライブラリをインストール(エラーを無視)
%cd yolov5 # yolov5フォルダに移動

import torch
from IPython.display import Image, clear_output  # 画像を表示する
from utils.google_utils import gdrive_download  # モデルをダウンロード

clear_output()
print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))
```

成功すると「YOLOv5_mask」フォルダ内に「YOLOv5」フォルダが作成され、位置が「yolov5」に変更される。

```
pwd
```

/content/drive/My Drive/yolov5/YOLOv5_mask/yolov5と表示される。

### デフォルトデータで推論

YOLOv5をインストールした際にダウンロードされたデフォルトのデータを利用してYOLOv5の動きを確認する。

```python
!python detect.py --weights yolov5s.pt --img 416 --conf 0.4 --source inference/images/
Image(filename='inference/output/zidane.jpg', width=600)
```

<img src="/Users/imanim/Library/Application Support/typora-user-images/スクリーンショット 2020-09-02 13.38.44.png" alt="スクリーンショット 2020-09-02 13.38.44" style="zoom:40%;" />

引数のsourceで推論に使用するファイルを指定する。

```python
!python detect.py --source 0  # webカメラ
                           file.jpg  # 画像 
                           file.mp4  # 動画
                           path/  # フォルダで指定
                           path/*.jpg  # フォルダのファイルで指定
                            rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa  # rtsp
                            http://112.50.243.8/PLTV/88888888/224/3221225900/1.m3u8  # http 
```

--weights　重み設定ファイルの指定。デフォルトではyolov5s.ptで、一番軽いモデル。

他にはyolov5m.ptやyolov5l.ptなど。yolov5x.ptが精度が高いが、計算には時間がかかる。

注意：detect.pyを実行することでデフォルトデータで使用した重みファイルがgithubからダウンロードされる。次のステップでtrain.pyを行う際には重みファイルはダウンロードされないので、学習に使用したい重みを設定しておくこと。

良い例）detect.py --weights yolov5s.pt　→　train.py --cfg yolov5s.yaml 

エラー例) detect.py --weights yolov5s.pt　→　train.py --cfg yolov5x.yaml 



--img　画像のサイズを指定(px)

### Trainデータを学習

学習が早くなる。インストールに５分程度かかる。

```python
!git clone https://github.com/NVIDIA/apex
!pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex
```

学習を実行する前に、「yolov5」フォルダ内の「data」フォルダに、roboflowでダウンロードした「Mask Wearing~」フォルダ内の「data.yaml」を移動する。

ちなみにData.yamlを確認すると

```yaml
train: ../train/images
val: ../valid/images

nc: 2
names: ['mask', 'no-mask']
```

となっている。

「data.yaml」を移動したら以下を実行して学習を開始する。バッチ数とエポック数は以下の数字を変更すると任意の値で学習が可能になる。

```python
!python train.py --img 416 --batch 16 --epochs 300 --data data.yaml --cfg yolov5x.yaml
```

学習には２~3時間かかるので、ランタイムが途中で切れることに注意。

### 学習したデータをもとに推論

学習した重みを使用して推論を行う。学習した重みは runs/exp0/weights 内に「best.pt」という名前で保存されている。

1. 推論したい画像を「Mask Wearing~」フォルダのtest/imagesから選択。
2. 選択した画像を「yolov5」フォルダ内に移動して名前を「test.jpg」に変更。
3. 以下を実行する。

```python
!python detect.py --source test.jpg --weights runs/exp2/weights/best.pt --img 416
Image(filename='inference/output/test.jpg', width=419)
```

結果は、inference/output内に「test.jpg」として保存される。
