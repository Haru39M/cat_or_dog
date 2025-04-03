# cat_or_dog
犬猫を学習して判定するやつ

動作環境：Python 3.10.6

↑TensorFlowがPython3.12まで対応しているので新規に環境構築するなら3.12がおすすめ

## 使い方
### 学習時
main.pyを実行

### 推論時
cat_and_dog.pyを実行

results/にある学習済みモデルを使う

ローカルでflaskのWEBアプリが立ち上がるので，適当な画像を突っ込むと判定される．
