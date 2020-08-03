# CycelGANのPytorch実装
細かい話は以下記事へ．

# 実行環境
python: 3.7.7
pytorch: 1.5.1
OS: ubuntu
使用GPUメモリ: 9～11GB
依存ライブラリ: コード要確認

# 実行方法
1. dataディレクトリに本家リポジトリからhorse2zebra(もしくはそれ以外の)データセットをコピー．
data - horse2zebra - trainA
                   - trainB
                   - testA
                   - testB
2. python train.py -g 0 -m "model_name"
3. python test.py -g 0 -m "model_name"

# その他
* resultにhorse2zebra変換の結果例があります．
* sampleに指定イテレーション毎に変換例が格納されます．
* modelsにモデルが保存されます．
* identity mapping lossを用いる場合は，実行オプションで"--lambda_identity"を指定してください．