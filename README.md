# Deepvoice3の再現実装

[r9y9](https://github.com/r9y9/deepvoice3_pytorch) 様の実装したDeepvoice3を、より論文に近いネットワーク構造へと実装し直しました。具体的な変更点は

・言語処理部を、論文の形式へと変更

・1×1convをすべて全結合層（FC）へと変更
・attention layerを全てのDecoder layerに適用
・positional encodingはEmbeddingで特徴量次元に合わせるのではなく、特徴量方向にexpandしたものを使用
・各種ハイパーパラメータを論文に遵守

1. [arXiv:1710.07654](https://arxiv.org/abs/1710.07654): Deep Voice 3: Scaling Text-to-Speech with Convolutional Sequence Learning.
