# 2048 Double DQN

## Getting Started

このプロジェクトは、Mini2048 ゲームを強化学習で解くための Double DQN エージェントを実装しています。以下の手順で環境をセットアップし、エージェントをトレーニングできます。

### uv のインストール

**uv**を使用して、プロジェクトの依存関係を管理します。
[uv インストール手順](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)
に沿って、uv をインストールしてください。

### プロジェクトのセットアップ

```bash
uv sync
```

を実行して、プロジェクトの依存関係をインストールします。

### トレーニングの実行

```bash
uv run src/learning.py
```

トレーニングを開始します。トレーニングの進行状況は、コンソールに表示されます。

### モデルの保存

トレーニングが完了したら、モデルは `models/` ディレクトリに保存されます。

### 引数

```bash
uv run src/learning.py --help
```

で、利用可能な引数を確認できます。

## 仕様技術

| カテゴリ       | 名前    |
| -------------- | ------- |
| 言語           | Python  |
| フレームワーク | PyTorch |
| パッケージ管理 | uv      |
