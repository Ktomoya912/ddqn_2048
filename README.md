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

| category        | name    |
| --------------- | ------- |
| language        | Python  |
| framework       | PyTorch |
| package manager | uv      |

## Training Method

### learning.py

DDQN の論文の手法をそのまま実装。

**メインネットワーク**:

- メインネットワークの評価値に基づいて行動を選択
- 選択した行動のターゲットネットワークの評価値をキューに追加
- キューからターゲット評価値を取得
- ターゲット評価値を使ってメインネットワークを学習

**ターゲットネットワーク**:

- 一定期間（[period]ステップ）ごとにメインネットワークの重みをコピー
- 安定した学習のための目標値を提供

### learning-toggle.py

ゲームごとに学習するモデルとキューを切り替えて相手の評価値を学習する方法を実装。

| Pack1                          | Pack2                          |
| ------------------------------ | ------------------------------ |
| play from Pack1value           | play from Pack2value           |
| put Pack2value Pack1Queue      | put Pack1value Pack2Queue      |
| get Pack2value from Pack1Queue | get Pack1value from Pack2Queue |
| learn from Pack2value          | learn from Pack1value          |
