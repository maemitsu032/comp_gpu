# comp_gpu

以下の例では、次の2点を満たす実装を示します。

1. **知識や考察を記載した部分をパースしやすくする**  
   - `[Knowledge]` ブロックに詳述された考察をテキストとして取得できるようにする

2. **最後のトークン（Yes または No）の logit を取得して、そのトークンが回答になるようにする**  
   - `[Final Label]:` の直後に **必ず1語** (Yes か No) を生成させ、その**生成タイミングのlogits**(確率分布) を参照できるようにする

ここでは、Ollama が `--logprobs` を付けたときに **JSON 形式** でトークンごとのログ確率を返す、という前提のサンプルコードを示します。実際には Ollama のバージョンやインターフェースにより返却フォーマットが異なる場合がありますので、必要に応じて修正してください。

---

## ポイント

- **プロンプトデザイン**  
  - `[Knowledge]` ブロックで考察を出力  
  - `[Final Label]:` の直後に Yes/No を1語だけ生成  
- **出力解析**  
  - テキスト全体から `[Knowledge]` 部分を正規表現等で抽出  
  - ログ確率情報（`logprobs`）から、最終トークン (Yes/No) のスコアを取り出してソフトマックス確率や logit を計算  
- **ラベル抽出**  
  - テキスト上にも Yes/No が書かれるのでパースは容易  
  - さらに logit 情報を使えば、Yes/No それぞれのスコア・確率を取得することも可能

---

```python
import pandas as pd
import subprocess
import json
import re
import math

def generate_prompt(row):
    """
    各行のメタデータから DeepSeek-14b (仮) 用のプロンプトを生成する関数。
    1. [Knowledge] ブロックに考察を詳述
    2. [Final Label]: の直後に Yes or No を1単語だけ出力
    """
    prompt = f"""以下はあるコンテンツのメタデータです。まず考察を詳述し、その後に「子供に適しているかどうか」をYes/Noで結論づけてください。
出力は必ず下記の形式を厳守してください。

[Knowledge]
(ここに考察を詳述)

[Final Label]:
(YesまたはNoのいずれかを1語で出力)

---
コンテンツ情報:
- コンテンツ形式: {row['content_type']}
- タイトル: {row['title']}
- 監督・ディレクター: {row['directors']}
- 出演者: {row['actors']}
- 公開された国: {row['countries']}
- コンテンツとして追加された日: {row['added_date']}
- 公開年: {row['release_year']}
- コンテンツの長さ: {row['duration']}
- ジャンル: {row['genres']}
- 説明文: {row['description']}

まず [Knowledge] ブロックに考察を記載し、
最後に [Final Label]: の直後に Yes or No を必ず1語だけで答えてください。
"""
    return prompt


def call_ollama_with_logprobs(prompt, model="deepseek-14b"):
    """
    subprocess 経由で Ollama CLI を呼び出し、--logprobs オプション付きで JSON を取得する。
    返却値は Ollama の出力を JSON デコードしたPythonオブジェクト。
    
    ※Ollamaのバージョンによってはフォーマットが異なる可能性があるので要確認。
    """
    try:
        # 例: ollama chat <モデル名> -p "<prompt>" --logprobs
        command = ["ollama", "chat", model, "-p", prompt, "--logprobs"]
        output = subprocess.check_output(command, encoding="utf-8")
        data = json.loads(output)
        return data
    except subprocess.CalledProcessError as e:
        print("Error during inference:", e)
        return None
    except json.JSONDecodeError as je:
        print("JSON decode error:", je)
        return None


def extract_knowledge_text(generated_text):
    """
    生成テキストから [Knowledge] ブロックに書かれた考察部分を抜き出す。
    '[Knowledge]' ～ '[Final Label]:' の範囲を正規表現で取得。
    """
    pattern = r"\[Knowledge\](?s)(.*?)\[Final Label\]:"
    match = re.search(pattern, generated_text)
    if not match:
        return None
    knowledge_text = match.group(1).strip()
    return knowledge_text


def extract_final_label(generated_text):
    """
    生成テキストから [Final Label]: の後ろにある Yes or No を1単語だけ抽出する。
    """
    pattern = r"\[Final Label\]:\s*([Yy]es|[Nn]o)\b"
    match = re.search(pattern, generated_text)
    if match:
        # 大文字小文字を統一
        return match.group(1).capitalize()  # "Yes" or "No"
    return None


def calculate_label_probs(logprobs_data):
    """
    Ollamaの --logprobs 出力（JSON）からトークンごとのログ確率を取り出し、
    最終的に生成された「Yes」または「No」トークンのlogitを取得してソフトマックス確率を計算する。
    
    ※実際に最後のトークンが "Yes" or "No" になっている想定で処理する。
    """
    # logprobs_data は Ollama が返す JSON 全体 (辞書)
    # 例: {
    #   "model": "...",
    #   "response": "最終的なテキスト全体...",
    #   "logprobs": {
    #       "tokens": ["..."],
    #       "token_ids": [...],
    #       "top_logprobs": [ { "Yes": <logprob>, "No": <logprob>, ... }, ... ],
    #       ...
    #   }
    # }
    if "logprobs" not in logprobs_data or "top_logprobs" not in logprobs_data["logprobs"]:
        return None
    
    top_logprobs_list = logprobs_data["logprobs"]["top_logprobs"]  # 各トークン毎の logprob 辞書のリスト
    tokens_list = logprobs_data["logprobs"]["tokens"]              # 実際に生成されたトークン列
    
    # 最終トークンがYes/Noであることを想定し、最後のトークンの top_logprobs を確認
    # ただし改行や空白トークンが生まれる可能性を考慮して、末尾からYes/Noを探す。
    yes_no_index = None
    
    # 末尾から探す
    for i in range(len(tokens_list) - 1, -1, -1):
        token = tokens_list[i].strip()
        if token.lower() in ["yes", "no"]:
            yes_no_index = i
            break
    
    if yes_no_index is None:
        # 見つからなかった場合
        return None
    
    # yes_no_index番目の top_logprobs を取り出す
    final_top_logprobs = top_logprobs_list[yes_no_index]  # 例: { "Yes": -1.2, "No": -2.3, ... }
    
    # そこから "Yes" / "No" のログ確率を取得
    logprob_yes = final_top_logprobs.get("Yes", None)
    logprob_no = final_top_logprobs.get("No", None)
    
    if logprob_yes is None or logprob_no is None:
        # 片方しかない、あるいは無い場合は None
        return None
    
    # ソフトマックス計算のため、相対的に exp する
    # （Pythonのmath.logの出力がlogprobなのでexpすると確率になる）
    # スケーリング (max) を取ってoverflow回避
    max_logprob = max(logprob_yes, logprob_no)
    p_yes = math.exp(logprob_yes - max_logprob)
    p_no = math.exp(logprob_no - max_logprob)
    s = p_yes + p_no
    prob_yes = p_yes / s
    prob_no = p_no / s
    
    return {
        "logprob_yes": logprob_yes,
        "logprob_no": logprob_no,
        "prob_yes": prob_yes,
        "prob_no": prob_no
    }


def main():
    # サンプルの DataFrame （実際にはご自身のデータを読み込み）
    data = {
        "content_type": ["映画", "テレビ"],
        "title": ["サンプル映画タイトル", "サンプルテレビタイトル"],
        "directors": ["山田太郎, 佐藤花子", "田中次郎"],
        "actors": ["俳優A, 俳優B", "俳優C, 俳優D"],
        "countries": ["日本, アメリカ", "日本"],
        "added_date": ["2023-01-01", "2023-02-15"],
        "release_year": [2022, 2021],
        "duration": ["120分", "1シーズン"],
        "genres": ["アクション, ドラマ", "コメディ, ファミリー"],
        "description": [
            "これはサンプル映画の説明文です。",
            "これはサンプルテレビ番組の説明文です。"
        ]
    }
    df = pd.DataFrame(data)
    
    knowledge_list = []
    final_label_list = []
    logprob_info_list = []
    
    for idx, row in df.iterrows():
        prompt = generate_prompt(row)
        
        print(f"--- Prompt for row {idx} ---")
        print(prompt)
        print("---------------\n")
        
        # Ollama で --logprobs を付けて推論
        result = call_ollama_with_logprobs(prompt)
        if not result:
            knowledge_list.append(None)
            final_label_list.append(None)
            logprob_info_list.append(None)
            continue
        
        # 生成されたテキスト全体
        generated_text = result.get("response", "")
        print(f"--- Generated Text for row {idx} ---")
        print(generated_text)
        print("---------------\n")
        
        # 1) [Knowledge] ブロック抜き出し
        knowledge_text = extract_knowledge_text(generated_text)
        
        # 2) [Final Label]: の直後に書かれた Yes/No をテキストから抽出
        final_label = extract_final_label(generated_text)
        
        # 3) 最終トークンのログ確率情報から Yes/No のスコアを計算
        logprob_info = calculate_label_probs(result)
        
        # 保存
        knowledge_list.append(knowledge_text)
        final_label_list.append(final_label)
        logprob_info_list.append(logprob_info)
        
        # デバッグ表示
        print(f"[Knowledge]\n{knowledge_text}\n")
        print(f"[Final Label]: {final_label}")
        print(f"LogProb Info: {logprob_info}\n")
        print("====================================\n")
    
    # DataFrame に格納
    df["knowledge"] = knowledge_list
    df["final_label"] = final_label_list
    df["logprob_info"] = logprob_info_list
    
    # 表示
    print("=== Final DataFrame ===")
    print(df[["title", "knowledge", "final_label", "logprob_info"]])

if __name__ == '__main__':
    main()
```

---

## 使い方・流れ

1. **DataFrame の各行**（各コンテンツのメタデータ）について、`generate_prompt()` でプロンプトを作成し、**[Knowledge]** ブロックと **[Final Label]:** の形式を強制。  
2. **Ollama** を `--logprobs` オプション付きで呼び出し、JSON を受け取る (`call_ollama_with_logprobs`)。  
3. 受け取った `result` から  
   - **生成テキスト** (`result["response"]`) をパースして `[Knowledge] ... [Final Label]: <Yes/No>` を抽出。  
   - **トークンごとのログ確率** (`result["logprobs"]`) を確認し、最終トークンが Yes/No となっている箇所を探して確率を計算 (`calculate_label_probs`)。  
4. 最後に各行に対して、  
   - 考察 (`knowledge_text`)  
   - 出力ラベル (`final_label`)  
   - ログ確率情報 (`logprob_info`)  
   を DataFrame にまとめる。

こうすることで

- **「知識や考察をテキストとして利用できる」**  
  → `[Knowledge]` 部分をそのままファインチューニングデータの追加テキストや特徴量に使える  
- **「最後のトークン (Yes/No) を logit（logprob）で扱う」**  
  → `calculate_label_probs()` で Yes/No の確率を算出し、「大規模モデルが最終的にどれだけの確信度を持ってYes/Noを出したのか」を定量的に捉える

という2点を同時に実現できます。

---

### 注意点

- 上記コードは **Ollama の JSON 出力形式** や **トークン化** が正しく機能し、かつ `[Final Label]` の次に本当に1語だけ生成されるという想定に基づきます。  
- 実際には改行やスペースなどが入る場合もあるため、運用時は**追加の正規表現処理**や**トークン列の末尾を慎重に処理**するなどの工夫が必要になる場合があります。  
- また、「ログ確率一覧の**どのトークンが本当に最終の Yes/No なのか**」を見極めるため、末尾から逆方向にトークンを探す実装になっています。

上記を調整しつつご利用ください。

