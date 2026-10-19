# 中職二軍打者的一軍表現預測

使用 Rebas 野球革命的公開資料，研究年輕打者的二軍表現與接續一軍表現之間的關係，並以 Ridge regression 產生一軍打擊指標預測。

目前建模與預測以**打者**為主；資料蒐集工具也支援投手，但尚未建立對應的投手預測模型。

> **目前模型使用 2020–2025 年資料訓練。2026 年資料只作推論，不會在預測過程中重新訓練或調整模型。**
>
> 2026 年預測對象已改為所有「二軍 PA >50、年初實足年齡 ≤25 歲」的打者，**不限制一軍 PA 或出賽紀錄**。先前「一軍 PA＝0」的條件已移除。

## 1. 研究設計

### 訓練資料與預測資料的差別

| 項目 | 模型訓練 | 2026 年預測 |
|---|---|---|
| 年份 | 目前使用 2020–2025 年 | 2026 年 |
| 一筆資料 | 同季某段二軍出賽，接續某段一軍出賽 | 一位球員的二軍例行賽整季數據 |
| 年齡 | 球季當年 1 月 1 日實足年齡 ≤25 歲 | 2026 年 1 月 1 日實足年齡 ≤25 歲 |
| 二軍打席門檻 | **區間 PA ≥50**，包含剛好 50 | **整季 PA >50**，至少 51 |
| 一軍條件 | 須有接續一軍區間；各目標排除分母為零的案例 | 不限制一軍打席或出賽 |
| 跨區間處理 | 不將同一球員不同轉換區間的 PA 合併 | 同球員跨球隊的二軍整季計數加總 |
| 用途 | 擬合模型、交叉驗證與保留集評估 | 輸出預測，不擬合、不計算 MAE／R² |

訓練區間依**實際出賽日期**定義，不是正式升降登錄日期。一軍→二軍→一軍的回歸情況也可能形成案例，因此不是只研究生涯第一次升上一軍。

### 模型輸入與輸出

四個獨立 Ridge 模型共用 11 個二軍輸入特徵：

| 類型 | 特徵 |
|---|---|
| 基本打擊率 | `minor_AVG`、`minor_OBP`、`minor_SLG` |
| 衍生打擊指標 | `minor_OPS`、`minor_ISO` |
| 三振與保送 | `minor_K_pct`、`minor_BB_pct` |
| 每打席比率 | `minor_HR_PA`、`minor_HBP_PA` |
| 每打數比率 | `minor_Double_AB`、`minor_Triple_AB` |

特徵由基礎計數重新計算，不直接使用排行榜已四捨五入的率值；訓練與預測共用同一組計算函式。

預測目標：

- `major_AVG`：一軍打擊率。
- `major_OBP`：一軍上壘率。
- `major_SLG`：一軍長打率。
- `major_HR_PA`：一軍每打席全壘打率，不是全壘打支數。

年齡、姓名、球員 ID、年份、一軍數據都**不作為 X 特徵**。目前缺乏足夠逐場資訊的 BABIP、揮空率、wOBA 等指標，不以整季數值補進訓練區間。

### 一軍 PA 加權

四個主模型皆採用：

```text
訓練權重 = 該案例一軍 PA ÷ 當前訓練折的一軍平均 PA
```

例如一軍 100 PA 的案例，其訓練損失權重是一軍 10 PA 案例的 10 倍；不是將預測值乘以 10。

- 每次交叉驗證，只使用該訓練折計算權重正規化所需的平均。
- 缺值補值與 Z-score 標準化也只在訓練折擬合，前處理本身不加權。
- 以訓練集內 5-fold 交叉驗證的 **PA 加權 pooled out-of-fold MAE** 選擇 alpha。
- 最後同時報告等案例與 PA 加權評估結果。
- 一軍 PA 只影響訓練與評估，不是推論時必須提供的模型特徵。

案例以 seed＝42 隨機切成 80% 訓練、20% 驗證。alpha 候選值為 `0.01、0.1、1、10、100、1000、10000`。

## 2. 專案檔案

| 檔案 | 用途 |
|---|---|
| [rebas_crawler.py](./rebas_crawler.py) | 下載一、二軍球季打者／投手統計，保存 CSV、原始 JSON 與來源紀錄 |
| [filter_same_season.py](./filter_same_season.py) | 篩選同季兩個層級皆出賽的球員，供建立訓練區間使用 |
| [filter_age.py](./filter_age.py) | 取得生日並依年初實足年齡篩選 |
| [rebas_transitions.py](./rebas_transitions.py) | 下載逐場紀錄，建立二軍→一軍區間與品質報告 |
| [train_batter.py](./train_batter.py) | 篩選訓練案例、擬合四個一軍 PA 加權 Ridge 模型 |
| [predict_batter_2026.py](./predict_batter_2026.py) | 篩選 2026 年球員並載入既有模型推論 |
| [diagnose_batter.py](./diagnose_batter.py) | 依出賽量等條件分析既有驗證誤差 |
| [compare_weighted_batter.py](./compare_weighted_batter.py) | 比較既有模型與「各目標分母加權」模型 |
| [compare_single_batter.py](./compare_single_batter.py) | 比較多特徵與單特徵模型，並進行球員群集 bootstrap |
| [requirements_training.txt](./requirements_training.txt) | 建模與推論所需的 Python 套件 |

原始與整理後的資料位於 [data](./data/)；常規訓練輸出位於 [training_outputs](./training_outputs/)，推論輸出位於 [prediction_outputs](./prediction_outputs/)。

## 3. 執行環境

程式以 Python 3.9+ 語法撰寫。爬蟲、年齡篩選及逐場整理使用標準函式庫；建模與推論需要 NumPy、pandas、scikit-learn、joblib。

以下指令均在**專案根目錄的 WSL／Bash** 執行。已有 `datagame` 環境時可先啟用：

```bash
conda activate datagame
```

若是新環境，才需安裝相依套件：

```bash
python -m pip install -r ./requirements_training.txt
```

> 載入既有模型應使用訓練時的套件環境。預測程式會檢查 scikit-learn 版本是否與模型紀錄相同；僅滿足相依檔的最低版本，不保證能載入所有舊模型。只載入可信來源的 joblib 檔案。

**WSL 路徑使用 `/`。** 不要在 Bash 中使用未加引號的 `.\training_outputs\...`，反斜線會被解讀為跳脫字元。

大多數產生新資料夾的指令都拒絕覆寫。下方指定的輸出名稱若已存在，請換新名稱；不要為了重跑而刪除原始資料或舊結果。

## 4. 使用現有資料訓練

[train_batter.py](./train_batter.py) 的預設輸入目前指向專案已整理好的歷史年齡資料與轉換案例，而不是自動搜尋最新資料夾。

在本專案現有資料完整的情況下：

```bash
python ./train_batter.py \
  --output ./training_outputs/minor50_major_pa_weighted_run1
```

若改用其他資料，請明確指定 `--episodes` 與 `--ages`，不要只更換輸出位置。

### 目前已有的模型

本專案目前用於 2026 預測的模型，位於：

[.training_outputsminor50_major_pa_weighted_run1](./.training_outputsminor50_major_pa_weighted_run1/)

這是先前在 Bash 使用 Windows 路徑寫法所產生的隱藏資料夾，模型本身可正常使用。它與其他歷史訓練結果是不同批次。

後續範例以變數指定這批既有模型：

```bash
MODEL_RUN='./.training_outputsminor50_major_pa_weighted_run1'
```

若使用剛重新訓練的模型，請改為其實際位置，例如：

```bash
MODEL_RUN='./training_outputs/minor50_major_pa_weighted_run1'
```

### 訓練輸出

每批輸出包含：

| 檔名 | 內容 |
|---|---|
| `dataset.csv` | 符合條件的案例、特徵、目標與 train／validation 分組 |
| `excluded_cases.csv` | 資料準備時排除的案例及原因；年齡排除數另見 manifest |
| `major_AVG.joblib` 等四個模型檔 | 模型 pipeline、特徵順序與相關設定 |
| `validation_predictions.csv` | 驗證集實績、預測、基準與一軍 PA |
| `metrics.json` | MAE、RMSE、R² 與各種基準比較 |
| `training_cv.csv` | 各目標、各 alpha 的交叉驗證結果 |
| `coefficients.csv`、`preprocessing.csv` | 係數、補值與標準化參數 |
| `manifest.json` | 資料來源、門檻、權重、版本、樣本數與執行狀態 |

主要先比較 `metrics.json` 中的：

- `validation_major_pa_weighted`：模型的 PA 加權誤差。
- `pa_weighted_train_mean_baseline_major_pa_weighted`：以訓練集 PA 加權平均作固定預測的誤差。
- `validation`：每個案例等權重時的誤差。

MAE、RMSE 越低越好。不要只看模型分數下降，也要確認是否勝過**相同評估方式下的基準**。不同樣本、不同權重的分數不能直接視為同一個比較。

## 5. 預測 2026 年符合條件的所有二軍打者

### 步驟一：準備球季資料

若 [data/rebas_2026_prediction1](./data/rebas_2026_prediction1/) 已存在，且其 manifest 顯示 `status: complete`，可直接使用，不必再次執行下載。

若尚未下載：

```bash
python ./rebas_crawler.py \
  --years 2026 \
  --levels major minor \
  --roles batter \
  --output ./data/rebas_2026_prediction1
```

爬蟲預設仍為 2020–2025 年；2026 年必須明確指定。

### 步驟二：執行推論

先設定前一節的 `MODEL_RUN`，再執行：

```bash
python ./predict_batter_2026.py \
  --data ./data/rebas_2026_prediction1 \
  --run "$MODEL_RUN" \
  --output ./prediction_outputs/2026_all_minor50_run2
```

若已有下載過的生日資料，可額外加上以下參數，重用 profile JSON；找不到的球員仍會下載：

```bash
--cache ./prediction_outputs/2026_all_minor50_run1/raw
```

本流程：

1. 按球員 ID 彙整 2026 年二軍資料，跨球隊加總基礎計數。
2. 保留二軍 PA >50 且計數合理者。
3. 取得生日，保留 2026 年初實足年齡 ≤25 歲者。
4. 以共用公式建立 11 個二軍特徵。
5. 載入四個既有主模型，只呼叫預測，不重新擬合。

目前不限制國籍，也不判定正式新秀資格。未列於一軍排行榜的參考數據保持空白，**不把缺資料推定成零**。生日缺失或超齡者會明確排除；一軍打擊計數的邏輯不一致會標記，但不作為候選資格篩選。

### 步驟三：看預測結果

新批次的 `predictions.csv` 可用 Excel 開啟。主要欄位：

| 欄位 | 意義 |
|---|---|
| `player_name`、`team_abbr` | 球員與二軍球隊；跨隊時可能列出多隊 |
| `age_jan1` | 2026 年初實足年齡 |
| `minor_PA`、`minor_AVG` 等 | 實際二軍數據及重算特徵 |
| `major_PA`、`major_games` | 已下載球季排行榜中的一軍參考數據，並非篩選條件 |
| `major_table_status` | `listed` 表示一軍表有紀錄；`not_listed` 表示未列出 |
| `predicted_major_AVG` | 預測一軍打擊率 |
| `predicted_major_OBP` | 預測一軍上壘率 |
| `predicted_major_SLG` | 預測一軍長打率 |
| `predicted_major_HR_PA` | 預測一軍每打席全壘打率 |
| `major_*_out_of_range` | 預測是否超出該率值的合理範圍；不自動裁切 |
| `seen_in_training` | 該球員 ID 是否曾出現在模型的 train 案例中 |
| `evaluation_status` | 目前為 `prediction_only_not_scored` |
| `feature_window` | 目前為 `full_2026_minor_regular_season` |

例如 `predicted_major_HR_PA = 0.008` 是 **0.8%**，不是 0.008%。若假設取得 100 PA，對應預期約 0.8 支全壘打，並不是保證會打出的數量。

在 WSL 印出精簡表格：

```bash
python - <<'PY'
import pandas as pd

df = pd.read_csv("./prediction_outputs/2026_all_minor50_run2/predictions.csv")
columns = [
    "player_name", "team_abbr", "age_jan1", "minor_PA", "major_PA",
    "predicted_major_AVG", "predicted_major_OBP",
    "predicted_major_SLG", "predicted_major_HR_PA",
]
print(df[columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
PY
```

同批還會保存 `excluded.csv`、`manifest.json` 與 `raw`。請先確認 manifest 的 `status` 為 `complete`；失敗或中斷的輸出不代表完整結果。

> [prediction_outputs/2026_run1](./prediction_outputs/2026_run1/) 是先前限制一軍 PA＝0 的舊批次；[prediction_outputs/2026_all_minor50_run1](./prediction_outputs/2026_all_minor50_run1/) 是移除一軍 PA 限制後的批次。更新程式不會自動改寫舊 CSV，解讀結果時應同時看該批 manifest。

## 6. 從原始資料重建訓練案例

已有整理好的資料時可略過本節。完整資料流程是：

```text
球季統計
  → 同季一、二軍均有出賽
  → 年初實足年齡 ≤25 歲
  → 逐場紀錄與二軍→一軍轉換區間
  → 二軍區間 PA ≥50、品質與目標可用性檢查
  → PA 加權模型訓練
```

下載歷史打者資料並篩選：

```bash
python ./rebas_crawler.py \
  --years 2020 2021 2022 2023 2024 2025 \
  --levels major minor \
  --roles batter \
  --output ./data/rebas_train_new

python ./filter_same_season.py ./data/rebas_train_new/all_batter.csv
python ./filter_age.py ./data/rebas_train_new/all_batter_same_season.csv
```

年齡篩選會產生帶時間戳的資料夾。**先把以下 `AGE_DIR` 替換成終端機顯示的實際輸出位置**，再執行：

```bash
AGE_DIR='./data/rebas_train_new/age25_請替換實際時間戳'

python ./rebas_transitions.py "$AGE_DIR/filtered.csv" \
  --output ./data/batter_transitions_new

python ./train_batter.py \
  --episodes ./data/batter_transitions_new/episodes.csv \
  --ages "$AGE_DIR/filtered.csv" \
  --output ./training_outputs/minor50_major_pa_weighted_new
```

年齡篩選與逐場整理都可透過 `--cache` 指定既有的原始 JSON 目錄，減少重複請求。不要把 2026 年推論資料混入這條歷史訓練流程。

## 7. 額外診斷與模型比較

所有命令都建議明確指定資料夾。這些腳本的預設位置可能指向較早的歷史實驗，而不是目前使用的主模型。

```bash
python ./diagnose_batter.py --run "$MODEL_RUN"
python ./compare_weighted_batter.py --run "$MODEL_RUN"
```

**主模型已經使用一軍 PA 加權，不需要執行比較腳本才會加權。**

[compare_weighted_batter.py](./compare_weighted_batter.py) 的額外實驗採用「各目標分母加權」：

- AVG、SLG：一軍 AB。
- OBP：一軍 AB + BB + HBP + SF。
- HR/PA：一軍 PA。

這不同於主模型四項一律使用一軍 PA 的設計。

完成後可對同一批加權比較做單特徵比較；先將 `COMPARISON` 替換為實際產生的資料夾：

```bash
COMPARISON="$MODEL_RUN/weighted_comparison_請替換實際時間戳"

python ./compare_single_batter.py \
  --run "$MODEL_RUN" \
  --comparison "$COMPARISON" \
  --bootstrap 2000 \
  --seed 42
```

單特徵實驗為二軍 AVG→一軍 AVG、OBP→OBP、SLG→SLG、HR/PA→HR/PA；這些單特徵模型同樣採各目標分母加權。Bootstrap 以球員為群集，差異定義為「多特徵誤差 − 比較對象誤差」，負值代表多特徵較低。

## 8. 測試與常見訊息

執行內建單元測試，使用模擬資料，不會下載整批研究資料或重新訓練正式模型：

```bash
python -m unittest \
  rebas_crawler.CrawlerTests \
  filter_same_season.FilterTests \
  filter_age.AgeTests \
  rebas_transitions.TransitionTests \
  train_batter.TrainingTests \
  predict_batter_2026.PredictionTests \
  compare_weighted_batter.WeightedTests \
  compare_single_batter.ComparisonTests
```

| 訊息 | 意義與處理 |
|---|---|
| 輸出資料夾已存在 | 換新批次名稱，或使用已有完整輸出；不要覆蓋舊研究結果 |
| `players appear in BOTH subsets` | 同一球員跨訓練／驗證集出現，評估不是未見球員測試 |
| `best alpha is search upper bound` | 交叉驗證選到候選最大 alpha，不是執行失敗，也不保證擴大範圍就會改善 |
| `No eligible players with verified ages` | 檢查該批排除名單、生日來源與門檻；不會產生假的成功預測 |
| scikit-learn 版本不符 | 使用模型 manifest 記錄的訓練環境載入 |
| `prediction_only_not_scored` | 正常的推論狀態，不表示預測準確度已驗證 |

## 9. 研究限制與資料使用

1. **隨機切案例不等於未來預測能力。** 同一球員可能出現在兩側，訓練案例也可能晚於驗證案例；若要主張未來球季或新球員泛化能力，需另外設計時間或球員分組驗證。
2. **訓練樣本帶有取得一軍機會的選擇偏差。** 模型學自已形成二軍→一軍區間的球員，套用到尚無一軍打席者，不代表已證明能可靠外推。
3. **訓練與 2026 推論的觀察窗口不同。** 訓練使用先二軍、後一軍的區間；推論使用二軍整季數據，可能與同年一軍數據交錯。即使已有一軍實績，也不能直接拿整季值比較後宣稱是無時間洩漏的前瞻測試。
4. **短一軍區間仍有很大的隨機波動。** PA 加權降低短樣本的相對影響，但不是消除不確定性；也會讓出賽機會較多者更影響模型。
5. **不要把小數點後的微小差距直接當成實力排名。** 應同時看基準、樣本量、權重與不確定性。Ridge 係數也不是因果效果。
6. **Bootstrap 結果是探索性的。** 現有比較重複使用驗證集，信賴區間以固定模型為條件，未涵蓋重新擬合的不確定性，也未作多重比較校正。
7. **資料可能有來源缺漏。** 同日跨層級無法判定先後、逐場缺失及不合理計數都有對應的排除或品質紀錄；應一併檢查，不能把缺失直接視為零。
8. **沒有自動納入新的球季更新。** 每批資料、生日來源與模型均以當次執行紀錄為準；需要更新時應建立新的資料與輸出批次。

資料來源：[Rebas 野球革命](https://www.rebas.tw/)。

爬蟲使用公開 API，遵守 robots.txt、設定請求間隔並保留來源紀錄，不登入也不繞過存取限制。**公開可存取不等於可任意再散布**；研究、競賽與發表前，仍應確認網站最新條款及使用授權，並清楚註明資料來源。
