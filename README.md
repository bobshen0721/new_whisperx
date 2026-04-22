# 語音轉錄平台2.0v

這個 repo 是整理過的 WhisperX 公司使用版，目標很簡單：

- GitHub 下載 ZIP 後就知道怎麼用
- `large-v2` 模型只從 GitHub 取得
- 直接啟動 Gradio 網頁介面做語音轉錄

目前保留的核心內容只有：

- `app.py`
  Gradio 網頁介面主程式
- `whisperx/`
  WhisperX 核心程式
- `models/faster-whisper-large-v2/`
  已放入小檔與 manifest；`model.bin` 由 GitHub Release 下載重組
- `tools/`
  模型下載與重組腳本
- `01_download_large_v2_model.bat`
  一鍵下載並重組 `large-v2`
- `02_start_webui.bat`
  一鍵啟動網頁介面

## 你可以做什麼

- 上傳最長 2 小時音檔
- 做 ASR 轉錄
- 做 VAD
- 做 word-level alignment
- 有 Hugging Face token 時做 speaker diarization
- 將 speaker 重新整理成較可讀的區塊
- 點逐字稿時間軸跳播
- 匯出 `txt / srt / json / html / zip`

## 下載方式

你可以用這兩種方式拿到專案：

1. 直接從 GitHub 下載 ZIP
2. `git clone` 這個 repo

專案網址：
[new_whisperx](https://github.com/bobshen0721/new_whisperx)

`large-v2` 模型分片在這個 Release：
[model-large-v2](https://github.com/bobshen0721/new_whisperx/releases/tag/model-large-v2)

## 第一次使用

### 1. 準備 Python 環境

建議：

- Python `3.10` 到 `3.13`
- Windows
- 有 NVIDIA GPU 時可用 CUDA

這個 repo **不包含** Python 本體、CUDA runtime、或所有第三方套件安裝檔。

如果你的公司電腦不能直接連 PyPI，請用公司內部 mirror 或你已經準備好的 Python 環境。

### 2. 安裝 Python 套件

在 repo 根目錄執行：

```powershell
pip install uv
uv sync --all-extras
```

如果你使用的是特定 Python：

```powershell
py -3.11 -m pip install uv
uv sync --all-extras
```

`uv sync` 之後，依賴通常會裝在 repo 內的 `.venv`。  
`02_start_webui.bat` 會優先使用這個 `.venv`，不用你手動 activate。

### 3. 下載 `large-v2` 模型

直接雙擊：

[01_download_large_v2_model.bat](./01_download_large_v2_model.bat)

它會做這些事：

- 從 GitHub Release 下載分片
- 驗證 checksum
- 重組成：
  `models/faster-whisper-large-v2/model.bin`

如果你想手動執行 PowerShell 版：

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\download_and_reconstruct_large_v2.ps1 -Owner bobshen0721 -Repo new_whisperx -Tag model-large-v2
```

### 4. 設定 Hugging Face token

如果你要 speaker diarization，請先設定：

```powershell
setx WHISPERX_HF_TOKEN "你的 Hugging Face Token"
```

重新開一個 PowerShell 視窗後才會生效。

如果沒有 token，系統仍然可以轉錄，只是 speaker 會退回 `UNKNOWN`。

## 啟動方式

最簡單的方法是直接雙擊：

[02_start_webui.bat](./02_start_webui.bat)

或在 PowerShell 執行：

```powershell
.venv\Scripts\python.exe app.py --server-name 127.0.0.1 --server-port 7860
```

啟動後打開：

[http://127.0.0.1:7860](http://127.0.0.1:7860)

## 網頁介面使用方式

1. 上傳音檔
2. 選擇語言
3. 選擇講者人數
   預設是 `自動推斷`
   如果你已知就是 2 位或 4 位，也可以手動固定
4. 按 `開始轉錄`
5. 在右邊查看逐字稿
6. 點任一段逐字稿，左邊播放器會跳到該時間
7. 需要時下載輸出檔

## 輸出檔在哪裡

每次轉錄結果會放在：

```text
gradio_outputs\
```

通常會包含：

- `.txt`
- `.srt`
- `.json`
- `.html`
- `_bundle.zip`

## 建議使用流程

公司電腦如果只是要拿來使用，不需要先研究整個程式碼。照這個順序就可以：

1. 下載 ZIP 並解壓
2. 安裝 Python 套件
3. 執行 `01_download_large_v2_model.bat`
4. 視需要設定 `WHISPERX_HF_TOKEN`
5. 執行 `02_start_webui.bat`

## 重要提醒

- GitHub Free 不能直接把 `large-v2/model.bin` 以單檔放在 repo，所以改成 Release 分片下載
- 這個 repo 主要是「可直接使用的整理版」，不是完整保留上游研究說明的版本
- 預設會優先使用本機的 `models/faster-whisper-large-v2`
- 若 GPU 記憶體不足，可以改用較小模型，例如：

```powershell
python app.py --model tiny
```

## 保留的必要結構

```text
.
├─ app.py
├─ whisperx/
├─ models/
├─ tools/
├─ 01_download_large_v2_model.bat
├─ 02_start_webui.bat
├─ README.md
├─ pyproject.toml
└─ uv.lock
```

## 授權

- WhisperX 原始碼：BSD 2-Clause
- `faster-whisper-large-v2` 模型：依其上游授權與模型卡使用
