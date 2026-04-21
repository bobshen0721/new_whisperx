# Company Setup

This repository is prepared so a company machine can get both code and the `large-v2` Whisper model from GitHub only.

## What is included in the repository

- WhisperX source code
- Small `large-v2` model files under `models/faster-whisper-large-v2/`
  - `config.json`
  - `tokenizer.json`
  - `vocabulary.txt`
- PowerShell scripts to download split model assets from a GitHub Release and rebuild `model.bin`

## What is stored in the GitHub Release

Because `faster-whisper-large-v2/model.bin` is about `3.09 GB`, GitHub Free cannot host it as a single asset. The Release will contain split parts instead, plus checksums recorded in `models/faster-whisper-large-v2/release-manifest.json`.

## Company machine steps

1. Clone or download this repository from GitHub.
2. Open PowerShell in the repository root.
3. Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\download_and_reconstruct_large_v2.ps1 -Owner <github-owner> -Repo <github-repo> -Tag model-large-v2
```

4. After the script completes, confirm this file exists:

```text
.\models\faster-whisper-large-v2\model.bin
```

## Example WhisperX usage with local model files

```powershell
python -m whisperx .\sample.wav --model .\models\faster-whisper-large-v2 --language zh --device cuda --output_dir .\output
```

## Important notes

- This repository does not vendor every Python dependency or CUDA runtime.
- The company machine still needs a working Python environment, plus whichever CUDA or CPU runtime you plan to use.
- WhisperX source code here keeps its original BSD 2-Clause license.
- `Systran/faster-whisper-large-v2` is redistributed here based on its upstream MIT license and model card.
