# Training Documentation

## 🗂️ 1. Data Preparation

### Automatic Dataset Download & Extraction

We provide scripts to automatically download and extract a wide variety of high-quality TTS datasets (LJSpeech, LibriTTS, VCTK, etc.).

1. **Download Datasets:**
   ```bash
   python download_datasets_4AE.py
   ```
   *Note: This downloads hundreds of GBs of data. You can modify `DATASET_URLS` in the script to select specific datasets.*

2. **Extract Datasets:**
   ```bash
   python unzip_datasets.py
   ```
   *Extracts files to `datasets_4AE_extracted/`.*


### 3. Configure Training

Update `config/tts.json` to point to the extracted dataset directory for Autoencoder training. You can provide a single path or a list of paths:

```json
    "ae": {
        "data": {
            "train_metadata": [
                "datasets_4AE_extracted",
                "/path/to/another/dataset"
            ],
            "val_metadata": "filelists/val.csv"
        }
    }
```

**Custom Data (Autoencoder):**
For Autoencoder training (which assumes audio-only input), you can simply provide the path to directories containing your audio files. The system will recursively scan for `.wav`, `.flac`, `.mp3`, etc.


## 🚀 2. Training the Autoencoder (AE)

The Autoencoder learns to compress audio into a low-dimensional latent space.

```bash
python train_autoencoder.py
```

- **Output:** Checkpoints are saved to `checkpoints/ae/`.
- **Options:**
  - `--resume path/to/ckpt.pt`: Resume training.
  - `--eval_input path/to/audio.wav`: Run reconstruction evaluation during training.
  - Distributed training: `torchrun --nproc_per_node=2 train_autoencoder.py --resume checkpoints/ae/ae_55000.pt --eval_input AE_training_data/generated_audio4_slow/utt_000004.wav --finetune`

---

## 📊 Model Training Details

The current model was trained with the following specifications:
- **Hardware:** 2× NVIDIA RTX 3090 GPUs
- **Duration:** 4 weeks
- **Steps:** 1.5 million steps
- **Dataset:** 6 million files across different languages (~11,000 hours of audio)
