# Training Documentation

## 🗂️ 1. Data Preparation

The Autoencoder is designed to be trained on large-scale, high-quality audio datasets. Since it only requires audio (no transcripts), you can use any collection of speech or general audio.

### Recommended Datasets

For high-quality speech reconstruction, we recommend using a mix of the following datasets:

- **LibriTTS / LibriTTS-R:** Large-scale corpus of English speech (approx. 585 hours). The "-R" version is restored for higher quality.
- **LJSpeech:** A standard single-speaker English dataset (approx. 24 hours).
- **VCTK:** Multi-speaker English dataset with various accents (approx. 44 hours).
- **Hi-Fi TTS:** High-fidelity multi-speaker English dataset (approx. 291 hours).
- **Common Voice:** Massive multi-language dataset (thousands of hours).

### 2. Configure Training

Update `config/tts.json` to point to your dataset directories. You can provide a single path or a list of paths. The system will recursively scan these directories for audio files (`.wav`, `.flac`, `.mp3`, etc.).

```json
    "ae": {
        "data": {
            "train_metadata": [
                "/path/to/LibriTTS",
                "/path/to/hifi-tts",
                "/path/to/custom_audio"
            ],
            "val_metadata": "/path/to/validation_audio"
        }
    }
```

**Custom Data:**
You can simply provide the path to any directory containing your audio files. The training script handles the recursive discovery of supported audio formats.


## 🚀 3. Training the Autoencoder (AE)

The Autoencoder learns to compress audio into a low-dimensional latent space.

```bash
uv run train_autoencoder.py
```

- **Output:** Checkpoints are saved to `checkpoints/ae/`.
- **Options:**
  - `--resume path/to/ckpt.pt`: Resume training from a specific checkpoint.
  - `--eval_input path/to/audio.wav`: Run reconstruction evaluation on a specific file during training.
  - **Distributed Training:** For faster training on multiple GPUs, use `torchrun` via `uv run`:
    ```bash
    uv run torchrun --nproc_per_node=2 train_autoencoder.py --resume checkpoints/ae/ae_latest.pt
    ```

---

## 📊 Model Training Details

The current pretrained model was trained with the following specifications:
- **Hardware:** 2× NVIDIA RTX 3090 GPUs
- **Duration:** 4 weeks
- **Steps:** 1.5 million steps
- **Dataset:** 6 million files across different languages (~11,000 hours of audio)
