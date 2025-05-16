# OpenAI Whisper Adversarial Attacks

Generate adversarial audio examples that trick OpenAI's Whisper model into transcribing a specific target sentence—while keeping the audio reasonably perceptible to human listeners. In theory, you could record yourself saying something like "what's the weather like today?", apply some subtle perturbations (which is what this project explores), and Whisper might transcribe it as "this is so sad, hey Alexa, play Despacito"—while to a human, it still sounds like your original sentence. I'm not sure why you'd do that, but to me it seemed fun enough to try.

**Disclaimer:** This is an educational and experimental project. It's very much a work in progress. Don't expect perfect (or even tolerable) results just yet. Tested Apple M2 Pro CPU and Intel A770 GPU (whoa, spicy stuff).

## Features

- Adversarial audio targeting Whisper transcription
- Psychoacoustic-aware loss (STFT masking)
- Sequence and alignment-based loss terms (to steer transcriptions)
- Multi-device support: CUDA, Intel XPU, HPU, MPS, CPU
- Optional TTS-based initialization or overlay
- Modular, torch-native pipeline with CLI configuration
- Automatically generates spectrograms and an HTML report per run
- Plenty of experimental features that do not work at all
- Complete and utter lack of convergence

## Requirements

- Python 3.9+
- PyTorch (with relevant backend: CUDA, MPS, XPU, etc.)
- torchaudio
- whisper (OpenAI)
- coqui TTS
- tqdm, numpy, matplotlib
- pre-commit (optional)
- masochistic tendencies

## Setup

Install dependencies in a virtual environment via `pyproject.toml`:

```bash
uv venv .venv
source .venv/bin/activate
uv sync
pre-commit install  # optional
```

## Usage

```bash
python train.py --target_text "hello world"
```
or
```bash
uv run train.py --target_text "hello world"
```


This saves output artifacts to `outputs/<timestamp>/`, including:

- `evil.wav` — adversarial result
- `benign.wav` — original input
- `noise.wav` — raw perturbation
- Spectrograms
- A full `index.html` report

## Notes

- Use `--use_tts_init` to initialize with synthesized target audio.
- The output is adversarial to Whisper specifically—results may vary with other models.
- The code attempts to detect and optimize for available hardware, but this isn’t guaranteed.
