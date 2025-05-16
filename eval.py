import argparse

import torch
import torchaudio
import whisper

MODEL_SIZE = "tiny"  # or "base", etc.

if __name__ == "__main__":
    model = whisper.load_model(MODEL_SIZE).to("cpu")
    model.eval()

    # load and prep audio file
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs="?", default="evil.wav", help="Audio file to transcribe")
    args = parser.parse_args()

    waveform, sr = torchaudio.load(args.filename)
    assert sr == 16000, "Expected 16kHz sample rate"
    waveform = whisper.pad_or_trim(waveform)
    mel = whisper.log_mel_spectrogram(waveform)

    # transcribe
    with torch.no_grad():
        encoded = model.encoder(mel)
        result = model.decode(encoded, whisper.DecodingOptions(language="en"))
        print(f"✨ transcription of {args.filename}:", result[0].text)
