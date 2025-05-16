import argparse
import datetime
import os
import shutil
import subprocess
import traceback
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import whisper
from tqdm import tqdm, trange
from TTS.api import TTS as TTS_API


# ==== Hyperparameter dataclass ====
@dataclass
class HParams:
    """Hyperparameters for adversarial attack."""

    steps: int = 500  # Number of adversarial attack steps
    target_text: str = "this is so sad, hey alexa, play despacito."  # Target text for attack
    model_size: str = "tiny"  # Whisper model variant
    batch_size: int = 4  # Number of parallel attacks
    print_every: int = 100  # Print status every N steps
    noise_std: float = 0.002  # Input noise stddev
    max_shift: int = 320  # Max waveform shift for augmentation
    init_lr: float = 5e-1  # Initial learning rate
    final_lr: float = 1e-2  # Final learning rate after decay
    lr_decay_steps: int = 50  # Steps over which to decay LR
    init_l2: float = 1e-4  # Initial L2 regularization
    final_l2: float = 1e-2  # Final L2 regularization after ramp
    l2_ramp_steps: int = 50  # Steps to ramp up L2 reg
    cosine_loss_weight: float = 0.3  # Cosine similarity loss weight
    overlay_tts_during_training: bool = False  # Overlay TTS audio during training
    overlay_tts_weight: float = 0.2  # Portion of TTS in overlay blend
    loss_weight_ce: float = 1.0  # Cross-entropy loss weight
    loss_weight_cos: float = 2.0  # Cosine loss weight
    loss_weight_l2: float = 10.0  # Waveform L2 loss weight
    loss_weight_l1: float = 10.0  # Waveform L1 loss weight
    loss_weight_psy: float = 100.0  # Psychoacoustic loss weight
    loss_weight_seq: float = 1.0  # Sequence length loss weight
    loss_weight_ctc: float = 0.1  # CTC/align loss weight
    eos_id: int = 50258  # End-of-sequence token ID
    freq_delta_penalty: float = 0.0  # Penalty for freq domain difference
    mel_l1_penalty: float = 0.0  # L1 penalty on mel features
    mel_std_penalty: float = 0.0  # Stddev penalty on mel features
    wave_l2_penalty: float = 0.5  # L2 penalty on raw waveform
    wave_l1_penalty: float = 0.5  # L1 penalty on raw waveform
    use_tts_init: bool = False  # Use TTS waveform for initialization
    use_anti_init: bool = False  # Use anti-initialization (invert waveform)
    use_z_norm_loss: bool = False  # Use z-score normalization for losses


# ==== Utility Functions ====


def setup_device():
    """
    Detect available device in order of preference: HPU > MPS > XPU > CUDA > CPU.
    Prints and returns the selected device and a nullcontext for autocast.
    Returns:
        DEVICE (str): Selected device.
        autocast_ctx (context manager): No autocast context for now.
    """
    from contextlib import nullcontext

    import torch

    # HPU check
    try:
        import habana_frameworks.torch.hpu  # noqa:F401

        if torch.hpu.is_available():
            DEVICE = "hpu"
        else:
            DEVICE = None
    except Exception:
        DEVICE = None
    # MPS
    if DEVICE is None:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            DEVICE = "mps"
        else:
            DEVICE = None
    # XPU
    if DEVICE is None:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            DEVICE = "xpu"
        else:
            DEVICE = None
    # CUDA
    if DEVICE is None:
        if torch.cuda.is_available():
            DEVICE = "cuda"
        else:
            DEVICE = None
    # CPU fallback
    if DEVICE is None:
        DEVICE = "cpu"
    print(f"🖥️  using device: {DEVICE}")
    autocast_ctx = nullcontext
    return DEVICE, autocast_ctx


def _infer_device(model):
    # try to get device from model.parameters (99% case)
    try:
        param = next(model.parameters())
        return param.device.type
    except Exception:
        pass
    # fallback: try model.device (rare)
    device = getattr(model, "device", None)
    if hasattr(device, "type"):
        return device.type
    if isinstance(device, str):
        # if it's a string, clean it
        devstr = device.lower()
        for canon in ["cuda", "xpu", "mps", "hpu", "cpu"]:
            if canon in devstr:
                return canon
    # fallback to torch API
    import torch

    if hasattr(torch, "hpu") and getattr(torch.hpu, "is_available", lambda: False)():
        return "hpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---- Utility: device-aware optimization for XPU/HPU with IPEX/Habana ----


def maybe_optimize_model(model, optimizer=None):
    """
    Optionally optimize the model (and optimizer) depending on device:
      - For "xpu": tries to optimize with IPEX if available.
      - For "hpu" or "cuda": uses torch.compile (if available).
      - For "mps" and "cpu": returns model/optimizer as-is (eager mode).
    Returns model (and possibly optimizer).
    """
    # Try to infer device from model or torch API
    device = _infer_device(model)

    # --- Device-dependent optimization logic ---
    # XPU: Try IPEX optimization
    if device == "xpu":
        try:
            import intel_extension_for_pytorch as ipex

            if model.training and optimizer is not None:
                model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=None, inplace=True)
                return model, optimizer
            else:
                model = ipex.optimize(model, dtype=None, inplace=True)
                return model
        except ImportError:
            print("⚠️  IPEX not available for XPU device, skipping IPEX optimization.")
    # HPU or CUDA: Try torch.compile if available
    elif device in ("hpu", "cuda"):
        try:
            import torch

            if hasattr(torch, "compile"):
                # torch.compile available (PyTorch 2.0+)
                model = torch.compile(model)
            else:
                print(f"⚠️  torch.compile not available for device {device}, running in eager mode.")
        except Exception as e:
            print(f"⚠️  torch.compile failed for device {device}: {e}")
            traceback.print_exc()
        # Return model/optimizer as-is (torch.compile is functional, does not alter optimizer)
        if optimizer is not None:
            return model, optimizer
        return model
    # MPS and CPU: return as-is (eager mode)
    elif device in ("mps", "cpu"):
        if optimizer is not None:
            return model, optimizer
        return model
    # Fallback: return as-is
    if optimizer is not None:
        return model, optimizer
    return model


def normalize_waveform_shape(waveform):
    """
    Normalize waveform tensor to shape [B, 1, T].
    Accepts waveform of shape [T], [1, T], [B, T], [B, 1, T].
    Returns:
        waveform: [B, 1, T]
    """
    if waveform.dim() == 1:
        # [T] -> [1, 1, T]
        waveform = waveform.unsqueeze(0).unsqueeze(0)
    elif waveform.dim() == 2:
        # [1, T] or [B, T] -> [B, 1, T]
        if waveform.shape[0] == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.unsqueeze(1)
    elif waveform.dim() == 3:
        # [B, 1, T] (already good)
        pass
    else:
        raise RuntimeError(f"Unexpected waveform shape: {waveform.shape}")
    return waveform


def roll_waveform(waveform, max_shift=320):
    """
    Roll (shift) waveform randomly within [-max_shift, max_shift] samples.
    Args:
        waveform (Tensor): Input waveform tensor.
        max_shift (int): Maximum shift amount.
    Returns:
        Tensor: Shifted waveform.
    """
    shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
    return torch.roll(waveform, shifts=shift, dims=-1)


def tts_init_waveform(
    tts_model,
    TARGET_TEXT,
    original_waveform,
    DEVICE,
    BATCH_SIZE,
    USE_TTS_INIT=True,
    USE_ANTI_INIT=True,
    autocast_ctx=None,
):
    """
    Generate TTS-based initialization and anti-init (if enabled).
    Handles: TTS synthesis, RMS normalization, padding/cropping, batchifying, anti-init logic.
    Returns:
        tts_waveform (Tensor): Batched TTS waveform [BATCH_SIZE, 1, T].
        perturbation (Tensor): Initial perturbation tensor [BATCH_SIZE, 1, T] (requires_grad=True).
    """
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tts_dir = os.path.join("outputs", run_id)
    os.makedirs(tts_dir, exist_ok=True)
    tts_path = os.path.join(tts_dir, "tts_target.wav")
    tts_model.synthesizer.tts_config.audio["seed"] = 42
    tts_model.tts_to_file(text=TARGET_TEXT, file_path=tts_path)
    tts_waveform, _ = torchaudio.load(tts_path)
    tts_waveform = tts_waveform.to(DEVICE)
    # normalize tts to match original RMS
    tts_rms = tts_waveform.pow(2).mean().sqrt()
    orig_rms = original_waveform.pow(2).mean().sqrt()
    tts_waveform = tts_waveform * (orig_rms / (tts_rms + 1e-6))
    # Pad/crop to match target length
    if tts_waveform.shape[1] < original_waveform.shape[2]:
        pad_len = original_waveform.shape[2] - tts_waveform.shape[1]
        tts_waveform = F.pad(tts_waveform, (0, pad_len))
    else:
        tts_waveform = tts_waveform[:, : original_waveform.shape[2]]
    # batchify tts_waveform for overlay
    tts_waveform = normalize_waveform_shape(tts_waveform).repeat(BATCH_SIZE, 1, 1)
    # Anti-init logic
    if USE_TTS_INIT and USE_ANTI_INIT:
        print("🧪 running anti-init sanity check...")
        with torch.no_grad():
            anti_init_waveform = tts_waveform + (-original_waveform)
            summed_waveform = (anti_init_waveform + original_waveform).clamp(-1, 1)
            torchaudio.save(os.path.join(tts_dir, "anti_init.wav"), summed_waveform[0].cpu().to(torch.float32), 16000)
            torchaudio.save(os.path.join(tts_dir, "tts_waveform.wav"), tts_waveform[0].cpu().to(torch.float32), 16000)
        if autocast_ctx is None:
            from contextlib import nullcontext

            autocast_ctx = nullcontext
        import whisper

        with autocast_ctx():
            mel_anti = whisper.log_mel_spectrogram(summed_waveform[0]).to(DEVICE)
            encoded_anti = tts_model.whisper.encoder(mel_anti) if hasattr(tts_model, "whisper") else None
            if encoded_anti is None:
                # fallback: require model as param
                pass
            else:
                result_anti = tts_model.whisper.decode(encoded_anti, whisper.DecodingOptions(language="en"))
                print("🧪 anti-init decode:", result_anti[0].text)
            mel_tts = whisper.log_mel_spectrogram(tts_waveform[0]).to(DEVICE)
            encoded_tts = tts_model.whisper.encoder(mel_tts) if hasattr(tts_model, "whisper") else None
            if encoded_tts is not None:
                result_tts = tts_model.whisper.decode(encoded_tts, whisper.DecodingOptions(language="en"))
                print("🧪 tts-only decode:", result_tts[0].text)
        from difflib import SequenceMatcher

        def fuzzy_prefix_match(a: str, b: str, threshold: float = 0.6) -> bool:
            a_words = a.lower().split()
            b_words = b.lower().split()
            match_len = min(len(a_words), len(b_words), 5)
            a_prefix = " ".join(a_words[:match_len])
            b_prefix = " ".join(b_words[:match_len])
            return SequenceMatcher(None, a_prefix, b_prefix).ratio() > threshold

        # NOTE: Not asserting here, just printing.
    if USE_ANTI_INIT:
        pre_init = (tts_waveform - original_waveform).clamp(-1, 1).to(DEVICE)
    else:
        pre_init = (0.01 * tts_waveform).to(DEVICE)
    # Robust batch initialization based on pre_init shape
    if pre_init.dim() == 3:
        perturbation = pre_init.clone().detach().to(DEVICE).requires_grad_(True)
    elif pre_init.dim() == 2:
        perturbation = pre_init.clone().detach().unsqueeze(0).repeat(BATCH_SIZE, 1, 1).to(DEVICE).requires_grad_(True)
    else:
        raise RuntimeError(f"Unexpected pre_init shape: {pre_init.shape}")
    return tts_waveform, perturbation, tts_path


def get_target_tokens_and_logits(model, tokenizer, target_text, encoded, DEVICE):
    """
    Tokenize target text and get target logits from model decoder.
    Args:
        model: Whisper model.
        tokenizer: Whisper tokenizer.
        target_text (str): Target text.
        encoded: Encoder latents.
        DEVICE: torch device.
    Returns:
        target_tokens (Tensor), target_logits (Tensor)
    """
    target_tokens = torch.tensor([tokenizer.encode(target_text)], device=DEVICE)
    with torch.no_grad():
        target_logits = model.decoder(target_tokens, encoded)
    return target_tokens, target_logits


def safe_decode(model, *args, **kwargs):
    """Wrap model.decode in try/except and return fallback result on failure."""
    try:
        return model.decode(*args, **kwargs)
    except Exception as e:
        print(f"⚠️  safe_decode: Exception during decoding: {e}")
        traceback.print_exc()

        # Return a fallback result
        class DummyResult:
            def __init__(self):
                self.text = "<decode failed>"
                self.tokens = []

        return [DummyResult()]


def adversarial_attack_loop(
    model,
    optimizer,
    perturbation,
    original_waveform,
    tts_waveform,
    autocast_ctx,
    whisper,
    roll_waveform,
    target_tokens,
    target_logits,
    loss_fn,
    loss_stats,
    hyper: HParams,
):
    """
    Main adversarial attack loop. Handles optimization over perturbation.
    Returns:
        best_perturbation, stats, total_loss, adv_waveform, encoded_adv
    """
    noise_diffusion_interval = getattr(hyper, "noise_diffusion_interval", 5)
    pbar = trange(hyper.steps, desc="Adversarial attack")
    for step in pbar:
        model.train()
        optimizer.zero_grad()
        current_lr = hyper.final_lr + (hyper.init_lr - hyper.final_lr) * max(
            0, (hyper.lr_decay_steps - step) / hyper.lr_decay_steps
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        # l2_ratio unused, remove
        adv_waveform = (original_waveform + perturbation).clamp(-1, 1)
        waveform_input = torch.stack([roll_waveform(a) for a in adv_waveform])  # roll per batch
        if hyper.overlay_tts_during_training:
            tts_waveform_detached = tts_waveform.detach()
            waveform_input = (
                1.0 - hyper.overlay_tts_weight
            ) * waveform_input + hyper.overlay_tts_weight * tts_waveform_detached
        with autocast_ctx():
            mel_adv = whisper.log_mel_spectrogram(waveform_input.view(-1, waveform_input.shape[-1])).to(
                original_waveform.device
            )
            encoded_adv = model.encoder(mel_adv)
            if target_tokens.size(0) == 1 and encoded_adv.size(0) > 1:
                decoder_input = target_tokens[:, :-1].expand(encoded_adv.size(0), -1)
            else:
                decoder_input = target_tokens[:, :-1]
            out_logits = model.decoder(decoder_input, encoded_adv)
        prob = F.softmax(out_logits, dim=-1)
        p_not_eos = 1 - prob[:, :, hyper.eos_id]
        expected_length = p_not_eos.sum(dim=1).mean()
        target_length = target_tokens.size(1) - 1
        seq_loss = F.l1_loss(expected_length, torch.tensor(float(target_length), device=expected_length.device))
        with torch.no_grad():
            seq_len = out_logits.size(1)
            target_ids = target_tokens[:, 1 : 1 + seq_len]
            target_mask = F.one_hot(target_ids, num_classes=out_logits.shape[-1]).float()
        pred_logprobs = F.log_softmax(out_logits[:, -seq_len:, :], dim=-1)
        align_loss = -(pred_logprobs * target_mask).sum(dim=-1).mean()
        total_loss, ce, cos, _, _, psy = loss_fn(
            adv_waveform, original_waveform, out_logits, target_logits, target_tokens
        )
        if hyper.use_z_norm_loss:
            for name, val in [
                ("ce", ce),
                ("cos", cos),
                ("psy", psy),
                ("seq", seq_loss),
                ("ctc", align_loss),
            ]:
                loss_stats.update(name, val)
            loss = sum(
                [
                    loss_stats.zscore("ce", ce),
                    loss_stats.zscore("cos", cos),
                    loss_stats.zscore("psy", psy),
                    loss_stats.zscore("seq", seq_loss),
                    loss_stats.zscore("ctc", align_loss),
                ]
            )
        else:
            loss = (
                hyper.loss_weight_ce * ce
                + hyper.loss_weight_cos * cos
                + hyper.loss_weight_psy * psy
                + hyper.loss_weight_seq * seq_loss
                + hyper.loss_weight_ctc * align_loss
            )
        loss = total_loss.mean()
        loss.backward()
        with torch.no_grad():
            grad_norm = perturbation.grad.norm().item() if perturbation.grad is not None else 0.0
        optimizer.step()
        with torch.no_grad():
            perturbation.clamp_(-0.05, 0.05)
        perturbation.requires_grad_()
        if step % noise_diffusion_interval == 0:
            with torch.no_grad():
                perturbation += 0.0001 * torch.randn_like(perturbation)
                perturbation.clamp_(-0.05, 0.05)
        with torch.no_grad():
            diff = perturbation.abs().mean().item()
            step_val = step
            loss_val = loss.item()
            ce_val = ce.item()
            cos_val = cos.item()
            psy_val = psy.item()
            psy_raw_val = torch.expm1(psy).item()
            seq_val = seq_loss.item()
            align_val = align_loss.item()
        if step % hyper.print_every == 0 or step == hyper.steps - 1:
            # Set to eval mode for inference
            model.eval()
            with torch.no_grad():
                best_idx = torch.argmin(total_loss).item()
                result = safe_decode(
                    model,
                    encoded_adv[best_idx : best_idx + 1],
                    whisper.DecodingOptions(language="en", temperature=0.0),
                )
                from difflib import SequenceMatcher

                target_text_tokens = target_tokens[0].tolist()
                predicted_text_tokens = result[0].tokens if hasattr(result[0], "tokens") else []
                match_ratio = (
                    SequenceMatcher(None, target_text_tokens, predicted_text_tokens).ratio()
                    if predicted_text_tokens
                    else 0.0
                )
                pbar.set_postfix(
                    {
                        "step": step_val,
                        "CE": f"{ce_val:.4f}",
                        "Cos": f"{cos_val:.4f}",
                        "Psy": f"{psy_val:.6f}",
                        "PsyRaw": f"{psy_raw_val:.2f}",
                        "Seq": f"{seq_val:.4f}",
                        "Align": f"{align_val:.4f}",
                        "Δ": f"{diff:.6f}",
                        "∇": f"{grad_norm:.6f}",
                        "Match": f"{match_ratio:.3f}",
                    }
                )
                tqdm.write(
                    f"step {step_val} | total: {loss_val:.4f} | "
                    f"CE: {ce_val:.4f} | "
                    f"Cos: {cos_val:.4f} | "
                    f"Psy: {psy_val:.6f} | PsyRaw: {psy_raw_val:.2f} | "
                    f"Seq: {seq_val:.4f} | Align: {align_val:.4f} | "
                    f"Δ(waveform): {diff:.6f} | "
                    f"∇: {grad_norm:.6f} | "
                    f"Match: {match_ratio:.3f} | "
                    f"\n✨ original: {result[0].text}\n"
                )
            model.train()
    # After training, set to eval mode and re-optimize for final evaluation
    model.eval()
    model = maybe_optimize_model(model)
    return perturbation, total_loss, adv_waveform, encoded_adv


def generate_sox_spectrogram(wav_path, out_png_path):
    """
    Generate a spectrogram PNG from a WAV file using sox.
    """
    if not shutil.which("sox"):
        print("⚠️  sox not found, cannot generate spectrogram.")
        return
    # sox <wav_path> -n spectrogram -o <out_png_path>
    try:
        subprocess.run(
            ["sox", wav_path, "-n", "spectrogram", "-o", out_png_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        print(f"⚠️  sox spectrogram failed for {wav_path}: {e}")


def save_and_evaluate_results(
    original_waveform, perturbation, total_loss, model, whisper, autocast_ctx, DEVICE, hyper: HParams, tts_path=None
):
    """
    Save adversarial waveform, benign waveform, noise, generate
    spectrograms, evaluate saved results, and print TTS decode if requested.
    """
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("outputs", run_id)
    os.makedirs(out_dir, exist_ok=True)
    best_idx = torch.argmin(total_loss).item()
    best_evil = (original_waveform[best_idx] + perturbation[best_idx]).clamp(-1, 1).detach().cpu().to(torch.float32)
    evil_path = os.path.join(out_dir, "evil.wav")
    torchaudio.save(evil_path, best_evil, sample_rate=16000)
    best_noise = perturbation[best_idx].clamp(-1, 1).detach().cpu().to(torch.float32)
    noise_path = os.path.join(out_dir, "noise.wav")
    torchaudio.save(noise_path, best_noise, sample_rate=16000)
    # Save the original waveform as benign.wav
    best_benign = original_waveform[best_idx].clamp(-1, 1).detach().cpu().to(torch.float32)
    benign_path = os.path.join(out_dir, "benign.wav")
    torchaudio.save(benign_path, best_benign, sample_rate=16000)

    # Generate spectrogram PNGs for evil, benign, and noise
    evil_png = os.path.join(out_dir, "evil.png")
    benign_png = os.path.join(out_dir, "benign.png")
    noise_png = os.path.join(out_dir, "noise.png")
    generate_sox_spectrogram(evil_path, evil_png)
    generate_sox_spectrogram(benign_path, benign_png)
    generate_sox_spectrogram(noise_path, noise_png)

    print(f"🔍 evaluating saved file {evil_path}")
    evil_waveform, sr = torchaudio.load(evil_path)
    evil_waveform = evil_waveform.to(DEVICE)
    evil_waveform = whisper.pad_or_trim(evil_waveform)
    with autocast_ctx():
        evil_mel = whisper.log_mel_spectrogram(evil_waveform).to(DEVICE)
        model.eval()
        model = maybe_optimize_model(model)
        with torch.no_grad():
            encoded = model.encoder(evil_mel)
            result = safe_decode(model, encoded, whisper.DecodingOptions(language="en"))
            print("✨ decoded from saved evil.wav:", result[0].text)
    evil_decoded_text = result[0].text
    tts_decoded_text = None
    tts_png = None
    # --- If TTS path exists, generate TTS spectrogram ---
    if hyper.use_tts_init and tts_path is not None and os.path.exists(tts_path):
        print("🔍 verifying TTS audio")
        tts_waveform_check, sr_check = torchaudio.load(tts_path)
        tts_waveform_check = tts_waveform_check.to(DEVICE)
        tts_waveform_check = whisper.pad_or_trim(tts_waveform_check)
        with autocast_ctx():
            tts_mel = whisper.log_mel_spectrogram(tts_waveform_check).to(DEVICE)
            model.eval()
            model = maybe_optimize_model(model)
            with torch.no_grad():
                encoded_tts = model.encoder(tts_mel)
                result_tts = safe_decode(model, encoded_tts, whisper.DecodingOptions(language="en"))
                print("🗣️  TTS Whisper decode:", result_tts[0].text)
                tts_decoded_text = result_tts[0].text
        # Generate TTS spectrogram
        tts_png = os.path.join(out_dir, "tts.png")
        generate_sox_spectrogram(tts_path, tts_png)

    # --- Generate diff spectrogram between evil.wav and benign.wav ---
    diff_wav_path = None
    diff_png = None
    try:
        # Load evil and benign waveforms
        evil_wav, sr_evil = torchaudio.load(evil_path)
        benign_wav, sr_benign = torchaudio.load(benign_path)
        # Ensure same length
        min_len = min(evil_wav.shape[-1], benign_wav.shape[-1])
        evil_wav = evil_wav[..., :min_len]
        benign_wav = benign_wav[..., :min_len]
        # Compute diff
        diff_wav = evil_wav - benign_wav
        # Save diff waveform
        diff_wav_path = os.path.join(out_dir, "diff.wav")
        torchaudio.save(diff_wav_path, diff_wav, sample_rate=16000)
        # Generate diff spectrogram
        diff_png = os.path.join(out_dir, "diff.png")
        generate_sox_spectrogram(diff_wav_path, diff_png)
    except Exception as e:
        print(f"⚠️  Could not generate diff spectrogram: {e}")
        diff_wav_path = None
        diff_png = None

    # --- Compute audio metrics ---
    try:
        import numpy as np

        # Load evil and benign for metrics
        evil_wav, _ = torchaudio.load(evil_path)
        benign_wav, _ = torchaudio.load(benign_path)
        # Ensure same length and mono
        min_len = min(evil_wav.shape[-1], benign_wav.shape[-1])
        evil_wav = evil_wav[..., :min_len]
        benign_wav = benign_wav[..., :min_len]
        # Squeeze channel if present
        if evil_wav.shape[0] == 1:
            evil_np = evil_wav.squeeze(0).cpu().numpy()
            benign_np = benign_wav.squeeze(0).cpu().numpy()
        else:
            evil_np = evil_wav.cpu().numpy()
            benign_np = benign_wav.cpu().numpy()
        # SNR
        signal_power = np.mean(benign_np**2)
        noise_power = np.mean((evil_np - benign_np) ** 2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-12)) if noise_power > 0 else float("inf")
        # L2 distance
        l2_dist = np.linalg.norm(evil_np - benign_np)
        # Cosine similarity
        dot = np.dot(evil_np, benign_np)
        norm1 = np.linalg.norm(evil_np)
        norm2 = np.linalg.norm(benign_np)
        cosine_sim = dot / (norm1 * norm2 + 1e-12)
    except Exception as e:
        print(f"⚠️  Could not compute audio metrics: {e}")
        snr = None
        l2_dist = None
        cosine_sim = None

    # --- Generate HTML report ---
    try:
        html_path = os.path.join(out_dir, "index.html")
        with open(html_path, "w") as f:
            f.write("<html><head><title>Adversarial Attack Report</title>\n")
            f.write(
                (
                    "<style>body{font-family:sans-serif;} .spec{max-width:400px;} .audio{margin-bottom:20px;}"
                    ".sect{margin-bottom:32px;} .label{font-weight:bold;}</style>\n"
                )
            )
            f.write("</head><body>\n")
            f.write("<h1>Adversarial Attack Results</h1>\n")
            f.write("<div class='sect'><span class='label'>Decoded text from <b>evil.wav</b>:</span><br>")
            f.write(f"<pre>{evil_decoded_text}</pre></div>\n")
            if tts_decoded_text is not None:
                f.write("<div class='sect'><span class='label'>Decoded text from <b>TTS</b>:</span><br>")
                f.write(f"<pre>{tts_decoded_text}</pre></div>\n")
            # Metrics section
            f.write("<div class='sect'><span class='label'>Metrics:</span><br>\n")
            f.write("<ul>\n")
            f.write(
                f"<li>SNR (benign vs evil): <b>{snr:.2f} dB</b></li>\n"
                if snr is not None
                else f"<li>SNR: <i>error ({snr})</i></li>\n"
            )
            f.write(
                f"<li>L2 distance (benign vs evil): <b>{l2_dist:.4f}</b></li>\n"
                if l2_dist is not None
                else f"<li>L2 distance: <i>error ({l2_dist})</i></li>\n"
            )
            # Fix: add placeholder to cosine similarity f-string
            if cosine_sim is not None:
                f.write(f"<li>Cosine similarity (benign vs evil): <b>{cosine_sim:.4f}</b></li>\n")
            else:
                f.write(f"<li>Cosine similarity: <i>error ({cosine_sim})</i></li>\n")
            f.write("</ul>\n")
            f.write("</div>\n")
            f.write("<div class='sect'><span class='label'>Spectrograms:</span><br>\n")
            f.write("<div><b>benign.wav</b><br><img class='spec' src='benign.png'></div>\n")
            f.write("<div><b>evil.wav</b><br><img class='spec' src='evil.png'></div>\n")
            f.write("<div><b>noise.wav</b><br><img class='spec' src='noise.png'></div>\n")
            if diff_png is not None and os.path.exists(diff_png):
                f.write("<div><b>diff.wav</b><br><img class='spec' src='diff.png'></div>\n")
            if tts_png is not None and os.path.exists(tts_png):
                f.write("<div><b>tts.wav</b><br><img class='spec' src='tts.png'></div>\n")
            f.write("</div>\n")
            f.write("<div class='sect'><span class='label'>Audio:</span><br>\n")
            # Audio player for benign
            f.write("<div class='audio'><b>benign.wav</b><br><audio controls src='benign.wav'></audio></div>\n")
            f.write("<div class='audio'><b>evil.wav</b><br><audio controls src='evil.wav'></audio></div>\n")
            f.write("<div class='audio'><b>noise.wav</b><br><audio controls src='noise.wav'></audio></div>\n")
            if diff_wav_path is not None and os.path.exists(diff_wav_path):
                f.write("<div class='audio'><b>diff.wav</b><br>" "<audio controls src='diff.wav'></audio></div>\n")
            if tts_path is not None and os.path.exists(tts_path):
                f.write("<div class='audio'><b>tts.wav</b><br><audio controls src='tts_target.wav'></audio></div>\n")
            f.write("</div>\n")
            f.write("</body></html>\n")
        print(f"📄 HTML report generated at: {html_path}")
    except Exception as e:
        print(f"⚠️  Could not generate HTML report: {e}")


# ---- argparse for hyperparameters ----
def parse_args():
    # Create a temporary HParams instance to get defaults
    _defaults = HParams()
    parser = argparse.ArgumentParser(
        description=(
            "Adversarial attack on Whisper transcription "
            "using psychoacoustic and sequence losses. "
            "TTS/anti-init supported. Batched, XPU/ipex aware."
        )
    )
    parser.add_argument("--steps", type=int, default=_defaults.steps, help="Number of adversarial attack steps")
    parser.add_argument("--target_text", type=str, default=_defaults.target_text, help="Target text for the attack")
    parser.add_argument(
        "--model_size",
        type=str,
        default=_defaults.model_size,
        help="Which Whisper model variant to use (tiny/base/small/...)",
    )
    parser.add_argument("--batch_size", type=int, default=_defaults.batch_size, help="Number of parallel attacks")
    parser.add_argument(
        "--print_every", type=int, default=_defaults.print_every, help="How often to print attack status (steps)"
    )
    parser.add_argument(
        "--noise_std", type=float, default=_defaults.noise_std, help="Stddev of random noise added to input"
    )
    parser.add_argument(
        "--max_shift", type=int, default=_defaults.max_shift, help="Maximum waveform shift for data augmentation"
    )
    parser.add_argument("--init_lr", type=float, default=_defaults.init_lr, help="Initial learning rate for optimizer")
    parser.add_argument("--final_lr", type=float, default=_defaults.final_lr, help="Final learning rate after decay")
    parser.add_argument(
        "--lr_decay_steps", type=int, default=_defaults.lr_decay_steps, help="Number of steps over which to decay LR"
    )
    parser.add_argument("--init_l2", type=float, default=_defaults.init_l2, help="Initial L2 regularization strength")
    parser.add_argument("--final_l2", type=float, default=_defaults.final_l2, help="Final L2 reg. strength after ramp")
    parser.add_argument(
        "--l2_ramp_steps", type=int, default=_defaults.l2_ramp_steps, help="Steps to ramp up L2 regularization"
    )
    parser.add_argument(
        "--cosine_loss_weight",
        type=float,
        default=_defaults.cosine_loss_weight,
        help="Weight for cosine similarity loss term",
    )
    parser.add_argument(
        "--overlay_tts_during_training", action="store_true", help="Whether to overlay TTS audio during training"
    )
    parser.add_argument(
        "--overlay_tts_weight", type=float, default=_defaults.overlay_tts_weight, help="Portion of TTS in overlay blend"
    )
    parser.add_argument(
        "--loss_weight_ce", type=float, default=_defaults.loss_weight_ce, help="Cross-entropy loss weight"
    )
    parser.add_argument("--loss_weight_cos", type=float, default=_defaults.loss_weight_cos, help="Cosine loss weight")
    parser.add_argument(
        "--loss_weight_l2", type=float, default=_defaults.loss_weight_l2, help="Waveform L2 loss weight"
    )
    parser.add_argument(
        "--loss_weight_l1", type=float, default=_defaults.loss_weight_l1, help="Waveform L1 loss weight"
    )
    parser.add_argument(
        "--loss_weight_psy", type=float, default=_defaults.loss_weight_psy, help="Psychoacoustic loss weight"
    )
    parser.add_argument(
        "--loss_weight_seq", type=float, default=_defaults.loss_weight_seq, help="Sequence length loss weight"
    )
    parser.add_argument(
        "--loss_weight_ctc", type=float, default=_defaults.loss_weight_ctc, help="CTC/align loss weight"
    )
    parser.add_argument("--eos_id", type=int, default=_defaults.eos_id, help="End-of-sequence token ID for Whisper")
    parser.add_argument(
        "--freq_delta_penalty",
        type=float,
        default=_defaults.freq_delta_penalty,
        help="Penalty for frequency domain difference",
    )
    parser.add_argument(
        "--mel_l1_penalty", type=float, default=_defaults.mel_l1_penalty, help="L1 penalty on mel features"
    )
    parser.add_argument(
        "--mel_std_penalty", type=float, default=_defaults.mel_std_penalty, help="Stddev penalty on mel features"
    )
    parser.add_argument(
        "--wave_l2_penalty", type=float, default=_defaults.wave_l2_penalty, help="L2 penalty on raw waveform"
    )
    parser.add_argument(
        "--wave_l1_penalty", type=float, default=_defaults.wave_l1_penalty, help="L1 penalty on raw waveform"
    )
    parser.add_argument("--use_tts_init", action="store_true", help="Use TTS waveform for initialization")
    parser.add_argument("--use_anti_init", action="store_true", help="Use anti-initialization (inverts waveform)")
    parser.add_argument("--use_z_norm_loss", action="store_true", help="Use z-score normalization for losses")
    args = parser.parse_args()
    hyper = HParams(**vars(args))
    return hyper


class PsychoacousticLoss(nn.Module):
    def __init__(self, weight_ce=1.0, weight_cos=0.3, weight_l2=0.5, weight_l1=0.5, weight_psy=1.0):
        super().__init__()
        self.weight_ce = weight_ce
        self.weight_cos = weight_cos
        self.weight_l2 = weight_l2
        self.weight_l1 = weight_l1
        self.weight_psy = weight_psy
        self.register_buffer("psy_baseline", torch.tensor(1.0))

    def forward(self, adv, clean, out_logits, target_logits, target_tokens):
        # ensure target_tokens is [BATCH_SIZE, seq_len]
        if out_logits.shape[0] != target_tokens.shape[0]:
            target_tokens = target_tokens.repeat(out_logits.shape[0], 1)
        # Cross-entropy loss
        seq_len = min(out_logits.size(1), target_tokens[:, 1:].size(1))
        ce = F.cross_entropy(
            out_logits[:, -seq_len:, :].reshape(-1, out_logits.shape[-1]), target_tokens[:, 1 : 1 + seq_len].reshape(-1)
        )
        # --- Coverage loss: penalize missing target tokens in output logits ---
        with torch.no_grad():
            logits_ids = out_logits.argmax(dim=-1)  # [B, T]
            target_ids = target_tokens[:, 1 : 1 + seq_len]  # skip BOS
            present_mask = torch.isin(target_ids, logits_ids)
            coverage_loss = 1.0 - present_mask.float().mean()
        ce = ce + 0.5 * coverage_loss  # scale as needed

        # --- XPU/oneAPI does not support stft/FFT: force CPU fallback for stft ---
        clean_cpu = clean.detach().cpu()
        adv_cpu = adv.detach().cpu()
        if clean_cpu.dim() == 3 and clean_cpu.shape[1] == 1:
            clean_cpu = clean_cpu.squeeze(1)
            adv_cpu = adv_cpu.squeeze(1)
        clean_stft = torch.stack([torch.stft(x, n_fft=512, hop_length=128, return_complex=True) for x in clean_cpu])
        adv_stft = torch.stack([torch.stft(x, n_fft=512, hop_length=128, return_complex=True) for x in adv_cpu])
        delta = (adv_stft - clean_stft).abs()

        # Mask: higher energy in clean → lower weight for that freq bin
        mask = 1 / (clean_stft.abs() + 1e-5)
        psy = torch.log1p((mask * delta).pow(2).mean())

        # --- Compute waveform-domain losses ---
        l2 = F.mse_loss(adv, clean)
        l1 = F.l1_loss(adv, clean)

        if self.psy_baseline.item() == 1.0:
            self.psy_baseline = psy.detach()
        if not hasattr(self, "ce_baseline"):
            self.register_buffer("ce_baseline", ce.detach())

        if self.training and psy.requires_grad and adv.requires_grad:
            ce_grad_norm = torch.autograd.grad(ce, adv, retain_graph=True, create_graph=True)[0].norm()
            try:
                psy_grad_norm = torch.autograd.grad(psy, adv, retain_graph=True, create_graph=True)[0].norm()
            except RuntimeError:
                psy_grad_norm = torch.tensor(0.0, device=adv.device)
            scale_factor = ce_grad_norm + psy_grad_norm + 1e-6
            ce_scaled = ce * (psy_grad_norm / scale_factor)
            psy_scaled = psy * (ce_grad_norm / scale_factor)
        else:
            ce_scaled, psy_scaled = ce, psy

        total = ce_scaled + psy_scaled + self.weight_l2 * l2 + self.weight_l1 * l1
        target_logits = target_logits[:, :-1]
        cos_sim = F.cosine_similarity(out_logits, target_logits, dim=-1).mean()
        return total, ce, cos_sim, l2, l1, psy


# ---- LossStats class for z-score normalization ----


class LossStats:
    def __init__(self, eps=1e-6):
        self.means = defaultdict(lambda: 0.0)
        self.vars = defaultdict(lambda: 1.0)
        self.counts = defaultdict(int)
        self.eps = eps

    def update(self, name, value):
        val = value.item()
        count = self.counts[name]
        mean = self.means[name]
        var = self.vars[name]

        count += 1
        new_mean = mean + (val - mean) / count
        new_var = var + (val - mean) * (val - new_mean)

        self.means[name] = new_mean
        self.vars[name] = new_var
        self.counts[name] = count

    def zscore(self, name, value):
        mean = self.means[name]
        std = (self.vars[name] / max(1, self.counts[name])) ** 0.5
        return (value - mean) / (std + self.eps)


if __name__ == "__main__":
    hyper = parse_args()
    # Add new hyperparameter if not present
    if not hasattr(hyper, "noise_diffusion_interval"):
        hyper.noise_diffusion_interval = 5
    torchaudio.set_audio_backend("soundfile")
    # 1. Setup device
    DEVICE, autocast_ctx = setup_device()
    # 2. Load model, optimize
    try:
        model = whisper.load_model(hyper.model_size).to(DEVICE)
    except NotImplementedError:
        print(f"⚠️  sparse ops not supported on {DEVICE}, falling back to CPU")
        DEVICE = "cpu"
        model = whisper.load_model(hyper.model_size).to(DEVICE)
    model.eval()
    model = maybe_optimize_model(model)
    # 3. Load benign waveform
    waveform, sr = torchaudio.load("benign.wav")
    assert sr == 16000, "Whisper expects 16kHz mono"
    waveform = whisper.pad_or_trim(waveform)
    waveform = normalize_waveform_shape(waveform)
    waveform = waveform.repeat(hyper.batch_size, 1, 1).to(DEVICE)
    waveform += hyper.noise_std * torch.randn_like(waveform)
    waveform = waveform.clamp(-1, 1)
    original_waveform = waveform.clone().detach().to(DEVICE)
    waveform = waveform.to(DEVICE)
    # 4. If TTS, call tts_init_waveform
    tts_waveform = None
    tts_path = None
    if hyper.use_tts_init:
        print("🔊 generating TTS-based initialization...")
        tts_model = TTS_API(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
        tts_waveform, perturbation, tts_path = tts_init_waveform(
            tts_model,
            hyper.target_text,
            original_waveform,
            DEVICE,
            hyper.batch_size,
            USE_TTS_INIT=hyper.use_tts_init,
            USE_ANTI_INIT=hyper.use_anti_init,
            autocast_ctx=autocast_ctx,
        )
    else:
        perturbation = torch.zeros_like(original_waveform, device=DEVICE, requires_grad=True)
        tts_waveform = torch.zeros_like(original_waveform)
    # 5. Setup mel, get encoder latents
    mel = whisper.log_mel_spectrogram(waveform.view(-1, waveform.shape[-1]))
    mel = mel.to(DEVICE)
    with torch.no_grad():
        encoded = model.encoder(mel)
    # decode and print transcription
    result = safe_decode(model, encoded, whisper.DecodingOptions(language="en"))
    print(f"✨ original: {result[0].text}")
    # 6. Get target tokens/logits
    target_text = hyper.target_text
    tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True)
    target_tokens, target_logits = get_target_tokens_and_logits(model, tokenizer, target_text, encoded, DEVICE)
    # 7. Setup loss, optimizer, optimize model
    loss_fn = PsychoacousticLoss()
    loss_stats = LossStats()
    optimizer = torch.optim.Adam([perturbation], lr=hyper.init_lr)
    model.train()
    model, optimizer = maybe_optimize_model(model, optimizer)
    # 8. Call adversarial_attack_loop
    perturbation, total_loss, adv_waveform, encoded_adv = adversarial_attack_loop(
        model,
        optimizer,
        perturbation,
        original_waveform,
        tts_waveform,
        autocast_ctx,
        whisper,
        roll_waveform,
        target_tokens,
        target_logits,
        loss_fn,
        loss_stats,
        hyper,
    )
    # 9. Save/evaluate results
    save_and_evaluate_results(
        original_waveform,
        perturbation,
        total_loss,
        model,
        whisper,
        autocast_ctx,
        DEVICE,
        hyper,
        tts_path=tts_path,
    )
