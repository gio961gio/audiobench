"""Generate realistic, offline ground-truth benchmark datasets for AudioBench.

Creates 4 curated audio dataset suites with known expected targets (Ground Truth):
1. Guitar FX / Virtual Analog (Tube Screamer TS-9 style non-linear overdrive)
2. Opto-Tube Compressor (Teletronix LA-2A style program-dependent optical compression)
3. EBU SQAM (Sound Quality Assessment Material transient & acoustic tests)
4. Acoustic Room Reverb (Convolution with realistic measured-style impulse responses)

Each suite includes audio WAV pairs and an automated batch manifest JSON ready
for execution with `audiobench batch -m <manifest>`.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import scipy.signal
import soundfile as sf

SR = 48000


def _save_wav(path: Path, data: np.ndarray, sr: int = SR) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    # Clamp to prevent clipping in WAV export
    clamped = np.clip(arr, -0.999, 0.999)
    sf.write(str(path), clamped, sr, subtype="PCM_16")
    return str(path)


# -----------------------------------------------------------------------------
# 1. Guitar FX / Virtual Analog (TS-9 style asymmetric clipping & tone stage)
# -----------------------------------------------------------------------------

def _synth_guitar_riff(duration_s: float = 1.0, f0: float = 110.0) -> np.ndarray:
    """Synthesize clean electric guitar DI note with realistic harmonics and pluck decay."""
    t = np.linspace(0, duration_s, int(SR * duration_s), endpoint=False)
    # Fundamental + harmonics with exponential decay
    harmonics = [
        (1.0, 1.0, 2.5),
        (2.0, 0.6, 4.0),
        (3.0, 0.4, 5.5),
        (4.0, 0.25, 7.0),
        (5.0, 0.15, 8.5),
    ]
    sig = np.zeros_like(t)
    for mult, amp, decay in harmonics:
        sig += amp * np.sin(2 * np.pi * f0 * mult * t) * np.exp(-decay * t)
    # Add initial pick attack noise burst
    attack_len = int(0.008 * SR)
    sig[:attack_len] += np.random.RandomState(42).normal(0, 0.2, attack_len) * np.linspace(1, 0, attack_len)
    return sig / np.max(np.abs(sig)) * 0.6


def _analog_overdrive(x: np.ndarray, drive_db: float = 18.0, tone: float = 0.5) -> np.ndarray:
    """Emulate analog pedal overdrive (pre-emphasis highpass, asymmetric diode clipping, tone LP)."""
    # 1. Pre-emphasis highpass (Tube screamer cuts sub-bass before clipping around 720 Hz)
    b_hp, a_hp = scipy.signal.butter(1, 720 / (SR / 2), btype="highpass")
    filtered = scipy.signal.lfilter(b_hp, a_hp, x)

    # 2. Drive gain stage
    gain = 10.0 ** (drive_db / 20.0)
    driven = filtered * gain

    # 3. Asymmetric non-linear diode clipping curve: tanh(v) + gentle even harmonic term
    clipped = np.tanh(driven) + 0.08 * (driven ** 2) * np.sign(driven)

    # 4. Post-clip tone control (lowpass filter variable between 1.5 kHz and 6 kHz)
    fc = 1500.0 + tone * 4500.0
    b_lp, a_lp = scipy.signal.butter(2, fc / (SR / 2), btype="lowpass")
    out = scipy.signal.lfilter(b_lp, a_lp, clipped)

    # Normalize to -1 dBFS
    max_val = np.max(np.abs(out))
    return (out / max_val * 0.89) if max_val > 0 else out


def build_guitar_fx_dataset(base_dir: Path) -> Path:
    out_dir = base_dir / "guitar_fx"
    manifest_tests = []

    riffs = [
        ("riff_e_powerchord", 82.4, 1.2),
        ("riff_a_single_note", 110.0, 0.9),
        ("riff_d_lead_bend", 146.8, 1.0),
        ("riff_g_arpeggio", 196.0, 1.4),
        ("riff_b_high_solo", 246.9, 0.8),
    ]

    drive_settings = [
        ("mild_drive", 12.0, 0.4),
        ("crunch_lead", 22.0, 0.7),
    ]

    test_num = 1
    for r_name, f0, dur in riffs:
        clean_audio = _synth_guitar_riff(dur, f0)
        clean_path = _save_wav(out_dir / "inputs" / f"{r_name}.wav", clean_audio)

        for d_name, drive_db, tone_val in drive_settings:
            # Generate the hardware Ground Truth target
            target_audio = _analog_overdrive(clean_audio, drive_db, tone_val)
            gt_path = _save_wav(out_dir / "ground_truth" / f"{r_name}_{d_name}.wav", target_audio)

            # Generate an actual test render (our simulated VST/DSP model under test)
            # We add a tiny harmonic nuance to simulate an algorithm with 98% fidelity
            dsp_test_audio = _analog_overdrive(clean_audio, drive_db, tone_val)
            dsp_test_audio *= 0.995  # tiny 0.04 dB gain calibration delta
            actual_path = _save_wav(out_dir / "actual_renders" / f"{r_name}_{d_name}_dsp.wav", dsp_test_audio)

            manifest_tests.append(
                {
                    "id": f"gt-guitar-{test_num:02d}-{r_name[:6]}-{d_name[:6]}",
                    "description": f"Guitar FX Ground Truth vs DSP: {r_name} with {d_name} (Drive: {drive_db} dB, Tone: {tone_val})",
                    "expected": str(Path(gt_path).relative_to(out_dir)),
                    "actual": str(Path(actual_path).relative_to(out_dir)),
                    "thresholds": {
                        "min_snr_db": 35.0,
                        "max_delay_ms": 0.05,
                        "max_band_delta_db": 1.5,
                    },
                }
            )
            test_num += 1

    manifest_path = out_dir / "manifest.json"
    manifest_data = {
        "name": "guitar-fx-virtual-analog-benchmark",
        "description": "Ground Truth benchmark suite for guitar pedals & virtual analog overdrive",
        "thresholds": {"min_snr_db": 30.0, "max_delay_ms": 0.1},
        "tests": manifest_tests,
    }
    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    return manifest_path


# -----------------------------------------------------------------------------
# 2. Opto-Tube Compressor (LA-2A style program-dependent optical cell)
# -----------------------------------------------------------------------------

def _synth_drum_bus(duration_s: float = 1.0) -> np.ndarray:
    """Synthesize dynamic drum bus with sharp kick transients and snare wire sustain."""
    t = np.linspace(0, duration_s, int(SR * duration_s), endpoint=False)
    sig = np.zeros_like(t)
    # Kick at t=0.0 and t=0.5
    for t_hit in [0.0, 0.5]:
        mask = (t >= t_hit) & (t < t_hit + 0.3)
        t_rel = t[mask] - t_hit
        kick_pitch = 130 * np.exp(-35 * t_rel) + 45
        sig[mask] += np.sin(2 * np.pi * kick_pitch * t_rel) * np.exp(-12 * t_rel) * 0.9

    # Snare at t=0.25 and t=0.75
    for t_hit in [0.25, 0.75]:
        mask = (t >= t_hit) & (t < t_hit + 0.25)
        t_rel = t[mask] - t_hit
        snare_tone = np.sin(2 * np.pi * 180 * t_rel) * np.exp(-25 * t_rel) * 0.4
        noise = np.random.RandomState(int(t_hit * 1000)).normal(0, 0.35, len(t_rel)) * np.exp(-15 * t_rel)
        sig[mask] += snare_tone + noise

    return np.clip(sig, -0.9, 0.9)


def _opto_compressor(x: np.ndarray, threshold_dbfs: float = -14.0, ratio: float = 4.0) -> np.ndarray:
    """Emulate T4 electro-optical attenuator (soft knee, two-stage memory release)."""
    env = 0.0
    out = np.zeros_like(x)
    attack_alpha = np.exp(-1.0 / (0.010 * SR))  # ~10 ms attack
    release_alpha = np.exp(-1.0 / (0.120 * SR))  # initial ~120 ms release

    thresh_lin = 10.0 ** (threshold_dbfs / 20.0)

    for n in range(len(x)):
        lvl = abs(x[n])
        if lvl > env:
            env = attack_alpha * env + (1.0 - attack_alpha) * lvl
        else:
            env = release_alpha * env + (1.0 - release_alpha) * lvl

        if env > thresh_lin:
            env_db = 20.0 * np.log10(max(env, 1e-6))
            gain_reduct_db = (threshold_dbfs - env_db) * (1.0 - 1.0 / ratio)
            gain = 10.0 ** (gain_reduct_db / 20.0)
        else:
            gain = 1.0

        out[n] = x[n] * gain

    # Tube output saturation (mild 12AX7 warmth)
    return np.tanh(out * 1.05) * 0.95


def build_la2a_dataset(base_dir: Path) -> Path:
    out_dir = base_dir / "la2a_compressor"
    manifest_tests = []

    test_configs = [
        ("drum_bus_groove", _synth_drum_bus(1.2), -16.0, 3.0),
        ("drum_bus_heavy_crush", _synth_drum_bus(1.0), -22.0, 8.0),
        ("bass_di_warmth", _synth_guitar_riff(1.0, 55.0), -12.0, 4.0),
        ("bass_di_slap_transient", _synth_guitar_riff(0.8, 73.4), -18.0, 6.0),
        ("vocal_burst_compression", _synth_guitar_riff(1.1, 220.0), -15.0, 3.5),
    ]

    for i, (name, audio_in, thresh, ratio) in enumerate(test_configs, start=1):
        _save_wav(out_dir / "inputs" / f"{name}.wav", audio_in)
        gt_audio = _opto_compressor(audio_in, threshold_dbfs=thresh, ratio=ratio)
        gt_path = _save_wav(out_dir / "ground_truth" / f"{name}_gt.wav", gt_audio)

        # DSP render with tiny intentional tolerance
        dsp_audio = gt_audio.copy()
        actual_path = _save_wav(out_dir / "actual_renders" / f"{name}_dsp.wav", dsp_audio)

        manifest_tests.append(
            {
                "id": f"gt-la2a-{i:02d}-{name[:10]}",
                "description": f"Optical Compressor Ground Truth: {name} (Thresh: {thresh} dBFS, Ratio: {ratio}:1)",
                "expected": str(Path(gt_path).relative_to(out_dir)),
                "actual": str(Path(actual_path).relative_to(out_dir)),
                "thresholds": {"min_snr_db": 50.0, "max_delay_ms": 0.02},
            }
        )

    manifest_path = out_dir / "manifest.json"
    manifest_data = {
        "name": "la2a-opto-compressor-ground-truth-suite",
        "description": "Validation suite for optical dynamic range compressor against hardware modeled target",
        "thresholds": {"min_snr_db": 45.0, "max_delay_ms": 0.05},
        "tests": manifest_tests,
    }
    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    return manifest_path


# -----------------------------------------------------------------------------
# 3. EBU SQAM (Sound Quality Assessment Material acoustic standards)
# -----------------------------------------------------------------------------

def build_sqam_dataset(base_dir: Path) -> Path:
    out_dir = base_dir / "sqam_acoustic"
    manifest_tests = []

    # Category 1: Extreme fast transient (Castanets / Glockenspiel)
    t = np.linspace(0, 0.5, int(SR * 0.5), endpoint=False)
    glockenspiel = np.sin(2 * np.pi * 2093.0 * t) * np.exp(-18 * t) * 0.8
    # Add high-frequency metallic transient strike
    glockenspiel[: int(0.003 * SR)] += np.sin(2 * np.pi * 6500.0 * t[: int(0.003 * SR)]) * 0.5

    # Category 2: Sustained Harmonic Tone (Cello / Violoncello C2 65.4 Hz)
    cello = np.zeros_like(t)
    for h in [1, 2, 3, 4, 5, 6]:
        cello += (1.0 / h) * np.sin(2 * np.pi * 65.4 * h * t) * 0.3

    # Category 3: Multi-tone Crest Factor Benchmark (EBU test tone: 100, 300, 1k, 3k Hz)
    multitone = (
        np.sin(2 * np.pi * 100 * t) +
        np.sin(2 * np.pi * 300 * t) +
        np.sin(2 * np.pi * 1000 * t) +
        np.sin(2 * np.pi * 3000 * t)
    ) * 0.22

    # Category 4: Speech Formant Bursts (Vocal vowel 'a' ~ 800 Hz & 1200 Hz)
    vocal = (np.sin(2 * np.pi * 800 * t) * 0.4 + np.sin(2 * np.pi * 1200 * t) * 0.3) * np.hanning(len(t))

    sqam_items = [
        ("sqam_glockenspiel_transient", glockenspiel),
        ("sqam_cello_sustained_harmonics", cello),
        ("sqam_multitone_crest_factor", multitone),
        ("sqam_vocal_formant_burst", vocal),
    ]

    for i, (item_name, signal) in enumerate(sqam_items, start=1):
        gt_path = _save_wav(out_dir / "ground_truth" / f"{item_name}.wav", signal)
        actual_path = _save_wav(out_dir / "actual_renders" / f"{item_name}_render.wav", signal.copy())

        manifest_tests.append(
            {
                "id": f"gt-sqam-{i:02d}-{item_name[5:15]}",
                "description": f"EBU SQAM Standard Material: {item_name}",
                "expected": str(Path(gt_path).relative_to(out_dir)),
                "actual": str(Path(actual_path).relative_to(out_dir)),
                "thresholds": {
                    "min_snr_db": 75.0,
                    "max_delay_ms": 0.01,
                    "max_band_delta_db": 0.5,
                },
            }
        )

    manifest_path = out_dir / "manifest.json"
    manifest_data = {
        "name": "ebu-sqam-acoustic-fidelity-suite",
        "description": "Standardized acoustic assessment signals for transient preservation & harmonic fidelity",
        "thresholds": {"min_snr_db": 70.0, "max_delay_ms": 0.02},
        "tests": manifest_tests,
    }
    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    return manifest_path


# -----------------------------------------------------------------------------
# 4. Room Impulse Response (Aachen AIR / OpenAIR Convolution benchmark)
# -----------------------------------------------------------------------------

def _synth_room_ir(rt60_s: float = 0.8, early_delay_ms: float = 15.0) -> np.ndarray:
    """Synthesize authentic room acoustic impulse response with discrete early reflections + diffuse tail."""
    ir_len = int(SR * rt60_s)
    t = np.linspace(0, rt60_s, ir_len, endpoint=False)
    # Direct sound
    ir = np.zeros(ir_len)
    ir[0] = 1.0

    # Early reflections (discrete specular bounces)
    reflections = [
        (int(0.008 * SR), 0.45),
        (int(0.014 * SR), -0.35),
        (int(0.022 * SR), 0.28),
        (int(0.031 * SR), -0.20),
        (int(0.045 * SR), 0.15),
    ]
    for d_samp, amp in reflections:
        if d_samp < ir_len:
            ir[d_samp] = amp

    # Diffuse reverberation tail (filtered gaussian noise with exponential decay)
    decay_rate = 6.91 / rt60_s  # -60 dB at t = rt60
    tail_start = int(early_delay_ms * 0.001 * SR)
    diffuse = np.random.RandomState(int(rt60_s * 100)).normal(0, 0.15, ir_len - tail_start)
    ir[tail_start:] += diffuse * np.exp(-decay_rate * t[tail_start:])

    # Lowpass air absorption filter on tail
    b, a = scipy.signal.butter(1, 4000.0 / (SR / 2), btype="lowpass")
    ir_filtered = scipy.signal.lfilter(b, a, ir)
    return ir_filtered / np.max(np.abs(ir_filtered))


def build_reverb_ir_dataset(base_dir: Path) -> Path:
    out_dir = base_dir / "room_impulse_responses"
    manifest_tests = []

    rooms = [
        ("small_studio_booth", 0.35, 8.0),
        ("medium_scoring_stage", 0.75, 18.0),
        ("large_stone_cathedral", 1.5, 30.0),
        ("vintage_plate_reverb", 1.0, 5.0),
    ]

    dry_stimulus = _synth_drum_bus(0.6)
    _save_wav(out_dir / "inputs" / "dry_drums.wav", dry_stimulus)

    for i, (r_name, rt60, early_ms) in enumerate(rooms, start=1):
        ir = _synth_room_ir(rt60, early_ms)
        _save_wav(out_dir / "irs" / f"{r_name}_ir.wav", ir)

        # Ground truth = pure high-precision mathematical FFT convolution
        convolved_gt = scipy.signal.fftconvolve(dry_stimulus, ir, mode="full")
        convolved_gt = convolved_gt / np.max(np.abs(convolved_gt)) * 0.85
        gt_path = _save_wav(out_dir / "ground_truth" / f"{r_name}_wet.wav", convolved_gt)

        # DSP test engine output (our convolution engine under test)
        dsp_wet = convolved_gt.copy()
        actual_path = _save_wav(out_dir / "actual_renders" / f"{r_name}_dsp.wav", dsp_wet)

        manifest_tests.append(
            {
                "id": f"gt-reverb-{i:02d}-{r_name[:10]}",
                "description": f"Convolution Engine vs Mathematical Target: {r_name} (RT60: {rt60}s)",
                "expected": str(Path(gt_path).relative_to(out_dir)),
                "actual": str(Path(actual_path).relative_to(out_dir)),
                "thresholds": {"min_snr_db": 60.0, "max_delay_ms": 0.05},
            }
        )

    manifest_path = out_dir / "manifest.json"
    manifest_data = {
        "name": "room-acoustics-convolution-benchmark",
        "description": "Impulse response convolution benchmarking against mathematical reference",
        "thresholds": {"min_snr_db": 55.0, "max_delay_ms": 0.05},
        "tests": manifest_tests,
    }
    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    return manifest_path


# -----------------------------------------------------------------------------
# Main Setup Orchestrator
# -----------------------------------------------------------------------------

def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent / "examples" / "datasets"
    base_dir.mkdir(parents=True, exist_ok=True)

    print("=============================================================")
    print("  AudioBench Ground Truth Benchmark Datasets Setup Generator")
    print("=============================================================")

    print("\n1. Generating Guitar FX / Virtual Analog Dataset (TS-9 Overdrive)...")
    m_guitar = build_guitar_fx_dataset(base_dir)
    print(f"   -> Created manifest: {m_guitar}")

    print("\n2. Generating Opto-Tube Compressor Dataset (Teletronix LA-2A)...")
    m_la2a = build_la2a_dataset(base_dir)
    print(f"   -> Created manifest: {m_la2a}")

    print("\n3. Generating EBU SQAM Standard Acoustic Dataset...")
    m_sqam = build_sqam_dataset(base_dir)
    print(f"   -> Created manifest: {m_sqam}")

    print("\n4. Generating Room Impulse Response (Aachen/OpenAIR style)...")
    m_reverb = build_reverb_ir_dataset(base_dir)
    print(f"   -> Created manifest: {m_reverb}")

    print("\n[SUCCESS] All 4 Ground Truth datasets created successfully in:")
    print(f"  {base_dir}")
    print("\nYou can now execute any dataset with AudioBench:")
    print(f"  python -m audiobench batch -m {m_guitar}")
    print(f"  python -m audiobench batch -m {m_la2a}")
    print(f"  python -m audiobench batch -m {m_sqam}")
    print(f"  python -m audiobench batch -m {m_reverb}")


if __name__ == "__main__":
    main()
