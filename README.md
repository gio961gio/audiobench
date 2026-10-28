# AudioBench

**A lightweight offline regression test bench for audio renders and DSP algorithms.**

AudioBench compares an expected/reference render with an actual render, computes signal-level metrics, applies configurable quality gates, and produces a self-contained HTML report plus machine-readable JSON. It is designed as a small portfolio project that can be extended into automated DSP/plugin QA.

> **Current scope:** AudioBench analyzes files that have already been rendered to WAV. It does not yet load or execute VST3/AU plugin binaries, control a DAW, resample inputs, or claim standards-compliant true-peak/THD measurements.

## Features

- Deterministic test stimuli: impulse, 1 kHz sine, logarithmic sweep, seeded white noise, and silence.
- Reference-vs-actual comparison without silently resampling files.
- Duration, peak and RMS levels, RMS error, SNR, relative error, sample correlation, maximum sample error, mean spectral-magnitude delta, gain-normalized spectral-shape error, and an estimated sample delay.
- Independent quality gates for fidelity, latency, and duration.
- Batch test suite execution from a JSON manifest with fault isolation, relative path resolution, and per-test threshold overrides.
- Standalone HTML reports (per-test and suite summary dashboard) with waveform overview and embedded JSON results.
- CLI exit codes suitable for local scripts and CI: `0` for PASS/WARN, `1` for FAIL/ERROR, `2` for invalid input/runtime/manifest error. Use `--fail-on-warn` to make warnings fail a CI job.
- Comprehensive unit tests for deterministic signals, comparison metrics, delay sign, batch manifests, error isolation, and CLI exit codes.

## Quick start

Python 3.10 or newer is recommended.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

# 2. Install the project and test dependencies
python -m pip install -e ".[dev]"

# 3. Generate deterministic WAV stimuli
audiobench generate --output-dir examples/test_signals --sample-rate 48000 --duration 1

# 4. Compare two renders (replace these paths with your own files)
audiobench compare \
  --expected examples/reference.wav \
  --actual examples/render.wav \
  --report reports/comparison.html \
  --json reports/comparison.json

# 5. Run a full regression suite from a JSON manifest
audiobench batch \
  --manifest examples/demo/demo_manifest.json \
  --output-dir reports/batch

# 6. Run the tests
pytest
```

The CLI is also available without installing the console script:

```bash
python -m audiobench --help
python -m audiobench generate --help
python -m audiobench compare --help
python -m audiobench batch --help
```

## Batch regression testing (JSON manifest)

Run an entire suite of regression comparisons with a single command:

```bash
audiobench batch \
  --manifest path/to/manifest.json \
  --output-dir reports/batch
```

### Manifest schema

```json
{
  "name": "dsp-regression-suite",
  "thresholds": {
    "min_snr_db": 60.0,
    "max_delay_ms": 0.5,
    "max_duration_delta_ms": 1.0,
    "max_error_dbfs": -80.0
  },
  "tests": [
    {
      "id": "filter-bypass",
      "description": "Bypass processing must preserve reference audio",
      "expected": "renders/ref_bypass.wav",
      "actual": "renders/test_bypass.wav"
    },
    {
      "id": "delay-compensation",
      "description": "Lookahead processing with dedicated latency tolerance",
      "expected": "renders/ref_lookahead.wav",
      "actual": "renders/test_lookahead.wav",
      "thresholds": {
        "max_delay_ms": 5.0
      }
    }
  ]
}
```

* **Relative paths:** `expected` and `actual` paths are resolved relative to the directory containing the manifest file.
* **Threshold overrides:** Individual test cases inherit suite-level thresholds and can selectively override specific limits.
* **Fault isolation:** An unreadable, corrupted, or incompatible WAV in one test case is recorded as `ERROR` without aborting the rest of the suite.
* **Dashboard output:** Generates `<output-dir>/summary.html` and `<output-dir>/summary.json`, alongside individual HTML and JSON reports for each test case.

## Example workflow

1. Choose one algorithm or processing chain you can render offline.
2. Generate or prepare a known input stimulus (for example `white_noise.wav` or `sweep.wav`).
3. Render the stimulus through a known-good/reference version and save that output as `reference.wav`.
4. Render the same stimulus through the code/version you are testing and save it as `render.wav`.
5. Run `audiobench compare ...` and inspect both the console result and the HTML report.
6. Keep the JSON report as a CI artifact or compare it across commits.

For an exact regression test, the expected and actual render should be produced with the same sample rate, channel layout, input, plugin settings, and render range. If the algorithm is intentionally changed, update the reference only after reviewing the differences.

## Metrics and interpretation

| Metric | What it tells you | Important caveat |
|---|---|---|
| Duration delta | Whether frame counts agree | A duration mismatch can also reflect a different render range. |
| Sample peak / RMS | Absolute output level | Sample peak is not the same as oversampled true peak. |
| RMS error / SNR | How far samples differ from the reference | A one-sample delay can produce a large error even when the sound is otherwise similar. |
| Relative error | Error RMS relative to reference RMS | Undefined for a silent reference. |
| Correlation | Similarity of sample shapes | Not a perceptual quality score. |
| Spectral magnitude delta | Mean absolute difference between magnitude spectra in dB | A diagnostic summary, not a calibrated frequency-response measurement. |
| Spectral-shape error | Difference after normalizing spectra for overall RMS level | Helps distinguish broad level changes from shape changes, but depends on the test signal. |
| Estimated delay | Approximate offset found by cross-correlation | Periodic inputs (especially pure sine waves) can make latency ambiguous. Prefer impulse or broadband/noise stimuli. |
| SNR after estimated alignment | Difference after compensating for the estimated delay | Diagnostic only; quality gates still use the unaligned render so a latency regression is not hidden. |

No single number describes audio quality. Use metrics as regression indicators and pair them with listening and domain-specific checks.

## Quality gates

Default gates are deliberately configurable:

- **Fidelity:** SNR must be at least `60 dB`, unless the signals are sample-identical. For a silent/near-silent reference where SNR is undefined, the absolute RMS error must be at or below `-80 dBFS`.
- **Latency:** absolute estimated delay must be at or below `0.5 ms`.
- **Duration:** absolute duration difference must be at or below `1 ms`.

Example with stricter tolerances:

```bash
audiobench compare \
  --expected reference.wav \
  --actual render.wav \
  --min-snr-db 80 \
  --max-delay-ms 0.1 \
  --max-duration-delta-ms 0 \
  --fail-on-warn
```

Thresholds depend on the system under test. A nonlinear effect, analog-modeling plugin, random modulation, or intentionally changed DSP behavior may need a different protocol. Document the test setup and never loosen a threshold merely to make a test pass.

## Architecture

```text
WAV files / Manifest ──> audio.py / batch.py ──> metrics.py ──> threshold evaluation
                                                       │                    │
                                                       └────────────┬───────┘
                                                                    v
                                                         report.py (HTML + JSON)
                                                                    │
                                                                 cli.py
```

- `audio.py`: safe WAV loading/writing and deterministic stimuli.
- `batch.py`: manifest parsing, fault-isolated suite runner, and path resolution.
- `metrics.py`: sample-based metrics, delay estimate, and pass/fail gates.
- `report.py`: standalone per-test and summary dashboard HTML reports; no CDN or browser-side dependencies.
- `cli.py`: command-line entry point with CI-friendly exit codes (`0`, `1`, `2`).

## Testing

```bash
pytest
```

The unit tests use generated arrays and temporary WAV files, so they do not require external datasets or plugins.

## Roadmap

Upcoming milestones:

1. [x] **Batch/manifest mode:** suite execution from JSON manifests with per-test thresholds and aggregated HTML/JSON reporting.
2. [ ] **Calibrated frequency-response checks:** estimate response from a sweep or known broadband input and report band-wise deviations. (The current MVP exposes diagnostic magnitude-spectrum metrics only.)
3. [ ] **THD+N test protocol:** add a calibrated sine-based analyzer with clear limits and test-signal requirements.
4. [ ] **Plugin adapter:** automate offline renders for one chosen host/format, starting with an open test plugin or a JUCE test harness.
5. [ ] **CI integration:** publish HTML and JSON reports as artifacts on every commit.
6. [ ] **Benchmark history:** track render metrics over time and show regressions across builds.

Do not market items on the roadmap as implemented features.

## Limitations

- The initial implementation compares rendered files; it is not yet a plugin host or automated DAW controller.
- Sample rates and channel counts must match. The tool refuses silent resampling because it can mask test setup errors.
- WAV files are decoded to floating point for analysis; the original integer bit depth is not treated as a quality metric.
- Delay is an estimate from cross-correlation. Silence and periodic signals may not support an unambiguous estimate.
- RMS/peak/SNR/correlation are diagnostics, not subjective listening tests or compliance certification.

## Continue development in Antigravity

Open this repository folder in Antigravity and start by reading [`AGENTS.md`](AGENTS.md) and [`ANTIGRAVITY_HANDOFF.md`](ANTIGRAVITY_HANDOFF.md). The handoff explains the project goal, current implementation, verified tests, known limitations, architecture, and the suggested next milestone (batch comparisons from a JSON manifest).

Recreate the included end-to-end PASS/FAIL examples with:

```bash
python scripts/create_demo.py
```

The reproducible audio fixtures and reports are in [`examples/demo/`](examples/demo/). They compare an exact render with a synthetic render delayed by 2.5 ms.

## License

MIT. See [LICENSE](LICENSE).
