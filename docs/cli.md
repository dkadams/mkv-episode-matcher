# Command Line Interface

MKV Episode Matcher uses a subcommand-based CLI.

## Usage

```bash
mkv-episode-matcher <command> [options]
```

Global options:

- `--version`: Show the version and exit
- `--verbose`, `-v`: Enable verbose output
- `--log-dir`: Directory where logs are written for this run (overrides config file)

Many commands also accept configuration overrides (see `config` below).

## Commands

### `config` (aliases: `onboard`)
Interactive configuration for API credentials.

```bash
mkv-episode-matcher config
mkv-episode-matcher onboard
```

Options:

- `--config`, `-c`: Path to `config.ini` (default: `~/.mkv-episode-matcher/config.ini`)
- `--set-log-dir`: Persist logs directory in `config.ini` under `[logging].log_dir`
- `--tmdb-api-key`
- `--open_subtitles_api_key`
- `--open_subtitles_user_agent`
- `--open_subtitles_username`
- `--open_subtitles_password`

### `init-series`
Initialize a series directory with metadata from TMDb.

```bash
mkv-episode-matcher init-series /path/to/Series
```

Options:

- `--name`: Series name (defaults to directory name)
- `--id`: TMDb series id (skip search prompt)
- `--refresh`: Re-fetch series details
- `--segment-duration`: Segment duration in seconds (default: `30`)
- `--subtitle-overlap-seconds`: Subtitle window overlap in seconds (default: `5`)
- `--random-seed`: Random seed for segment selection (default: `12345`)
- `--window-expansion-mode {always,two-stage}`: Neighbor-window expansion mode (default: `always`)
- `--window-neighbor-radius`: Neighbor-window radius (default: `1`)
- `--low-info-filter` / `--no-low-info-filter`: Enable/disable low-information transcript filtering (default: enabled)
- `--low-info-min-words`: Minimum token count used by low-information filter (default: `8`)
- `--low-info-cue-ratio`: Cue-token ratio threshold used by low-information filter (default: `0.25`)
- `--max-results-per-query`: Max ANN candidates per queried window (default: `10`)
- `--multi-episode-mode {auto,off,force-2}`: Multi-episode handling mode (default: `auto`)
- `--multi-episode-duration-ratio-threshold`: Duration ratio threshold for auto multi detection (default: `1.70`)
- `--multi-episode-segments-ratio-threshold`: Segment-count ratio threshold for auto multi detection (default: `1.70`)
- `--multi-episode-min-extra-minutes`: Minimum extra minutes above expected single runtime (default: `10.0`)
- `--multi-episode-min-extra-segments`: Minimum extra segments above expected single runtime (default: `6`)
- `--multi-episode-split-search-window-seconds`: Split search window around expected split (default: `180`)
- `--multi-episode-min-side-segments`: Minimum segments required on each split side (default: `4`)
- `--multi-episode-candidate-k`: Per-chunk episode candidate count (default: `10`)
- `--multi-episode-candidate-k-retry`: Retry candidate count if no consecutive pair is found (default: `20`)
- `--multi-episode-second-half-horizon-multiplier`: Extra second-half window horizon multiplier (default: `1.5`)
- `--multi-episode-pair-margin`: Margin required between best and second-best pair score (default: `0.05`)
- `--multi-episode-miss-penalty`: Penalty for missing chunk/episode hits (default: `1.20`)

### `fetch-subs`
Download subtitles for a series from OpenSubtitles.

```bash
mkv-episode-matcher fetch-subs /path/to/Series
```

Options:

- `--refresh`: Download even if subtitles exist
- `--subtitle-quality` / `--no-subtitle-quality`: Enable/disable subtitle quality checks with candidate retry selection (default: enabled)
- `--subtitle-quality-max-candidates`: Maximum candidate subtitles evaluated per episode (default: `3`)
- `--subtitle-quality-runtime-ratio-max`: Runtime-ratio hard-fail upper bound vs TMDB runtime (default: `1.45`)
- `--subtitle-quality-runtime-ratio-min`: Runtime-ratio hard-fail lower bound vs TMDB runtime (default: `0.55`)
- `--subtitle-quality-overlap-containment-threshold`: Neighbor line-containment suspicious threshold (default: `0.35`)
- `--subtitle-quality-report`: Optional JSON summary output for subtitle quality evaluations
- `--seasons <N...>` or `--episodes SEASON:SPEC`: Limit which episodes are processed

### `index-subs`
Build subtitle indexes for a series.

```bash
mkv-episode-matcher index-subs /path/to/Series
```

Options:

- `--rebuild`: Rebuild indexes from scratch
- `--subtitle-overlap-seconds`: Override subtitle window overlap in seconds (default: series setting or `5`)
- `--include-quarantined-subs` / `--no-include-quarantined-subs`: Include subtitles marked as quarantined by quality checks (default: excluded)
- `--annoy`, `--hnswlib`: Select index backend (default: hnswlib)
- `--seasons <N...>` or `--episodes SEASON:SPEC`: Limit which episodes are processed

Set `--subtitle-overlap-seconds 0` to use fixed, non-overlapping windows.

### `match`
Match video files against the indexed subtitle data.

```bash
mkv-episode-matcher match /path/to/Series /path/to/videos
```

Options:

- `--segments-per-minute`: Controls how many segments are transcribed (default: `0.5`)
- `--subtitle-overlap-seconds`: Override subtitle window overlap in seconds (default: series setting or `5`)
- `--window-expansion-mode {always,two-stage}`: Neighbor-window expansion mode (default: series setting or `always`)
- `--window-neighbor-radius`: Neighbor-window radius (default: series setting or `1`)
- `--low-info-filter` / `--no-low-info-filter`: Enable/disable low-information transcript filtering (default: series setting or enabled)
- `--low-info-min-words`: Minimum token count used by low-information filter
- `--low-info-cue-ratio`: Cue-token ratio threshold used by low-information filter
- `--max-results-per-query`: Max ANN candidates per queried window (default: series setting or `10`)
- `--multi-episode-mode {auto,off,force-2}`: Multi-episode handling mode (default: series setting or `auto`)
- `--multi-episode-duration-ratio-threshold`: Duration ratio threshold for auto multi detection
- `--multi-episode-segments-ratio-threshold`: Segment-count ratio threshold for auto multi detection
- `--multi-episode-min-extra-minutes`: Minimum extra minutes above expected single runtime
- `--multi-episode-min-extra-segments`: Minimum extra segments above expected single runtime
- `--multi-episode-split-search-window-seconds`: Split search window around expected split
- `--multi-episode-min-side-segments`: Minimum segments required on each split side
- `--multi-episode-candidate-k`: Per-chunk episode candidate count
- `--multi-episode-candidate-k-retry`: Retry candidate count when no consecutive pair is found
- `--multi-episode-second-half-horizon-multiplier`: Extra second-half window horizon multiplier
- `--multi-episode-pair-margin`: Margin required between best and second-best pair score
- `--multi-episode-miss-penalty`: Penalty for missing chunk/episode hits
- `--num-matches`, `-n`: Number of top matches to show (default: `5`)
- `--confidence`: Confidence threshold (default: `0.7`)
- `--no-transcription-cache`: Disable reusing cached transcriptions
- `--display-by-episode`, `-E`: Show results grouped by episode (default)
- `--display-by-file`, `-F`: Show results grouped by file
- `--whispercpp`, `--faster-whisper`, `--parakeet-mlx`: Transcriber backend (default: `parakeet-mlx` on macOS, `whispercpp` on other platforms). `parakeet-mlx` is macOS-only.

Whisper.cpp runtime tuning (when using `--whispercpp`):

- `WHISPERCPP_THREADS`: Sets whisper.cpp `-t/--threads` per transcription process.
  If unset, the app defaults to `2` threads per transcription process.
- `WHISPERCPP_PROCESSORS`: Sets whisper.cpp `-p/--processors` per transcription process.
- `WHISPERCPP_EXTRA_ARGS`: Appends extra whisper.cpp CLI flags (parsed with shell-style quoting), for example `-bo 1 -l en`.
- Whispercpp now supports Stage-B micro-batch submission and runs one `whisper-cli` call per submitted batch using repeated
  `-f`/`-of` pairs.

Example tuning loop on x86_64:

```bash
WHISPERCPP_THREADS=8 \
WHISPERCPP_PROCESSORS=1 \
mkv-episode-matcher match /path/to/Series /path/to/videos \
  --whispercpp \
  --transcribe-workers 2 \
  --io-workers 2
```

Then compare with:

```bash
WHISPERCPP_THREADS=4 \
mkv-episode-matcher match /path/to/Series /path/to/videos \
  --whispercpp \
  --transcribe-workers 4 \
  --io-workers 2
```

Use whichever setting yields lower wall-clock time on your hardware.

Faster-whisper runtime tuning (when using `--faster-whisper`):

- `FASTER_WHISPER_BATCH_SIZE`: Max number of audio inputs sent per batched inference call (default: `8`).
- `FASTER_WHISPER_DEVICE`: Device passed to faster-whisper model init (default: `auto`).
- `FASTER_WHISPER_COMPUTE_TYPE`: Compute type passed to faster-whisper model init (default: `default`).
- `FASTER_WHISPER_CPU_THREADS`: Optional CPU thread count override for the model backend.

Transcription scheduler micro-batching:

- `MEM_TRANSCRIBE_MICROBATCH_SIZE`: Global Stage-B micro-batch size override (integer >= 1).
  If unset, each backend default is used (`8` for `whispercpp`, `parakeet-mlx`, and `faster-whisper`;
  `1` for non-batch backends).
- `MEM_TRANSCRIBE_MICROBATCH_MAX_WAIT_MS`: Max wait time before flushing a partial Stage-B batch
  (default: `15` ms).

### `collect-dataset`
Collect labeled transcription/subtitle pairs for evaluation.

```bash
mkv-episode-matcher collect-dataset /path/to/Series --output-dir /tmp/dataset
```

Options (subset):

- `--segment-duration`: Override segment duration in seconds
- `--subtitle-overlap-seconds`: Override subtitle window overlap in seconds
- `--segments-per-minute`: Segment sampling rate for transcription
- `--include-aligned` / `--no-include-aligned`: Include aligned variants
- `--misalign-profiles`: Misalignment profiles to generate

### `evaluate-dataset`
Evaluate transcript-to-subtitle retrieval quality from a collected dataset.

```bash
mkv-episode-matcher evaluate-dataset /tmp/dataset
```

Options (subset):

- `--segment-duration`: Override segment duration in seconds
- `--subtitle-overlap-seconds`: Override subtitle window overlap in seconds
- `--window-expansion-mode {always,two-stage}`: Neighbor-window expansion mode
- `--window-neighbor-radius`: Neighbor-window radius
- `--low-info-filter` / `--no-low-info-filter`: Enable/disable low-information transcript filtering
- `--low-info-min-words`: Minimum token count used by low-information filter
- `--low-info-cue-ratio`: Cue-token ratio threshold used by low-information filter
- `--max-results-per-query`: Max ANN candidates per queried window
- `--support-window-bonus`: Bonus applied per additional supporting window
- `--support-offset-penalty`: Penalty applied for window offset from mapped interval
- `--multi-episode-mode {auto,off,force-2}`: Multi-episode handling mode
- `--multi-episode-duration-ratio-threshold`: Duration ratio threshold for auto multi detection
- `--multi-episode-segments-ratio-threshold`: Segment-count ratio threshold for auto multi detection
- `--multi-episode-min-extra-minutes`: Minimum extra minutes above expected single runtime
- `--multi-episode-min-extra-segments`: Minimum extra segments above expected single runtime
- `--multi-episode-split-search-window-seconds`: Split search window around expected split
- `--multi-episode-min-side-segments`: Minimum segments required on each split side
- `--multi-episode-candidate-k`: Per-chunk episode candidate count
- `--multi-episode-candidate-k-retry`: Retry candidate count when no consecutive pair is found
- `--multi-episode-second-half-horizon-multiplier`: Extra second-half window horizon multiplier
- `--multi-episode-pair-margin`: Margin required between best and second-best pair score
- `--multi-episode-miss-penalty`: Penalty for missing chunk/episode hits
- `--top-k`: Top-k values to report
- `--profiles`: Variant profiles to include
- `--errors-output`: Optional path for JSONL/CSV/HTML top-1 mismatch export
- `--include-match-text`: Include matched subtitle window text/time details in mismatch exports
- `--include-quarantined-subs` / `--no-include-quarantined-subs`: Include subtitles marked as quarantined by quality checks (default: excluded)

This command uses indexed ANN retrieval over subtitle windows (hnswlib), support-aware per-episode scoring, and optional low-information transcript filtering before computing Top-k/MRR against manifest labels.
Mismatch exports include derived segment time ranges (seconds and `HH:MM:SS`) alongside segment indexes.
It also reports video-level assignment metrics and diagnostics for split multi-episode resolution (`single` vs `multi_2`) using runtime-profile detection.

### `benchmark-transcribers`
Benchmark available transcription backends against one or more input paths.

```bash
mkv-episode-matcher benchmark-transcribers /path/to/videos
mkv-episode-matcher benchmark-transcribers /path/to/videos --backend whispercpp --backend parakeet-mlx
mkv-episode-matcher benchmark-transcribers /path/to/videos --backend whispercpp --backend faster-whisper
```

Options:

- `--extension`, `-e`: File extension(s) to include when scanning directories (default: `.mkv`)
- `--segments-per-minute`: Number of audio chunks to sample per minute of media (default: `0.5`)
- `--segment-duration`: Override segment duration in seconds (default: series value or `30`)
- `--random-seed`: Override random seed for deterministic segment selection (default: series value or `12345`)
- `--thread-workers`: Thread pool size for subprocess backends (default: `10`)
- `--process-workers`: Process pool size for Python model backends (default: `8`)
- `--backend`: Repeatable backend filter (`whispercpp`, `faster-whisper`, `parakeet-mlx`)

Notes:

- By default, all backends are benchmarked one-by-one.
- Transcription outputs are temporary and discarded; this command does not persist benchmark transcriptions under your series `.mkv-episode-matcher` directory.

## Episode Specifiers

Commands that support `--episodes SEASON:SPEC` accept:

- `N` (single episode)
- `A,B,C` (list)
- `A-B`, `-B`, `A-` (range)
- `*` (all)

Examples:

```bash
mkv-episode-matcher fetch-subs /path/to/Series --episodes 1:1 2:1,2 3:5-8
```

## Logging

By default, logs are stored in:

```
~/.mkv-episode-matcher/logs/
```

Files are written per run with timestamps, for example:

- `stdout-YYYYMMDDTHHmmss.log`
- `stderr-YYYYMMDDTHHmmss.log`

You can change the log directory using either:

- CLI for one run: `--log-dir /path/to/logs`
- `config.ini`:

```ini
[logging]
log_dir = /path/to/logs
```

Precedence is `--log-dir` > `config.ini` > default path.
