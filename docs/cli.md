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

### `fetch-subs`
Download subtitles for a series from OpenSubtitles.

```bash
mkv-episode-matcher fetch-subs /path/to/Series
```

Options:

- `--refresh`: Download even if subtitles exist
- `--seasons <N...>` or `--episodes SEASON:SPEC`: Limit which episodes are processed

### `index-subs`
Build subtitle indexes for a series.

```bash
mkv-episode-matcher index-subs /path/to/Series
```

Options:

- `--rebuild`: Rebuild indexes from scratch
- `--subtitle-overlap-seconds`: Override subtitle window overlap in seconds (default: series setting or `5`)
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
- `--top-k`: Top-k values to report
- `--profiles`: Variant profiles to include
- `--errors-output`: Optional path for JSONL/CSV/HTML top-1 mismatch export
- `--include-match-text`: Include matched subtitle window text/time details in mismatch exports

This command uses indexed ANN retrieval over subtitle windows (hnswlib), support-aware per-episode scoring, and optional low-information transcript filtering before computing Top-k/MRR against manifest labels.
Mismatch exports include derived segment time ranges (seconds and `HH:MM:SS`) alongside segment indexes.

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
