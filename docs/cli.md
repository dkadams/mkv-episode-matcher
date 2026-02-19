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
- `--random-seed`: Random seed for segment selection (default: `12345`)

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
- `--chroma`, `--annoy`, `--hnswlib`: Select index backend (default: hnswlib)
- `--seasons <N...>` or `--episodes SEASON:SPEC`: Limit which episodes are processed

### `match`
Match video files against the indexed subtitle data.

```bash
mkv-episode-matcher match /path/to/Series /path/to/videos
```

Options:

- `--segments-per-minute`: Controls how many segments are transcribed (default: `0.5`)
- `--num-matches`, `-n`: Number of top matches to show (default: `5`)
- `--confidence`: Confidence threshold (default: `0.7`)
- `--no-transcription-cache`: Disable reusing cached transcriptions
- `--display-by-episode`, `-E`: Show results grouped by episode (default)
- `--display-by-file`, `-F`: Show results grouped by file
- `--whisper`, `--faster-whisper`, `--whispercpp-cli`, `--whisperkit-cli`, `--parakeet-mlx`: Transcriber backend (default: whisper.cpp CLI)

### `benchmark-transcribers`
Benchmark available transcription backends against one or more input paths.

```bash
mkv-episode-matcher benchmark-transcribers /path/to/videos
mkv-episode-matcher benchmark-transcribers /path/to/videos --backend whispercpp-cli --backend parakeet-mlx
```

Options:

- `--extension`, `-e`: File extension(s) to include when scanning directories (default: `.mkv`)
- `--segments-per-minute`: Number of audio chunks to sample per minute of media (default: `0.5`)
- `--segment-duration`: Override segment duration in seconds (default: series value or `30`)
- `--random-seed`: Override random seed for deterministic segment selection (default: series value or `12345`)
- `--thread-workers`: Thread pool size for subprocess backends (default: `10`)
- `--process-workers`: Process pool size for Python model backends (default: `8`)
- `--backend`: Repeatable backend filter (`whisper`, `faster-whisper`, `whispercpp-cli`, `whisperkit-cli`, `parakeet-mlx`)

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
