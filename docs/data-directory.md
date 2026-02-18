# Data Directory (.mkv-episode-matcher)

MKV Episode Matcher uses two different `.mkv-episode-matcher` directories:

- **User-level config directory** under your home folder.
- **Series-level data directory** inside each TV series folder you initialize.

This page documents how each directory is created, where it lives by default, how (and if) its location can be configured, and the files/subdirectories it can contain.

## User-Level Config Directory

### Default location

- macOS/Linux: `~/.mkv-episode-matcher/`
- Windows: `%USERPROFILE%\.mkv-episode-matcher\`

### Creation

This directory is created automatically whenever the configuration module is loaded (for example, when you run any CLI command). It is created if it does not already exist.

### Location configuration

The base directory **is not configurable**.

You *can* choose a different **configuration file path** by passing `--config /path/to/config.ini`, but that only changes where the config file is read/written. It does **not** move the logs directory or other user-level files.

The logs directory is configurable:

- Per-run override: `--log-dir /path/to/logs`
- Persistent config:

```ini
[logging]
log_dir = /path/to/logs
```

Precedence is `--log-dir` > `config.ini` > default `~/.mkv-episode-matcher/logs/`.

### Contents

- `config.ini`
  - Created/updated by `mkv-episode-matcher config`.
  - Stores API credentials for TMDb and OpenSubtitles.
- `logs/`
  - Created at application startup.
  - Contains log files named like `stdout-YYYYMMDDTHHmmss.log` and `stderr-YYYYMMDDTHHmmss.log`.

## Series-Level Data Directory

### Default location

Inside each series root directory that you initialize:

```
/Path/To/Series
└── .mkv-episode-matcher/
```

### Creation

The directory is created by `mkv-episode-matcher init-series <series_dir>`.

- `init-series` creates `.mkv-episode-matcher/` and writes the initial metadata files.
- Additional subdirectories are created lazily by other commands (fetching subtitles, indexing, matching, etc.).

### Location configuration

The series data directory **always lives inside the series directory** you pass to `init-series`. There is no CLI flag or config setting to relocate it; you control its location by choosing the series root directory.

### Contents

The `.mkv-episode-matcher/` directory is organized like this (some items appear only after you run specific commands):

```
.mkv-episode-matcher/
├── series.tmdb.json
├── settings.json
├── subtitles/
│   ├── S01E01.srt
│   ├── S01E01.opensubtitles
│   └── ...
├── segments/
│   └── <segment_duration>/
│       ├── video-info-cache.json
│       ├── transcriptions/
│       │   ├── text/
│       │   │   └── <hashed_source>.json
│       │   └── embeddings/
│       │       └── <hashed_source>.npy
│       └── indexes/
│           ├── embeddings/
│           │   ├── interval-subs/
│           │   │   └── S01E01.srt
│           │   └── <model-name>/
│           │       └── <interval>.npy
│           ├── hnswlib.index/
│           │   └── <interval>.idx
│           ├── annoy.index/
│           │   └── <interval>.idx
│           └── chroma.index/
│               └── (chroma db files)
└── matches/
    └── <timestamp>.jsonl
```

#### `series.tmdb.json`
Created by `init-series`. Contains series details and full season/episode metadata fetched from TMDb.

#### `settings.json`
Created by `init-series`. Stores per-series settings:

- `segment_duration` (seconds, default `30`)
- `random_seed` (default `12345`)

You can override these defaults at init time with `--segment-duration` and `--random-seed`.

#### `subtitles/`
Created by `fetch-subs`. Contains:

- `.srt` subtitle files (one per episode)
- `.opensubtitles` metadata files (JSON payload from OpenSubtitles)

#### `segments/<segment_duration>/`
Created when you run `match` or other segment-based operations. The `<segment_duration>` directory name matches the value in `settings.json`. Changing the segment duration creates a new subdirectory.

- `video-info-cache.json`
  - A cache of video file metadata (duration, size) to avoid repeated probing.
- `transcriptions/text/`
  - Per-video transcription caches in JSON, keyed by segment index.
  - Filenames are based on a hash of the video path plus the original filename.
- `transcriptions/embeddings/`
  - Per-video embedding arrays created from the transcriptions (`.npy`).

#### `segments/<segment_duration>/indexes/`
Created by `index-subs`. Stores embedding data and index structures used for subtitle matching.

- `embeddings/interval-subs/`
  - Fixed-interval subtitle files generated from source `.srt` files.
- `embeddings/<model-name>/`
  - Embedding arrays per interval (`.npy`).
  - The default model directory is `sentence-transformers-all-MiniLM-L6-v2`.
- `hnswlib.index/` and `annoy.index/`
  - Per-interval index files (`.idx`) for the selected index backend.
- `chroma.index/`
  - On-disk Chroma database files (directory contents managed by Chroma).

#### `matches/`
Created by `match`. Contains a JSONL file per run with interval-level match results. Filenames are timestamps in ISO format.

## Notes

- The series-level `.mkv-episode-matcher/` directory is considered part of your series data. You can safely delete it to force a re-initialization, but you will lose cached subtitles, indexes, and match history.
- The user-level config directory is separate from series data and is shared across all series.
