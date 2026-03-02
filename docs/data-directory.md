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
│           └── w<segment>_o<overlap>/
│               ├── embeddings/
│               │   ├── interval-subs/
│               │   │   └── S01E01.srt
│               │   └── <model-name>/
│               │       └── <interval>.npy
│               ├── hnswlib.index/
│               │   └── <interval>.idx
│               └── annoy.index/
│                   └── <interval>.idx
└── matches/
    └── <timestamp>.jsonl
```

#### `series.tmdb.json`
Created by `init-series`. Contains series details and full season/episode metadata fetched from TMDb.

#### `settings.json`
Created by `init-series`. Stores per-series settings:

- `segment_duration` (seconds, default `30`)
- `subtitle_overlap_seconds` (seconds, default `5`)
- `random_seed` (default `12345`)
- `window_expansion_mode` (`always` or `two-stage`, default `always`)
- `window_neighbor_radius` (default `1`)
- `low_info_filter` (default `true`)
- `low_info_min_words` (default `8`)
- `low_info_cue_ratio` (default `0.25`)
- `max_results_per_query` (default `10`)
- `subtitle_quality_enabled` (default `true`)
- `subtitle_quality_max_candidates` (default `3`)
- `subtitle_quality_runtime_ratio_max` (default `1.45`)
- `subtitle_quality_runtime_ratio_min` (default `0.55`)
- `subtitle_quality_overlap_containment_threshold` (default `0.35`)
- `multi_episode_mode` (`auto`, `off`, `force-2`; default `auto`)
- `multi_episode_duration_ratio_threshold` (default `1.70`)
- `multi_episode_segments_ratio_threshold` (default `1.70`)
- `multi_episode_min_extra_minutes` (default `10.0`)
- `multi_episode_min_extra_segments` (default `6`)
- `multi_episode_split_search_window_seconds` (default `180`)
- `multi_episode_min_side_segments` (default `4`)
- `multi_episode_candidate_k` (default `10`)
- `multi_episode_candidate_k_retry` (default `20`)
- `multi_episode_second_half_horizon_multiplier` (default `1.5`)
- `multi_episode_pair_margin` (default `0.05`)
- `multi_episode_miss_penalty` (default `1.20`)

You can override these defaults at init time with `--segment-duration`, `--subtitle-overlap-seconds`, `--random-seed`, and the matching-policy flags documented in `docs/cli.md`.

#### `subtitles/`
Created by `fetch-subs`. Contains:

- `.srt` subtitle files (one per episode)
- `.opensubtitles` metadata files (JSON payload from OpenSubtitles)
- `quality/` subtitle-quality artifacts:
  - `SxxEyy.quality.json`: Per-episode quality diagnostics and verdict (`pass` or `quarantined`)
  - `quarantine.jsonl`: Append-only quarantine/replacement event log

#### `segments/<segment_duration>/`
Created when you run `match` or other segment-based operations. The `<segment_duration>` directory name matches the value in `settings.json`. Changing the segment duration creates a new subdirectory.

- `video-info-cache.json`
  - A cache of video file metadata (duration, size) to avoid repeated probing.
- `transcriptions/text/`
  - Per-video transcription caches in JSON, keyed by segment index.
  - Filenames are based on a hash of the video path plus the original filename.
- `transcriptions/embeddings/`
  - Per-video embedding arrays created from the transcriptions (`.npy`).
  - Per-video sidecar metadata (`.meta.json`) containing low-information interval annotations used by runtime filtering.

#### `segments/<segment_duration>/indexes/`
Created by `index-subs`. Stores embedding data and index structures used for subtitle matching.

- `w<segment>_o<overlap>/`
  - Window profile directory keyed by segment duration and subtitle overlap.
  - Example: `w30_o5` for 30-second windows with 5-second overlap.
- `w<segment>_o<overlap>/embeddings/interval-subs/`
  - Sliding-window subtitle files generated from source `.srt` files.
- `w<segment>_o<overlap>/embeddings/<model-name>/`
  - Embedding arrays per window index (`.npy`).
  - The default model directory is `sentence-transformers-all-MiniLM-L6-v2`.
- `w<segment>_o<overlap>/hnswlib.index/` and `w<segment>_o<overlap>/annoy.index/`
  - Per-window index files (`.idx`) for the selected index backend.

#### `matches/`
Created by `match`. Contains:

- `<timestamp>.jsonl` with interval-level match results.
- `<timestamp>.assignments.jsonl` with resolved per-video assignment diagnostics (`single` vs `multi_2`, assigned episodes, split, confidence, detector reasons).

## Notes

- The series-level `.mkv-episode-matcher/` directory is considered part of your series data. You can safely delete it to force a re-initialization, but you will lose cached subtitles, indexes, and match history.
- The user-level config directory is separate from series data and is shared across all series.
