# Quick Start Guide

Get started with MKV Episode Matcher quickly and efficiently.

## Basic Usage

### 1. Configure API credentials (first-time setup)

```bash
mkv-episode-matcher config
```

You will be prompted for:

- TMDb API key (required)
- OpenSubtitles API key, consumer name, username, and password (required for subtitle downloads)

The configuration file is stored at `~/.mkv-episode-matcher/config.ini` by default.

### 2. Initialize your series directory

Point the tool at the root directory of a TV series and initialize it:

```bash
mkv-episode-matcher init-series "/path/to/Series"
```

This creates a `.mkv-episode-matcher/` directory inside the series folder with metadata and settings.

### 3. (Optional) Download subtitles

```bash
mkv-episode-matcher fetch-subs "/path/to/Series"
```

### 4. Build subtitle indexes

```bash
mkv-episode-matcher index-subs "/path/to/Series"
```

### 5. Match video files

Provide one or more video files or directories to match:

```bash
mkv-episode-matcher match "/path/to/Series" "/path/to/Videos"
```

## Directory Structure

Expected TV show organization:

```
Show Name/
├── Season 1/
│   ├── episode1.mkv
│   ├── episode2.mkv
├── Season 2/
│   ├── episode1.mkv
│   └── episode2.mkv
```

## Configuration

Configuration can be updated any time with:

```bash
mkv-episode-matcher config
```

You can also override the config file path:

```bash
mkv-episode-matcher config --config "/custom/path/config.ini"
```

## Next Steps

- Read [Installation Guide](installation.md) for setup details
- See [Command Line Interface](cli.md) for all commands and options
- Check [Tips and Tricks](tips.md) for advanced usage
