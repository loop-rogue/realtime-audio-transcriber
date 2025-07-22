# Real-Time Audio Transcription & Translation (macOS)

A **local, privacy-focused** real-time audio transcription using whisper and translation application designed for **macOS**. This application captures and transcripts your Mac's system audio output using advanced streaming speech recognition optimized with the **LocalAgreement algorithm** for efficient real-time processing.

https://github.com/user-attachments/assets/31de31d1-3cb9-4310-867c-f2eb4687e3b5


## 🔒 Privacy & Local Processing

- **100% Local Processing**: All transcription and translation happens on your Mac - no data sent to external servers
- **No Privacy Concerns**: Your audio never leaves your device
- **Offline Capable**: Works without internet connection (except for initial model downloads)

## ✨ Features

- **Real-time audio capture** from macOS system audio via BlackHole virtual audio device
- **Live transcription** using Whisper tiny.en model with timestamp support
- **Intelligent text highlighting** - automatically highlights key words and important phrases in both English transcripts and Chinese translations for easier reading and comprehension
- **Streaming optimization** with LocalAgreement algorithm for minimal latency
- **Chinese translation** with highlighted keywords using local Ollama models
- **Terminal UI** with curses-based real-time display with colored highlighting
- **Audio recording** saves sessions to timestamped output folders
- **Session logging** with rotating file logs for debugging

## 🎯 LocalAgreement Algorithm Optimization

This application implements an optimized streaming transcription approach inspired by research in real-time speech recognition. The **LocalAgreement algorithm** provides:

- **Incremental Processing**: Audio is processed in overlapping chunks rather than waiting for complete utterances
- **Longest Common Prefix Matching**: Consecutive transcription results are compared to find confirmed text segments
- **Smart Buffer Management**: Uses punctuation marks (periods, commas) as natural breakpoints to trim audio buffers
- **Minimal Latency**: Outputs transcribed text as soon as segments are confirmed, rather than waiting for full sentences

This approach significantly reduces latency compared to traditional batch processing while maintaining transcription accuracy, making it ideal for real-time applications like live captioning or note-taking.

Reference: https://arxiv.org/pdf/2307.14743

## Prerequisites

- **uv** package manager ([install uv](https://docs.astral.sh/uv/getting-started/installation/))
- **BlackHole 2ch** virtual audio device (for macOS audio routing) (https://github.com/ExistentialAudio/BlackHole)
- **Ollama** with qwen2.5:14b or gemma3:27b model installed

### Install BlackHole (macOS)
```bash
brew install blackhole-2ch
```

### Setup BlackHole Audio Routing

BlackHole 2ch needs to be configured as your audio output device to capture system audio. However, when BlackHole is set as the output device, you won't hear any sound by default.

To both transcribe audio and hear it through your speakers/headphones, you'll need to set up proper audio routing. I recommend following this step-by-step video guide that demonstrates how to configure BlackHole with your Mac's audio settings: https://youtu.be/ZqA97qo4J9c?si=TciJUbp5oh4Xa9Ex&t=55. And then chose this device as your audio output.


### Install Ollama and Models
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull translation models
ollama pull qwen2.5:14b
# OR
ollama pull gemma3:27b
```

## Installation

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd real_time_transcript
```

2. **Install dependencies with uv:**
```bash
# Create virtual environment and install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate
```

3. **Alternative installation (pip):**
```bash
# If you prefer pip
pip install -r requirements.txt  # (generate with: uv export > requirements.txt)
```

## Usage

### Run the Application
```bash
# With uv
uv run python main.py

# Or with activated venv
python main.py
```

### Controls
- **Ctrl+C** - Stop recording and exit
- The application will automatically create timestamped output folders in `output/`

## Configuration

Key settings in `main.py`:
- `DEVICE = "BlackHole 2ch"` - Audio input device
- `SAMPLE_RATE = 16000` - Audio sample rate
- `CHUNK_DURATION = 2` - Processing chunk size in seconds
- Whisper model: "tiny.en" (can be changed to other models)
- Translation model: qwen2.5:14b or gemma3:27b in `process_transcription()`

## Development

### Install development dependencies:
```bash
uv sync --dev
```

### Code formatting and linting:
```bash
# Format code
uv run black .

# Lint code  
uv run ruff check .
```

### Run tests:
```bash
uv run pytest
```

## Troubleshooting

### UTF-8 Encoding Errors
The application includes error handling for malformed Whisper output text.


### Audio Device Issues
Ensure BlackHole 2ch is installed and set as your audio output device. 


## TODO

- [ ] Generate SRT file with timestamps
- [ ] Improve translation prompts
- [ ] Add configuration file support
- [ ] Add multiple language support

## License

MIT License
