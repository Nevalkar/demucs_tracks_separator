# Audio Track Separator

A powerful Python tool for separating audio tracks into individual stems using
various Demucs models. This tool allows you to extract different instruments and
vocals from your music files.

## Features

- Multiple model support (4-stem and 6-stem separation)
- Support for various audio formats (MP3, WAV, AIFF)
- Real-time processing feedback with progress indication
- Detailed audio analysis
- High-quality stem separation
- Clean and organized output structure

## Available Models

- **htdemucs**: Basic model, good for general-purpose separation (4 stems)
- **htdemucs_ft**: Fine-tuned model with improved vocal separation (4 stems)
- **htdemucs_6s**: Extended model with guitar and piano separation (6 stems)
- **mdx**: High-quality model focused on vocal/instrumental separation (4 stems)

## Prerequisites

### Windows

1. Install Python 3.9 or higher:

   - Download from
     [Python's official website](https://www.python.org/downloads/)
   - Make sure to check "Add Python to PATH" during installation

2. Install Visual C++ Build Tools:

   - Download and install
     [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   - During installation, select "Desktop development with C++"

3. Install FFmpeg:

   ```bash
   # Using Chocolatey (recommended)
   choco install ffmpeg

   # Or download manually from https://ffmpeg.org/download.html
   ```

### macOS

1. Install Python 3.9 or higher:

   ```bash
   brew install python
   ```

2. Install required system packages:

   ```bash
   # Install Homebrew if you haven't already
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

   # Install required packages
   brew install ffmpeg
   brew install sox
   brew install libsndfile
   brew install portaudio
   brew install cmake

   # Install Apple Command Line Tools if not already installed
   xcode-select --install
   ```

### Linux (Ubuntu/Debian)

1. Install Python 3.9 or higher and development packages:

   ```bash
   sudo apt update
   sudo apt install python3 python3-pip python3-dev
   ```

2. Install required system packages:
   ```bash
   sudo apt install ffmpeg
   sudo apt install sox
   sudo apt install libsox-fmt-all
   sudo apt install libsndfile1
   sudo apt install libasound2-dev
   sudo apt install portaudio19-dev
   sudo apt install build-essential
   sudo apt install cmake
   ```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/audio-track-separator.git
   cd audio-track-separator
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

```bash
python script.py -i path/to/your/song.mp3
```

### Advanced Options

```bash
# List available models
python script.py --list-models

# Use specific model
python script.py -i song.mp3 -m htdemucs_6s

# Change output format
python script.py -i song.mp3 -f wav
```

### Available Arguments

- `-i`, `--input`: Input audio file path (default: test.mp3)
- `-m`, `--model`: Model to use for separation (default: htdemucs_ft)
- `-f`, `--format`: Output format (aiff, wav, mp3) (default: aiff)
- `--list-models`: Display available models and exit

## Output Structure

```
output/
└── stems/
    └── song_name/
        ├── drums.aiff
        ├── bass.aiff
        ├── vocals.aiff
        ├── guitar.aiff (when using 6-stem model)
        ├── piano.aiff (when using 6-stem model)
        └── other.aiff
```

## Model Details

### htdemucs

- Basic 4-stem separation
- Stems: drums, bass, vocals, other
- Good for general use

### htdemucs_ft

- Fine-tuned 4-stem separation
- Stems: drums, bass, vocals, other
- Optimized for vocal extraction
- Default model

### htdemucs_6s

- Extended 6-stem separation
- Stems: drums, bass, vocals, guitar, piano, other
- Best for isolating specific instruments

### mdx

- High-quality 4-stem separation
- Stems: drums, bass, vocals, other
- Focused on clean vocal/instrumental separation

## Requirements

The following Python packages will be installed automatically via
`requirements.txt`:

```
torch>=2.0.0
torchaudio>=2.0.0
demucs>=4.0.0
numpy>=1.20.0
ffmpeg-python>=0.2.0
tqdm>=4.65.0
diffq>=0.2.3
```

## Troubleshooting

### Common Issues

1. **FFmpeg not found**

   - Verify FFmpeg is installed correctly
   - Make sure FFmpeg is in your system PATH
   - Try reinstalling FFmpeg

2. **Memory Issues**

   - Try processing shorter audio files
   - Close other applications
   - Ensure you have at least 8GB of RAM
   - Consider using a machine with more memory

3. **Slow Processing**

   - Processing time depends on:
     - Audio file length
     - Selected model
     - CPU/GPU capabilities
     - Available memory
   - The progress bar will show real-time status

4. **Installation Issues**
   - Make sure all prerequisites are installed
   - Use a virtual environment
   - Update pip before installing requirements
   - Check system compatibility

## Performance Tips

1. **For Best Quality:**

   - Use high-quality input files (WAV/AIFF preferred)
   - Use htdemucs_ft for vocal focus
   - Use htdemucs_6s for instrument separation
   - Allow processing to complete without interruption

2. **For Faster Processing:**
   - Use shorter audio clips
   - Close unnecessary applications
   - Use SSD for storage
   - Ensure adequate RAM (8GB minimum)
   - Consider using a machine with GPU support

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file
for details.

## Acknowledgments

- [Demucs](https://github.com/facebookresearch/demucs) by Facebook Research
- FFmpeg for audio processing
- PyTorch community

## Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter) Project Link:
[https://github.com/yourusername/audio-track-separator](https://github.com/yourusername/audio-track-separator)
