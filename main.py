#!/usr/bin/env python3
"""
Advanced Audio Track Separator
----------------------------
A professional tool to separate audio tracks into individual stems using various Demucs models.
Supports multiple model configurations and output formats.

Author: Your Name
License: MIT
"""

import os
import sys
import logging
import subprocess
import shutil
import argparse
from typing import List, Optional, Tuple, Dict, Union
from pathlib import Path
from datetime import datetime

# Audio processing
import torch
import torchaudio
import numpy as np
from demucs.pretrained import get_model
from demucs.apply import apply_model

class AudioSeparator:
    """
    Advanced audio track separator supporting multiple Demucs models and configurations.
    """
    
    # Available models and their configurations
    AVAILABLE_MODELS = {
        'htdemucs': {
            'stems': ['drums', 'bass', 'vocals', 'other'],
            'description': 'Basic model, good general-purpose separation',
            'stem_order': [0, 1, 3, 2]
        },
        'htdemucs_ft': {
            'stems': ['drums', 'bass', 'vocals', 'other'],
            'description': 'Fine-tuned model with better vocal separation',
            'stem_order': [0, 1, 3, 2]
        },
        'htdemucs_6s': {
            'stems': ['drums', 'bass', 'vocals', 'guitar', 'piano', 'other'],
            'description': 'Six stems separation including guitar and piano',
            'stem_order': [0, 1, 3, 2, 4, 5]
        },
        'mdx': {
            'stems': ['drums', 'bass', 'other', 'vocals'],  # Cambiado el orden
            'description': 'High quality model focused on vocal/instrumental separation',
            'stem_order': [0, 1, 3, 2]  # Cambiado el orden para reflejar la salida real del modelo
        }
    }
    
    def __init__(self, 
                 model_name: str = 'mhtdemucs_6s',  # Changed default to 10-stem model
                 output_format: str = 'aiff',
                 sample_rate: int = 44100) -> None:
        """
        Initialize the AudioSeparator with specified model and configuration.
        
        Args:
            model_name: Name of the Demucs model to use (see AVAILABLE_MODELS)
            output_format: Output audio format ('aiff', 'wav', or 'mp3')
            sample_rate: Output sample rate in Hz
        """
        # Validate model name
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Invalid model name: {model_name}. "
                f"Available models: {', '.join(self.AVAILABLE_MODELS.keys())}"
            )
        
        # Initialize configuration
        self.model_name = model_name
        self.output_format = output_format
        self.sample_rate = sample_rate
        self.model_config = self.AVAILABLE_MODELS[model_name]
        
        # Initialize paths
        self.project_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        self.output_dir = self.project_dir / 'output' / 'stems'
        self.temp_dir = self.project_dir / 'temp'
        self.log_dir = self.project_dir / 'logs'
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Create necessary directories
        self._create_directories()
        
        # Setup device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f'Using device: {self.device}')
        
        # Load model
        try:
            self.model = get_model(self.model_name)
            self.model.to(self.device)
            self.logger.info(f'Model {self.model_name} loaded successfully')
            self.logger.info(f'Available stems: {", ".join(self.model_config["stems"])}')
        except Exception as e:
            self.logger.error(f'Error loading model: {str(e)}')
            raise
        
        # Log configuration
        self.logger.info(f'Output format: {self.output_format}')
        self.logger.info(f'Sample rate: {self.sample_rate} Hz')
        
        # Verify dependencies
        self._check_dependencies()

    def _setup_logging(self) -> None:
        """Configure logging settings."""
        # Create logs directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate log filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.log_dir / f'audio_separator_{timestamp}.log'
        
        # Configure logging format
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Configure handlers
        handlers = [
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
        
        # Apply configuration
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=handlers,
            force=True
        )

    def _create_directories(self) -> None:
        """Create necessary directories for output and temporary files."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f'Directories created - Output: {self.output_dir}, Temp: {self.temp_dir}')

    def _check_dependencies(self) -> None:
        """Verify that all required external dependencies are installed."""
        try:
            # Check FFmpeg
            result = subprocess.run(['ffmpeg', '-version'], 
                                 capture_output=True, 
                                 text=True, 
                                 check=True)
            ffmpeg_version = result.stdout.split('\n')[0]
            self.logger.info(f'FFmpeg version: {ffmpeg_version}')
            
            # Check Python dependencies
            required_packages = {
                'torch': 'torch>=2.0.0',
                'torchaudio': 'torchaudio>=2.0.0',
                'demucs': 'demucs>=4.0.0',
                'numpy': 'numpy>=1.20.0',
                'diffq': 'diffq>=0.2.3'
            }
            
            missing_packages = []
            for package, requirement in required_packages.items():
                try:
                    __import__(package)
                except ImportError:
                    missing_packages.append(requirement)
            
            if missing_packages:
                print("\nInstalling missing dependencies...")
                subprocess.run([
                    'pip', 'install', *missing_packages
                ], check=True)
                print("Dependencies installed successfully!")
                
            # Log installed versions
            import torch
            import torchaudio
            import numpy as np
            
            for package, version in {
                'torch': torch.__version__,
                'torchaudio': torchaudio.__version__,
                'numpy': np.__version__
            }.items():
                self.logger.info(f'{package} version: {version}')
                
        except FileNotFoundError:
            self.logger.error('FFmpeg not found. Please install FFmpeg.')
            raise
        except subprocess.CalledProcessError as e:
            self.logger.error(f'Error installing dependencies: {e}')
            raise
        except Exception as e:
            self.logger.error(f'Dependency check failed: {str(e)}')
            raise

    def _load_audio(self, audio_path: Path) -> Tuple[torch.Tensor, int]:
        """
        Load and prepare audio file for processing.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Tuple containing the audio tensor and sample rate
        """
        self.logger.info(f'Loading audio file: {audio_path}')
        
        try:
            # Convert to WAV if needed
            if audio_path.suffix.lower() != '.wav':
                temp_wav = self.temp_dir / 'temp_input.wav'
                subprocess.run([
                    'ffmpeg', '-y',
                    '-i', str(audio_path),
                    '-acodec', 'pcm_s16le',
                    '-ar', str(self.sample_rate),
                    '-ac', '2',
                    str(temp_wav)
                ], check=True, capture_output=True)
                audio_path = temp_wav
            
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Ensure stereo
            if waveform.size(0) == 1:
                waveform = waveform.repeat(2, 1)
            
            self.logger.info(f'Audio loaded - Shape: {waveform.shape}, Sample rate: {sample_rate}Hz')
            return waveform, sample_rate
            
        except Exception as e:
            self.logger.error(f'Error loading audio: {str(e)}')
            raise

    def _save_audio(self, 
                   audio: torch.Tensor, 
                   sample_rate: int, 
                   output_path: Path) -> None:
        """Save audio tensor to file in specified format."""
        try:
            # First save as WAV
            temp_wav = self.temp_dir / 'temp_output.wav'
            torchaudio.save(temp_wav, audio, sample_rate)
            
            # Convert to desired format
            subprocess.run([
                'ffmpeg', '-y',
                '-i', str(temp_wav),
                '-f', self.output_format,
                str(output_path)
            ], check=True, capture_output=True)
            
            file_size = output_path.stat().st_size / (1024 * 1024)  # Size in MB
            self.logger.info(f'Saved {output_path.name} ({file_size:.1f} MB)')
            
        except Exception as e:
            self.logger.error(f'Error saving audio: {str(e)}')
            raise
        finally:
            # Clean up temporary file
            if temp_wav.exists():
                temp_wav.unlink()

    def separate_track(self, input_file: str) -> Dict[str, Path]:
        """
        Separate an audio track into its component stems.
        """
        try:
            from tqdm import tqdm
            import time
            import threading
            from itertools import cycle
        except ImportError:
            print("Installing tqdm for progress bars...")
            subprocess.run(['pip', 'install', 'tqdm'])
            from tqdm import tqdm
            import time
            import threading
            from itertools import cycle

        # Variable global para el progreso
        class Progress:
            def __init__(self):
                self.percentage = 0
                self.done = False
                self.start_time = None

        progress = Progress()

        # Función para mostrar una barra de progreso animada con porcentaje
        def show_progress_animation():
            spinner = cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
            progress.start_time = time.time()
            
            while not progress.done:
                elapsed_time = time.time() - progress.start_time
                # Estimamos que el proceso toma aproximadamente 3 minutos
                estimated_total_time = 180  # segundos
                
                # Calculamos el porcentaje basado en el tiempo transcurrido
                if elapsed_time < estimated_total_time:
                    progress.percentage = min(95, (elapsed_time / estimated_total_time) * 100)
                
                spinner_char = next(spinner)
                print(f"\rProcessing {spinner_char} [{progress.percentage:3.1f}%] This may take several minutes...", end='')
                time.sleep(0.1)

        input_path = Path(input_file)
        if not input_path.is_absolute():
            input_path = self.project_dir / input_path
            
        if not input_path.exists():
            raise FileNotFoundError(f'Input file not found: {input_path}')
            
        try:
            # Create output directory for this track
            track_output_dir = self.output_dir / input_path.stem
            track_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Load and process audio
            self.logger.info("Loading audio file...")
            wav, sr = self._load_audio(input_path)
            
            # Ensure correct dimensions [batch, channels, length]
            if wav.dim() == 2:
                wav = wav.unsqueeze(0)
            elif wav.dim() == 1:
                wav = wav.unsqueeze(0).unsqueeze(0)
                
            self.logger.info(f'Input tensor shape: {wav.shape}')
            
            # Move to device
            wav = wav.to(self.device)
            print(f"\nAudio length: {wav.shape[-1]/sr:.2f} seconds")
            
            print("\nSeparating audio tracks...")
            self.logger.info("Starting separation process")

            with torch.no_grad():
                # Iniciar la animación en un hilo separado
                animation_thread = threading.Thread(target=show_progress_animation)
                animation_thread.start()

                try:
                    # Process through model
                    self.logger.info("Applying model...")
                    sources = apply_model(self.model, wav, device=self.device)
                    progress.percentage = 95  # Casi completado
                    
                    self.logger.info(f'Sources shape after separation: {sources.shape}')
                    
                    # Move to CPU for processing
                    sources = sources.cpu()
                    progress.percentage = 100  # Completado
                    progress.done = True
                    
                finally:
                    # Asegurar que el hilo de animación se detenga
                    progress.done = True
                    animation_thread.join()
                    print("\rProcessing: Complete! [100%]" + " " * 40)

            print("\nAnalyzing separated sources:")
            for i in range(sources.shape[1]):
                print(f"Source {i}:")
                source_data = sources[0, i]
                avg_amplitude = source_data.abs().mean().item()
                max_amplitude = source_data.abs().max().item()
                print(f"  Average amplitude: {avg_amplitude:.4f}")
                print(f"  Maximum amplitude: {max_amplitude:.4f}")

            # Get stems configuration for current model
            stems = self.AVAILABLE_MODELS[self.model_name]['stems']
            print(f"\nModel configuration:")
            print(f"Model: {self.model_name}")
            print(f"Expected stems: {stems}")
            
            # Save each stem
            output_files = {}
            print("\nSaving individual stems:")
            
            for idx, name in enumerate(tqdm(stems, desc="Saving stems")):
                try:
                    print(f"\nProcessing stem: {name} (index: {idx})")
                    output_path = track_output_dir / f'{name}.{self.output_format}'
                    
                    # Extract stem data
                    stem_data = sources[0, idx] if sources.dim() == 4 else sources[idx]
                    
                    # Print statistics for this stem
                    avg_amp = stem_data.abs().mean().item()
                    max_amp = stem_data.abs().max().item()
                    print(f"  Statistics for {name}:")
                    print(f"    Average amplitude: {avg_amp:.4f}")
                    print(f"    Maximum amplitude: {max_amp:.4f}")
                    
                    self._save_audio(stem_data, sr, output_path)
                    output_files[name] = output_path
                    print(f"  ✓ Saved {name}")
                    
                except Exception as e:
                    print(f"\n  ✗ Error processing {name}: {str(e)}")
                    self.logger.error(f'Error processing stem {name}: {str(e)}')
                    continue

            if not output_files:
                raise RuntimeError("No stems were successfully processed")
            
            print("\n✨ Track separation completed successfully!")
            print("\nProcessed stems:")
            total_size = 0
            for stem_name, file_path in output_files.items():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                total_size += size_mb
                print(f"  ✓ {stem_name.ljust(10)}: {size_mb:.1f} MB")
            print(f"\nTotal size: {total_size:.1f} MB")
            print(f"Output directory: {track_output_dir}")
            
            return output_files
            
        except Exception as e:
            # Asegurar que el hilo de animación se detenga en caso de error
            if 'animation_thread' in locals():
                progress.done = True
                animation_thread.join()
            self.logger.error(f'Error during track separation: {str(e)}')
            self.logger.exception("Full traceback:")
            raise
        finally:
            # Clean up temporary directory
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                self.temp_dir.mkdir()

def list_available_models() -> None:
    """Display information about all available models."""
    models = AudioSeparator.AVAILABLE_MODELS
    
    print("\nAvailable Demucs Models:")
    print("------------------------")
    
    print("\n🌟 RECOMMENDED MODELS BY USE CASE:")
    print("\nBest for multi-instrument separation:")
    print("htdemucs_6s:")
    print("  - Six separate stems: drums, bass, vocals, guitar, piano, other")
    print("  - Best choice for separating individual instruments")
    print("  - Default model")
    
    print("\nBest for vocal isolation:")
    print("htdemucs_ft:")
    print("  - Four stems: drums, bass, vocals, other")
    print("  - Fine-tuned specifically for better vocal separation")
    
    print("\nBest for general use:")
    print("mdx:")
    print("  - Four stems: drums, bass, vocals, other")
    print("  - High quality general-purpose separation")

    print("\nBasic model:")
    print("htdemucs:")
    print("  - Four stems: drums, bass, vocals, other")
    print("  - Good for basic separation needs")

    print("\nUsage examples:")
    print("  Multi-instrument:    python script.py -i song.mp3 -m htdemucs_6s")
    print("  Best vocals:         python script.py -i song.mp3 -m htdemucs_ft")
    print("  General purpose:     python script.py -i song.mp3 -m mdx")

    print("\nNote: For best results, use high quality input files (WAV/AIFF or high bitrate MP3)")

def main():
    """Main function to run the audio separator."""
    # Default values
    DEFAULT_MODEL = 'htdemucs_6s'
    DEFAULT_INPUT = 'test.mp3'
    DEFAULT_FORMAT = 'aiff'
    
    # Create argument parser
    parser = argparse.ArgumentParser(
        description='Separate audio tracks into individual stems using Demucs.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Add arguments
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List all available models and exit'
    )
    
    parser.add_argument(
        '--input',
        '-i',
        type=str,
        default=DEFAULT_INPUT,
        help=f'Input audio file path (default: {DEFAULT_INPUT})'
    )
    
    parser.add_argument(
        '--model',
        '-m',
        type=str,
        default=DEFAULT_MODEL,
        choices=AudioSeparator.AVAILABLE_MODELS.keys(),
        help=f'Model to use for separation (default: {DEFAULT_MODEL})'
    )
    
    parser.add_argument(
        '--format',
        '-f',
        type=str,
        default=DEFAULT_FORMAT,
        choices=['aiff', 'wav', 'mp3'],
        help=f'Output format (default: {DEFAULT_FORMAT})'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Show available models if requested
        if args.list_models:
            list_available_models()
            return
        
        # Print initial info
        print("\nAudio Track Separator")
        print("--------------------")
        print(f"Input file: {args.input}")
        print(f"Model: {args.model}")
        print(f"Output format: {args.format}")
        model_info = AudioSeparator.AVAILABLE_MODELS[args.model]
        print(f"Stems to extract: {', '.join(model_info['stems'])}")
        print("--------------------\n")
        
        # Initialize separator
        separator = AudioSeparator(
            model_name=args.model,
            output_format=args.format,
            sample_rate=44100
        )
        
        # Process the file
        output_files = separator.separate_track(args.input)
        
        # Print results
        print("\nSeparation completed successfully!")
        print("\nGenerated files:")
        for stem_name, file_path in output_files.items():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"- {stem_name}: {file_path.name} ({size_mb:.1f} MB)")
        
        # Print output location
        print(f"\nFiles saved in: {output_files[next(iter(output_files))].parent}")
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Check if the input file exists and is readable")
        print("2. Install FFmpeg: brew install ffmpeg")
        print("3. Install dependencies: pip install -r requirements.txt")
        print("4. Check the logs directory for detailed error information")
        print("\nUsage examples:")
        print(f"  Default usage:   python script.py")
        print(f"  List models:     python script.py --list-models")
        print(f"  Custom input:    python script.py -i other_song.mp3")
        print(f"  Different model: python script.py -m htdemucs")
        print(f"  Change format:   python script.py -f wav")
        sys.exit(1)

if __name__ == "__main__":
    main()