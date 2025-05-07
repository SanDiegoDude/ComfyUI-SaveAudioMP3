# ComfyUI_SaveAudioMP3/save_audio_mp3_node.py

# ------------------------------------------------------------------------------------
# User Prerequisites for ComfyUI_SaveAudioMP3:
# 1. Install pydub:
#    In your ComfyUI's Python environment, run: pip install pydub
# 2. Install ffmpeg:
#    ffmpeg must be installed on your system and accessible in the PATH.
#    - Windows: Download from ffmpeg.org, extract, and add the 'bin' folder to your PATH.
#    - macOS: Install via Homebrew: brew install ffmpeg
#    - Linux: Install via package manager: sudo apt update && sudo apt install ffmpeg
# ------------------------------------------------------------------------------------

import os
import torch
import numpy as np
from pydub import AudioSegment # For MP3 conversion. Requires ffmpeg.
import folder_paths # ComfyUI utility for managing paths

class SaveAudioMP3:
    """
    A ComfyUI custom node to save raw audio waveforms as MP3 files.
    It uses pydub for MP3 conversion, which requires ffmpeg to be installed and accessible.
    """
    def __init__(self):
        # Get ComfyUI's main output directory. Files will be saved relative to this.
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output" # Not strictly necessary but good practice for output nodes

    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the input types for the node.
        - audio: The raw audio data (waveform_tensor, sample_rate).
        - filename_prefix: String to determine output file name and subfolder.
                           Default: "audio/comfy_audio".
        - bitrate: MP3 bitrate for encoding.
        """
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "audio/comfy_audio"}),
                "bitrate": (["128k", "192k", "256k", "320k", "VBR (q4 default)"], {"default": "192k"}),
            },
        }

    RETURN_TYPES = ()  # This node does not pass data to downstream nodes.
    FUNCTION = "save_audio_as_mp3"  # Method to execute node logic.
    OUTPUT_NODE = True  # Indicates node produces an output file.
    CATEGORY = "audio/save"  # Category in ComfyUI node menu.

    def save_audio_as_mp3(self, audio, filename_prefix, bitrate):
        """
        Saves the input audio waveform as an MP3 file.

        Args:
            audio (tuple): Contains (waveform_tensor, sample_rate).
                - waveform_tensor (torch.Tensor): Audio data (batch, channels, samples).
                                                 Expected float32, range [-1.0, 1.0].
                - sample_rate (int): Audio sample rate (e.g., 44100).
            filename_prefix (str): Prefix for output filename, with path patterns.
            bitrate (str): Target bitrate for MP3 encoding (e.g., "192k").
                           If "VBR (q4 default)", uses ffmpeg's default VBR quality (often -q:a 4).

        Returns:
            dict: UI feedback with the path of the saved file.
        """
        waveform_tensor, sample_rate = audio

        if waveform_tensor is None:
            print("[SaveAudioMP3] Warning: No audio data received.")
            return {"ui": {"text": ["No audio data to save."]}}

        # Process the first audio item if batch_size > 1.
        waveform = waveform_tensor[0].cpu()  # Shape: (channels, samples). Move to CPU.

        # Convert audio for pydub: PyTorch tensor (float32) -> NumPy array (int16)
        # pydub expects interleaved samples for multi-channel audio.
        # 1. Transpose from (channels, samples) to (samples, channels)
        waveform_np = waveform.numpy().T
        
        # 2. Scale to int16 range and convert type.
        #    astype() creates a new C-contiguous array by default.
        audio_data_int16 = (waveform_np * 32767).astype(np.int16)

        num_channels = audio_data_int16.shape[1] if audio_data_int16.ndim > 1 else 1
        
        # Create pydub AudioSegment
        try:
            audio_segment = AudioSegment(
                data=audio_data_int16.tobytes(),
                sample_width=2,  # 2 bytes for int16
                frame_rate=sample_rate,
                channels=num_channels
            )
        except Exception as e:
            print(f"[SaveAudioMP3] Error creating AudioSegment with pydub: {e}")
            print("[SaveAudioMP3] This might be due to ffmpeg not being installed or found in PATH.")
            print("[SaveAudioMP3] Please ensure ffmpeg is correctly installed and accessible.")
            return {"ui": {"text": [f"Error creating audio segment: {e}"]}}

        # Determine output path using ComfyUI's folder_paths utility.
        num_samples = waveform.shape[1]
        num_channels_for_path_arg = waveform.shape[0] # Used as a placeholder

        (
            full_output_folder,
            filename,
            counter,
            subfolder,
            _, 
        ) = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, num_samples, num_channels_for_path_arg
        )
        
        file = f"{filename}_{counter:05}.mp3"
        full_path = os.path.join(full_output_folder, file)

        export_parameters = {}
        if bitrate == "VBR (q4 default)":
            # Using ffmpeg's -q:a parameter for VBR. 
            # A common good quality VBR setting is -q:a 4.
            # pydub allows passing extra parameters to ffmpeg.
            export_parameters['parameters'] = ["-q:a", "4"] 
        else:
            export_parameters['bitrate'] = bitrate
        
        try:
            audio_segment.export(full_path, format="mp3", **export_parameters)
        except Exception as e:
            print(f"[SaveAudioMP3] Error exporting MP3 with pydub: {e}")
            print(f"[SaveAudioMP3] Attempted to save to: {full_path} with parameters: {export_parameters}")
            print("[SaveAudioMP3] This might be due to ffmpeg issues or problems with audio data.")
            return {"ui": {"text": [f"Error saving MP3: {e}"]}}
        
        ui_path = os.path.join(subfolder, file) if subfolder else file
        print(f"[SaveAudioMP3] Saved MP3 to: {full_path}")
        return {"ui": {"text": [f"Saved MP3 to: {ui_path}"], "filename": [ui_path]}}
