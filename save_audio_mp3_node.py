# ComfyUI_SaveAudioMP3/save_audio_mp3_node.py

# ------------------------------------------------------------------------------------
# User Prerequisites for ComfyUI_SaveAudioMP3:
# 1. Install pydub:
#    In your ComfyUI's Python environment, run: pip install pydub
# 2. Install ffmpeg:
#    ffmpeg must be installed on your system and accessible in the PATH.
#    Details in README or comments in __init__.py
# ------------------------------------------------------------------------------------

import os
import torch
import numpy as np
from pydub import AudioSegment # For MP3 conversion. Requires ffmpeg.
import folder_paths # ComfyUI utility for managing paths

class SaveAudioMP3:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output" 

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "audio/comfy_audio"}),
                "bitrate": (["128k", "192k", "256k", "320k", "VBR (q4 default)"], {"default": "192k"}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_audio_as_mp3"
    OUTPUT_NODE = True
    CATEGORY = "audio/save"

    def save_audio_as_mp3(self, audio, filename_prefix, bitrate):
        if not (isinstance(audio, tuple) and len(audio) == 2):
            print(f"[SaveAudioMP3] Error: Audio input is not a 2-element tuple. Got: {type(audio)}")
            return {"ui": {"text": ["Error: Invalid audio input format."]}}

        source_waveform_data, sample_rate = audio
        
        print(f"[SaveAudioMP3] Received audio. Source data type: {type(source_waveform_data)}, Sample rate: {sample_rate} (type: {type(sample_rate)})")

        audio_segment = None

        if isinstance(source_waveform_data, torch.Tensor):
            waveform_tensor = source_waveform_data
            print(f"[SaveAudioMP3] Source data is a Tensor. Shape: {waveform_tensor.shape}, Dtype: {waveform_tensor.dtype}")

            waveform_single_item = None
            if waveform_tensor.ndim == 3 and waveform_tensor.shape[0] > 0: # Batch, Channels, Samples
                print(f"[SaveAudioMP3] Tensor is 3D (Batch, Channels, Samples). Processing first item in batch.")
                waveform_single_item = waveform_tensor[0].cpu() 
            elif waveform_tensor.ndim == 2: # Channels, Samples
                print(f"[SaveAudioMP3] Tensor is 2D (Channels, Samples). Assuming single audio item.")
                waveform_single_item = waveform_tensor.cpu()
            else:
                msg = f"Error: Tensor has unexpected dimensions: {waveform_tensor.shape}. Expected 2D (C,S) or 3D (B,C,S)."
                print(f"[SaveAudioMP3] {msg}")
                return {"ui": {"text": [msg]}}

            # Convert PyTorch tensor (float32, range [-1.0, 1.0]) to NumPy array (int16)
            # pydub expects interleaved samples for multi-channel audio.
            # 1. Transpose from (channels, samples) to (samples, channels)
            waveform_np = waveform_single_item.numpy().T 
            # 2. Scale to int16 range and convert type.
            audio_data_int16 = (waveform_np * 32767).astype(np.int16)
            num_channels = audio_data_int16.shape[1] if audio_data_int16.ndim > 1 else 1
            
            try:
                audio_segment = AudioSegment(
                    data=audio_data_int16.tobytes(), # Raw audio data as bytes
                    sample_width=2,  # 2 bytes for int16
                    frame_rate=sample_rate,
                    channels=num_channels
                )
            except Exception as e:
                msg = f"Error creating AudioSegment from Tensor: {e}. Ensure ffmpeg is installed and in PATH if pydub requires it for internal operations."
                print(f"[SaveAudioMP3] {msg}")
                return {"ui": {"text": [msg]}}

        elif isinstance(source_waveform_data, str):
            audio_file_path = source_waveform_data
            print(f"[SaveAudioMP3] Source data is a String, assuming it's an audio file path: {audio_file_path}")

            # Try to resolve path if it's not absolute (e.g., relative to input, output, or ComfyUI root)
            if not os.path.isabs(audio_file_path):
                resolved_path = None
                # Check common ComfyUI folders
                for folder_type in ["input", "output", "temp"]: 
                    current_path = folder_paths.get_full_path(folder_type, audio_file_path)
                    if current_path and os.path.exists(current_path):
                        resolved_path = current_path
                        break
                if resolved_path:
                    audio_file_path = resolved_path
                elif not os.path.exists(audio_file_path): # Fallback: check if path is relative to ComfyUI root
                    msg = f"Error: Audio file path not found: {source_waveform_data} (also not found in input/output/temp folders or ComfyUI root)"
                    print(f"[SaveAudioMP3] {msg}")
                    return {"ui": {"text": [msg]}}
            
            print(f"[SaveAudioMP3] Attempting to load audio from resolved path: {audio_file_path}")
            try:
                audio_segment = AudioSegment.from_file(audio_file_path)
                # Optional: Resample if the provided sample_rate (from AUDIO tuple) is different
                # and considered authoritative. ACE-Step should ideally provide the correct rate.
                if audio_segment.frame_rate != sample_rate:
                    print(f"[SaveAudioMP3] Warning: File sample rate ({audio_segment.frame_rate} Hz) differs from provided rate ({sample_rate} Hz). Resampling to {sample_rate} Hz.")
                    audio_segment = audio_segment.set_frame_rate(sample_rate)
            except Exception as e:
                msg = f"Error loading audio file '{audio_file_path}' with pydub: {e}. Ensure ffmpeg is installed and in PATH."
                print(f"[SaveAudioMP3] {msg}")
                return {"ui": {"text": [msg]}}
        else:
            msg = f"Error: Audio data type not supported: {type(source_waveform_data)}. Expected Tensor or path String."
            print(f"[SaveAudioMP3] {msg}")
            return {"ui": {"text": [msg]}}

        if audio_segment is None:
            # This should ideally be caught by earlier returns, but as a safeguard
            msg = "Error: AudioSegment could not be prepared for saving. Unknown reason."
            print(f"[SaveAudioMP3] {msg}")
            return {"ui": {"text": [msg]}}

        # Determine output path for the MP3
        # Use audio_segment properties for get_save_image_path's "image-like" dimension arguments
        num_samples_for_path = int(audio_segment.duration_seconds * audio_segment.frame_rate)
        num_channels_for_path = audio_segment.channels

        (
            full_output_folder, # Absolute path to the directory for the file
            base_filename,      # Base filename (prefix part, without counter/extension)
            counter,            # Counter for uniqueness
            subfolder,          # Subfolder part, relative to self.output_dir
            _                   # filename_prefix_path (full prefix if it was absolute)
        ) = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, num_samples_for_path, num_channels_for_path
        )
        
        # Construct final filename and path
        mp3_filename = f"{base_filename}_{counter:05}.mp3"
        full_mp3_path = os.path.join(full_output_folder, mp3_filename)

        # Prepare export parameters (bitrate or VBR)
        export_parameters = {}
        if bitrate == "VBR (q4 default)":
            export_parameters['parameters'] = ["-q:a", "4"] # ffmpeg VBR quality scale -q:a 4 (good quality)
        else:
            export_parameters['bitrate'] = bitrate # e.g., "192k"
        
        # Export the audio to MP3
        try:
            print(f"[SaveAudioMP3] Exporting to: {full_mp3_path} with parameters: {export_parameters}")
            audio_segment.export(full_mp3_path, format="mp3", **export_parameters)
        except Exception as e:
            msg = f"Error exporting MP3 with pydub to '{full_mp3_path}': {e}. Check ffmpeg installation and write permissions."
            print(f"[SaveAudioMP3] {msg}")
            return {"ui": {"text": [msg]}}
        
        print(f"[SaveAudioMP3] Successfully saved MP3: {full_mp3_path}")
        
        # Return structure for ComfyUI to display an audio player
        return {
            "ui": {
                "audio": [
                    {
                        "filename": mp3_filename,    # e.g., "comfy_audio_00001.mp3"
                        "subfolder": subfolder,      # e.g., "audio" or "" if prefix is absolute/no_subdir
                        "type": self.type,           # "output"
                        "format": "audio/mpeg"       # MIME type for MP3
                    }
                ]
                # You can also add "text": [message] if you want additional text display
            }
        }
