# ComfyUI_SaveAudioMP3/save_audio_mp3_node.py

import os
import torch
import numpy as np
from pydub import AudioSegment
import folder_paths

class SaveAudioMP3:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output" 

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_data": ("AUDIO",), # Changed input name for clarity
                "filename_prefix": ("STRING", {"default": "audio/comfy_audio"}),
                "bitrate": (["128k", "192k", "256k", "320k", "VBR (q4 default)"], {"default": "192k"}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_audio_as_mp3"
    OUTPUT_NODE = True
    CATEGORY = "audio/save" # Or your preferred category

    def save_audio_as_mp3(self, audio_data, filename_prefix, bitrate):
        # The AUDIO input from ACE-Step nodes is a tuple containing a single dictionary:
        # e.g., ({"waveform": tensor, "sample_rate": int},)
        
        if not isinstance(audio_data, tuple) or not audio_data:
            print(f"[SaveAudioMP3] Error: Expected a non-empty tuple for audio_data. Got: {type(audio_data)}")
            return {"ui": {"text": ["Error: Invalid audio data format (expected tuple)."]}}

        audio_dict = audio_data[0] # Get the dictionary from the tuple

        if not isinstance(audio_dict, dict):
            print(f"[SaveAudioMP3] Error: Expected a dictionary inside the audio_data tuple. Got: {type(audio_dict)}")
            return {"ui": {"text": ["Error: Invalid audio data format (expected dict)."]}}

        if "waveform" not in audio_dict or "sample_rate" not in audio_dict:
            print(f"[SaveAudioMP3] Error: Audio dictionary missing 'waveform' or 'sample_rate' key. Keys: {audio_dict.keys()}")
            return {"ui": {"text": ["Error: Audio dictionary incomplete."]}}

        source_waveform_data = audio_dict["waveform"]
        sample_rate = audio_dict["sample_rate"]
        
        print(f"[SaveAudioMP3] Received audio. Waveform data type: {type(source_waveform_data)}, Sample rate: {sample_rate}")

        audio_segment = None

        if isinstance(source_waveform_data, torch.Tensor):
            waveform_tensor = source_waveform_data
            print(f"[SaveAudioMP3] Source data is a Tensor. Shape: {waveform_tensor.shape}, Dtype: {waveform_tensor.dtype}")

            # ACE-Step seems to return: audio_output[0][0].unsqueeze(0), so it's likely [1, Channels, Samples]
            # We need to handle if it's [Batch, Channels, Samples] or just [Channels, Samples]
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

            waveform_np = waveform_single_item.numpy().T 
            audio_data_int16 = (waveform_np * 32767).astype(np.int16)
            num_channels = audio_data_int16.shape[1] if audio_data_int16.ndim > 1 else 1
            
            try:
                audio_segment = AudioSegment(
                    data=audio_data_int16.tobytes(),
                    sample_width=2,
                    frame_rate=sample_rate,
                    channels=num_channels
                )
            except Exception as e:
                msg = f"Error creating AudioSegment from Tensor: {e}."
                print(f"[SaveAudioMP3] {msg}")
                return {"ui": {"text": [msg]}}

        elif isinstance(source_waveform_data, str): # Handling if AUDIO type could be a path string from other nodes
            audio_file_path = source_waveform_data
            print(f"[SaveAudioMP3] Source waveform data is a String, assuming it's an audio file path: {audio_file_path}")
            
            if not os.path.isabs(audio_file_path):
                resolved_path = None
                for folder_type in ["input", "output", "temp"]: 
                    current_path = folder_paths.get_full_path(folder_type, audio_file_path)
                    if current_path and os.path.exists(current_path):
                        resolved_path = current_path
                        break
                if resolved_path: audio_file_path = resolved_path
                elif not os.path.exists(audio_file_path):
                    msg = f"Error: Audio file path not found: {source_waveform_data}"
                    print(f"[SaveAudioMP3] {msg}")
                    return {"ui": {"text": [msg]}}
            
            print(f"[SaveAudioMP3] Attempting to load audio from resolved path: {audio_file_path}")
            try:
                audio_segment = AudioSegment.from_file(audio_file_path)
                if audio_segment.frame_rate != sample_rate:
                    print(f"[SaveAudioMP3] Warning: File sample rate ({audio_segment.frame_rate} Hz) differs from provided rate ({sample_rate} Hz). Resampling to {sample_rate} Hz.")
                    audio_segment = audio_segment.set_frame_rate(sample_rate)
            except Exception as e:
                msg = f"Error loading audio file '{audio_file_path}': {e}. Ensure ffmpeg is installed."
                print(f"[SaveAudioMP3] {msg}")
                return {"ui": {"text": [msg]}}
        else:
            msg = f"Error: Waveform data type in audio_dict not supported: {type(source_waveform_data)}. Expected Tensor or path String."
            print(f"[SaveAudioMP3] {msg}")
            return {"ui": {"text": [msg]}}

        if audio_segment is None:
            msg = "Error: AudioSegment could not be prepared."
            print(f"[SaveAudioMP3] {msg}")
            return {"ui": {"text": [msg]}}

        num_samples_for_path = int(audio_segment.duration_seconds * audio_segment.frame_rate)
        num_channels_for_path = audio_segment.channels

        (full_output_folder, base_filename, counter, subfolder, _) = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, num_samples_for_path, num_channels_for_path
        )
        
        mp3_filename = f"{base_filename}_{counter:05}.mp3"
        full_mp3_path = os.path.join(full_output_folder, mp3_filename)

        export_parameters = {}
        if bitrate == "VBR (q4 default)":
            export_parameters['parameters'] = ["-q:a", "4"] 
        else:
            export_parameters['bitrate'] = bitrate
        
        try:
            print(f"[SaveAudioMP3] Exporting to: {full_mp3_path} with parameters: {export_parameters}")
            audio_segment.export(full_mp3_path, format="mp3", **export_parameters)
        except Exception as e:
            msg = f"Error exporting MP3 to '{full_mp3_path}': {e}."
            print(f"[SaveAudioMP3] {msg}")
            return {"ui": {"text": [msg]}}
        
        print(f"[SaveAudioMP3] Successfully saved MP3: {full_mp3_path}")
        
        return {
            "ui": {
                "audio": [{
                    "filename": mp3_filename,
                    "subfolder": subfolder,
                    "type": self.type,
                    "format": "audio/mpeg" 
                }]
            }
        }
