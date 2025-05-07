# ComfyUI_SaveAudioMP3/save_audio_mp3_node.py

import os
import torch
import numpy as np
from pydub import AudioSegment
import folder_paths
import re # For sanitizing song_name

class SaveAudioMP3:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output" 

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_data": ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "audio/comfy_audio"}),
                "bitrate": (["128k", "192k", "256k", "320k", "VBR (q4 default)"], {"default": "320k"}),
            },
            "optional": { # New optional input for song name
                "song_name": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_audio_as_mp3"
    OUTPUT_NODE = True
    CATEGORY = "audio/save"

    def _sanitize_filename_component(self, name_string):
        if not name_string:
            return ""
        # Replace spaces and common problematic characters with underscores
        name_string = re.sub(r'[\s/:*?"<>|]+', '_', name_string)
        # Remove any characters not in a safe list (alphanumeric, underscore, hyphen)
        # This is a more restrictive approach for maximum safety.
        sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '', name_string)
        # Limit length to prevent overly long filenames
        return sanitized_name[:100]


    def save_audio_as_mp3(self, audio_data, filename_prefix, bitrate, song_name=None):
        if not isinstance(audio_data, dict):
            print(f"[SaveAudioMP3] Error: Expected a dictionary for audio_data. Got: {type(audio_data)}")
            return {"ui": {"text": ["Error: Invalid audio data format (expected dict)."]}}

        audio_dict = audio_data 
        if "waveform" not in audio_dict or "sample_rate" not in audio_dict:
            print(f"[SaveAudioMP3] Error: Audio dictionary missing 'waveform' or 'sample_rate' key. Keys: {audio_dict.keys()}")
            return {"ui": {"text": ["Error: Audio dictionary incomplete."]}}

        source_waveform_data = audio_dict["waveform"]
        sample_rate = audio_dict["sample_rate"]
        
        print(f"[SaveAudioMP3] Received audio. Waveform data type: {type(source_waveform_data)}, Sample rate: {sample_rate}")

        # --- AudioSegment Creation (remains the same) ---
        audio_segment = None
        if isinstance(source_waveform_data, torch.Tensor):
            waveform_tensor = source_waveform_data
            print(f"[SaveAudioMP3] Source data is a Tensor. Shape: {waveform_tensor.shape}, Dtype: {waveform_tensor.dtype}")
            waveform_single_item = None
            if waveform_tensor.ndim == 3 and waveform_tensor.shape[0] > 0: 
                waveform_single_item = waveform_tensor[0].cpu() 
            elif waveform_tensor.ndim == 2: 
                waveform_single_item = waveform_tensor.cpu()
            else:
                msg = f"Error: Tensor has unexpected dimensions: {waveform_tensor.shape}."
                print(f"[SaveAudioMP3] {msg}")
                return {"ui": {"text": [msg]}}
            waveform_np = waveform_single_item.numpy().T 
            audio_data_int16 = (waveform_np * 32767).astype(np.int16)
            num_channels = audio_data_int16.shape[1] if audio_data_int16.ndim > 1 else 1
            try:
                audio_segment = AudioSegment(data=audio_data_int16.tobytes(), sample_width=2, frame_rate=sample_rate, channels=num_channels)
            except Exception as e:
                msg = f"Error creating AudioSegment from Tensor: {e}."
                print(f"[SaveAudioMP3] {msg}")
                return {"ui": {"text": [msg]}}
        elif isinstance(source_waveform_data, str): 
            # ... (loading from path logic - remains the same) ...
            audio_file_path = source_waveform_data
            # (Path resolution and loading logic as before)
            if not os.path.isabs(audio_file_path): # Simplified for brevity, use previous full logic
                resolved_path = folder_paths.get_full_path("input", audio_file_path) # Check input first
                if not resolved_path or not os.path.exists(resolved_path) :
                     resolved_path = folder_paths.get_full_path("output", audio_file_path) # Then output
                     if not resolved_path or not os.path.exists(resolved_path):
                          resolved_path = folder_paths.get_full_path("temp", audio_file_path) # Then temp
                if resolved_path and os.path.exists(resolved_path): audio_file_path = resolved_path
                elif not os.path.exists(audio_file_path): # Fallback to check as is
                    msg = f"Error: Audio file path not found: {source_waveform_data}"
                    print(f"[SaveAudioMP3] {msg}"); return {"ui": {"text": [msg]}}
            try:
                audio_segment = AudioSegment.from_file(audio_file_path)
                if audio_segment.frame_rate != sample_rate:
                    audio_segment = audio_segment.set_frame_rate(sample_rate)
            except Exception as e:
                msg = f"Error loading audio file '{audio_file_path}': {e}."
                print(f"[SaveAudioMP3] {msg}"); return {"ui": {"text": [msg]}}
        else:
            msg = f"Error: Waveform data type unsupported: {type(source_waveform_data)}."
            print(f"[SaveAudioMP3] {msg}"); return {"ui": {"text": [msg]}}
        if audio_segment is None:
            msg = "Error: AudioSegment could not be prepared."
            print(f"[SaveAudioMP3] {msg}"); return {"ui": {"text": [msg]}}
        # --- End AudioSegment Creation ---

        # --- Filename and Path Generation ---
        # Sanitize song_name if provided and append to filename_prefix
        current_filename_prefix = filename_prefix
        if song_name and song_name.strip():
            sanitized_song_name = self._sanitize_filename_component(song_name.strip())
            if sanitized_song_name:
                # Append to the filename part of the prefix
                if current_filename_prefix.endswith(('/', '\\')):
                    current_filename_prefix = f"{current_filename_prefix}{sanitized_song_name}"
                else:
                    # If no trailing slash, append with an underscore
                    # To avoid merging with last part of prefix if it's not a directory
                    # e.g. "myprefix" + "song" -> "myprefix_song"
                    # e.g. "mydir/myprefix" + "song" -> "mydir/myprefix_song"
                    parts = os.path.split(current_filename_prefix)
                    if parts[1]: # if there is a filename component
                        current_filename_prefix = os.path.join(parts[0], f"{parts[1]}_{sanitized_song_name}")
                    else: # only a directory component
                        current_filename_prefix = os.path.join(parts[0], sanitized_song_name)


        print(f"[SaveAudioMP3] Effective filename_prefix for path generation: {current_filename_prefix}")

        num_samples_for_path = int(audio_segment.duration_seconds * audio_segment.frame_rate)
        num_channels_for_path = audio_segment.channels

        # get_save_image_path is expected to resolve patterns in current_filename_prefix
        # and create the necessary directories.
        # It should return:
        # - resolved_full_output_folder: Absolute path to the (resolved) directory for the file
        # - resolved_base_filename: Base filename (resolved prefix part, without counter/extension)
        # - counter: Counter for uniqueness
        # - resolved_subfolder: Subfolder part (resolved), relative to self.output_dir
        resolved_full_output_folder, resolved_base_filename, counter, resolved_subfolder, _ = folder_paths.get_save_image_path(
            current_filename_prefix, self.output_dir, num_samples_for_path, num_channels_for_path
        )
        
        # This filename should be based on the *resolved* base filename
        mp3_final_filename = f"{resolved_base_filename}_{counter:05}.mp3"
        # This path should be fully resolved and where the file is actually saved
        full_mp3_path_to_save = os.path.join(resolved_full_output_folder, mp3_final_filename)
        # --- End Filename and Path Generation ---

        export_parameters = {}
        if bitrate == "VBR (q4 default)":
            export_parameters['parameters'] = ["-q:a", "4"] 
        else:
            export_parameters['bitrate'] = bitrate
        
        try:
            print(f"[SaveAudioMP3] Attempting to save to (resolved path): {full_mp3_path_to_save} with parameters: {export_parameters}")
            audio_segment.export(full_mp3_path_to_save, format="mp3", **export_parameters)
        except Exception as e:
            # If export fails, it might be due to the path still containing unresolved patterns
            # or other filesystem issues.
            msg = f"Error exporting MP3 to '{full_mp3_path_to_save}': {e}. Check path and ffmpeg."
            print(f"[SaveAudioMP3] {msg}")
            # Also print the inputs to get_save_image_path for debugging the pattern issue
            print(f"[SaveAudioMP3] Inputs to get_save_image_path were: prefix='{current_filename_prefix}', output_dir='{self.output_dir}'")
            print(f"[SaveAudioMP3] Outputs from get_save_image_path were: folder='{resolved_full_output_folder}', base_filename='{resolved_base_filename}', subfolder='{resolved_subfolder}'")
            return {"ui": {"text": [msg]}}
        
        print(f"[SaveAudioMP3] Successfully saved MP3 to: {full_mp3_path_to_save}")
        
        # --- Player UI Payload ---
        # The filename and subfolder for the UI MUST match how ComfyUI's /view endpoint will find them.
        # These should be based on the *resolved* components.
        results_list = [{
            "filename": mp3_final_filename,       # e.g., "ComfyUI_20230101-Ace-Audio_MySong_00001.mp3"
            "subfolder": resolved_subfolder,     # e.g., "2023-01-01" (relative to output_dir)
            "type": self.type                    # "output"
        }]
        
        ui_payload = {"audio": results_list}
        
        print(f"[SaveAudioMP3] Returning UI data for player: { {'ui': ui_payload} }")
        
        return {"ui": ui_payload}
