# ComfyUI_SaveAudioMP3/save_audio_mp3_node.py

import os
import torch
import numpy as np
from pydub import AudioSegment
import folder_paths
import re
import comfy.utils # For ComfyUI-specific pattern resolution

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
            "optional": {
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
        name_string = re.sub(r'[\s/:*?"<>|]+', '_', name_string)
        sanitized_name = re.sub(r'[^a-zA-Z0-9_.-]', '', name_string) # Allow dots and hyphens
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

        audio_segment = None
        # --- AudioSegment Creation Logic (from your previous working version) ---
        if isinstance(source_waveform_data, torch.Tensor):
            waveform_tensor = source_waveform_data
            # print(f"[SaveAudioMP3] Source data is a Tensor. Shape: {waveform_tensor.shape}, Dtype: {waveform_tensor.dtype}")
            waveform_single_item = None
            if waveform_tensor.ndim == 3 and waveform_tensor.shape[0] > 0: 
                # print(f"[SaveAudioMP3] Tensor is 3D (Batch, Channels, Samples). Processing first item in batch.")
                waveform_single_item = waveform_tensor[0].cpu() 
            elif waveform_tensor.ndim == 2: 
                # print(f"[SaveAudioMP3] Tensor is 2D (Channels, Samples). Assuming single audio item.")
                waveform_single_item = waveform_tensor.cpu()
            else:
                msg = f"Error: Tensor has unexpected dimensions: {waveform_tensor.shape}. Expected 2D (C,S) or 3D (B,C,S)."
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
            audio_file_path = source_waveform_data
            # (Path resolution and loading logic as before - simplified here for brevity)
            if not os.path.isabs(audio_file_path):
                resolved_path = folder_paths.get_full_path("input", audio_file_path) or \
                                folder_paths.get_full_path("output", audio_file_path) or \
                                folder_paths.get_full_path("temp", audio_file_path)
                if resolved_path and os.path.exists(resolved_path): audio_file_path = resolved_path
                elif not os.path.exists(audio_file_path):
                    msg = f"Error: Audio file path not found: {source_waveform_data}"; print(f"[SaveAudioMP3] {msg}"); return {"ui": {"text": [msg]}}
            try:
                audio_segment = AudioSegment.from_file(audio_file_path)
                if audio_segment.frame_rate != sample_rate: audio_segment = audio_segment.set_frame_rate(sample_rate)
            except Exception as e:
                msg = f"Error loading audio file '{audio_file_path}': {e}."; print(f"[SaveAudioMP3] {msg}"); return {"ui": {"text": [msg]}}
        else:
            msg = f"Error: Waveform data type unsupported: {type(source_waveform_data)}."; print(f"[SaveAudioMP3] {msg}"); return {"ui": {"text": [msg]}}
        if audio_segment is None:
            msg = "Error: AudioSegment could not be prepared."; print(f"[SaveAudioMP3] {msg}"); return {"ui": {"text": [msg]}}
        # --- End AudioSegment Creation ---

        # --- Filename and Path Generation ---
        # 1. Combine filename_prefix and optional song_name
        current_prefix_template = filename_prefix
        if song_name and song_name.strip():
            sanitized_song_name = self._sanitize_filename_component(song_name.strip())
            if sanitized_song_name:
                # Append to the filename part of the prefix template
                # Ensure there's a separator if needed, or just append if template ends with /
                if current_prefix_template.endswith(('/', '\\')):
                    current_prefix_template = f"{current_prefix_template}{sanitized_song_name}"
                else:
                    # Smart append with underscore if there's a base filename part
                    dir_part, base_part = os.path.split(current_prefix_template)
                    if base_part: # "audio/myprefix" + "song" -> "audio/myprefix_song"
                        current_prefix_template = os.path.join(dir_part, f"{base_part}_{sanitized_song_name}")
                    else: # "audio/" + "song" -> "audio/song"
                        current_prefix_template = os.path.join(dir_part, sanitized_song_name)
        
        print(f"[SaveAudioMP3] Prefix template (with song_name, before ComfyUI pattern resolution): '{current_prefix_template}'")

        # 2. Resolve ComfyUI-specific patterns (e.g., %date:yyyy-MM-dd%)
        # We don't have prompt/extra_pnginfo here, so node_id is not passed.
        # format_path primarily handles date/time and basic node_id patterns.
        resolved_prefix_for_path_util = comfy.utils.format_path(current_prefix_template)
        print(f"[SaveAudioMP3] Prefix after ComfyUI pattern resolution (input to get_save_image_path): '{resolved_prefix_for_path_util}'")

        # 3. Use folder_paths.get_save_image_path to get unique filename components and ensure directory creation
        num_samples_for_path = int(audio_segment.duration_seconds * audio_segment.frame_rate)
        num_channels_for_path = audio_segment.channels

        # This function will create directories based on resolved_prefix_for_path_util
        # and give a unique counter.
        # The returned components (folder, base_filename, subfolder) should be fully resolved.
        actual_disk_output_folder, actual_disk_base_filename, counter, actual_ui_subfolder, _ = folder_paths.get_save_image_path(
            resolved_prefix_for_path_util, self.output_dir, num_samples_for_path, num_channels_for_path
        )
        
        # 4. Construct final filename and path for saving
        mp3_final_filename_on_disk = f"{actual_disk_base_filename}_{counter:05}.mp3"
        full_mp3_path_to_save_on_disk = os.path.join(actual_disk_output_folder, mp3_final_filename_on_disk)
        # --- End Filename and Path Generation ---

        export_parameters = {}
        if bitrate == "VBR (q4 default)": export_parameters['parameters'] = ["-q:a", "4"] 
        else: export_parameters['bitrate'] = bitrate
        
        try:
            print(f"[SaveAudioMP3] Attempting to save to actual disk path: '{full_mp3_path_to_save_on_disk}' with parameters: {export_parameters}")
            # folder_paths.get_save_image_path should have created the directory
            audio_segment.export(full_mp3_path_to_save_on_disk, format="mp3", **export_parameters)
        except Exception as e:
            msg = f"Error exporting MP3 to '{full_mp3_path_to_save_on_disk}': {e}."
            print(f"[SaveAudioMP3] {msg}")
            print(f"[SaveAudioMP3] Details for debugging path issue:")
            print(f"    - Original filename_prefix input: '{filename_prefix}'")
            print(f"    - Optional song_name input: '{song_name}'")
            print(f"    - Template passed to comfy.utils.format_path: '{current_prefix_template}'")
            print(f"    - Prefix after comfy.utils.format_path (passed to get_save_image_path): '{resolved_prefix_for_path_util}'")
            print(f"    - Outputs from get_save_image_path: folder='{actual_disk_output_folder}', base_filename='{actual_disk_base_filename}', subfolder='{actual_ui_subfolder}'")
            return {"ui": {"text": [msg]}}
        
        print(f"[SaveAudioMP3] Successfully saved MP3 to: '{full_mp3_path_to_save_on_disk}'")
        
        # --- Player UI Payload ---
        # Filename for UI is just the file part. Subfolder is relative to self.output_dir.
        # These are directly from get_save_image_path's results after it processed the resolved prefix.
        results_list = [{
            "filename": mp3_final_filename_on_disk, # This is actual_disk_base_filename + _counter + .mp3
            "subfolder": actual_ui_subfolder,        # This is the subfolder relative to output_dir
            "type": self.type
        }]
        
        ui_payload = {"audio": results_list}
        print(f"[SaveAudioMP3] Returning UI data for player: { {'ui': ui_payload} }")
        return {"ui": ui_payload}
