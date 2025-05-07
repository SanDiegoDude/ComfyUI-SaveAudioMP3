# ComfyUI_SaveAudioMP3/save_audio_mp3_node.py

import os
import torch
import numpy as np
from pydub import AudioSegment
import folder_paths # Primary module for path and pattern handling
import re
# No specific import from comfy.utils for path formatting needed here

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
                # For folder_paths.get_save_image_path to resolve patterns like [seed], [steps], etc.,
                # it might need access to prompt or extra_pnginfo.
                # These are standard names used by SaveImage.
                "prompt": ("PROMPT",), 
                "extra_pnginfo": ("EXTRA_PNGINFO",)
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
        # Allow dots and hyphens as they are common in filenames.
        sanitized_name = re.sub(r'[^a-zA-Z0-9_.-]', '', name_string)
        return sanitized_name[:100]

    # Added prompt and extra_pnginfo as optional args to match how SaveImage passes them
    # to folder_paths.get_save_image_path for pattern resolution.
    def save_audio_as_mp3(self, audio_data, filename_prefix, bitrate, 
                          song_name=None, prompt=None, extra_pnginfo=None): # Ensure these are passed
        if not isinstance(audio_data, dict):
            print(f"[SaveAudioMP3] Error: Expected a dictionary for audio_data. Got: {type(audio_data)}")
            return {"ui": {"text": ["Error: Invalid audio data format (expected dict)."]}}

        audio_dict = audio_data 
        if "waveform" not in audio_dict or "sample_rate" not in audio_dict:
            print(f"[SaveAudioMP3] Error: Audio dictionary missing 'waveform' or 'sample_rate' key. Keys: {audio_dict.keys()}")
            return {"ui": {"text": ["Error: Audio dictionary incomplete."]}}

        source_waveform_data = audio_dict["waveform"]
        sample_rate = audio_dict["sample_rate"]
        
        # print(f"[SaveAudioMP3] Received audio. Waveform data type: {type(source_waveform_data)}, Sample rate: {sample_rate}")

        audio_segment = None
        # --- AudioSegment Creation Logic (condensed for brevity, use your previous full logic) ---
        if isinstance(source_waveform_data, torch.Tensor):
            waveform_tensor = source_waveform_data; waveform_single_item = None
            if waveform_tensor.ndim == 3 and waveform_tensor.shape[0]>0: waveform_single_item=waveform_tensor[0].cpu()
            elif waveform_tensor.ndim == 2: waveform_single_item=waveform_tensor.cpu()
            else: msg=f"Error: Tensor unexpected dims: {waveform_tensor.shape}."; print(f"[SaveAudioMP3] {msg}"); return {"ui": {"text": [msg]}}
            waveform_np=waveform_single_item.numpy().T; audio_data_int16=(waveform_np*32767).astype(np.int16)
            num_channels=audio_data_int16.shape[1] if audio_data_int16.ndim>1 else 1
            try: audio_segment=AudioSegment(data=audio_data_int16.tobytes(),sample_width=2,frame_rate=sample_rate,channels=num_channels)
            except Exception as e: msg=f"Error creating AudioSegment: {e}."; print(f"[SaveAudioMP3] {msg}"); return {"ui": {"text": [msg]}}
        elif isinstance(source_waveform_data, str):
            audio_file_path = source_waveform_data
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
        else: msg=f"Error: Waveform unsupported: {type(source_waveform_data)}."; print(f"[SaveAudioMP3] {msg}"); return {"ui": {"text": [msg]}}
        if audio_segment is None: msg="Error: AudioSegment not prepared."; print(f"[SaveAudioMP3] {msg}"); return {"ui": {"text": [msg]}}
        # --- End AudioSegment Creation ---

        # --- Filename and Path Generation ---
        # 1. Combine filename_prefix and optional song_name
        current_prefix_template = filename_prefix # This is the raw prefix from input
        if song_name and song_name.strip():
            sanitized_song_name = self._sanitize_filename_component(song_name.strip())
            if sanitized_song_name:
                if current_prefix_template.endswith(('/', '\\')):
                    current_prefix_template = f"{current_prefix_template}{sanitized_song_name}"
                else:
                    dir_part, base_part = os.path.split(current_prefix_template)
                    if base_part: current_prefix_template = os.path.join(dir_part, f"{base_part}_{sanitized_song_name}")
                    else: current_prefix_template = os.path.join(dir_part, sanitized_song_name)
        
        print(f"[SaveAudioMP3] Filename prefix template passed to get_save_image_path: '{current_prefix_template}'")

        # 2. Use folder_paths.get_save_image_path. It is responsible for all pattern resolution.
        # Pass prompt and extra_pnginfo to it, as SaveImage does, in case they are needed for patterns.
        num_samples_for_path = int(audio_segment.duration_seconds * audio_segment.frame_rate)
        num_channels_for_path = audio_segment.channels

        # This call should resolve ALL patterns in current_prefix_template and create directories.
        # The returned components (folder, base_filename, subfolder) should be fully resolved.
        actual_disk_output_folder, actual_disk_base_filename, counter, actual_ui_subfolder, filename_prefix_resolved_by_util = folder_paths.get_save_image_path(
            current_prefix_template, self.output_dir, num_samples_for_path, num_channels_for_path, # Main args
            prompt, extra_pnginfo # Args for pattern resolution, like in SaveImage
        )
        
        # 3. Construct final filename and path for saving
        mp3_final_filename_on_disk = f"{actual_disk_base_filename}_{counter:05}.mp3"
        full_mp3_path_to_save_on_disk = os.path.join(actual_disk_output_folder, mp3_final_filename_on_disk)
        # --- End Filename and Path Generation ---

        export_parameters = {}
        if bitrate == "VBR (q4 default)": export_parameters['parameters'] = ["-q:a", "4"] 
        else: export_parameters['bitrate'] = bitrate
        
        try:
            # Log the path we are actually trying to save to
            print(f"[SaveAudioMP3] Attempting to save to actual disk path: '{full_mp3_path_to_save_on_disk}' with parameters: {export_parameters}")
            
            # Ensure the directory exists (get_save_image_path should handle this)
            # os.makedirs(actual_disk_output_folder, exist_ok=True) # Defensive, but get_save_image_path should do it.
            
            audio_segment.export(full_mp3_path_to_save_on_disk, format="mp3", **export_parameters)
        except Exception as e:
            msg = f"Error exporting MP3 to '{full_mp3_path_to_save_on_disk}': {e}."
            print(f"[SaveAudioMP3] {msg}")
            print(f"[SaveAudioMP3] Details for debugging path issue:")
            print(f"    - Original filename_prefix input: '{filename_prefix}'")
            print(f"    - Optional song_name input: '{song_name}'")
            print(f"    - Template passed to get_save_image_path: '{current_prefix_template}'")
            print(f"    - Outputs from get_save_image_path: folder='{actual_disk_output_folder}', base_filename='{actual_disk_base_filename}', subfolder='{actual_ui_subfolder}'")
            print(f"    - Resolved prefix from get_save_image_path: '{filename_prefix_resolved_by_util}'") # See what it thought it resolved
            return {"ui": {"text": [msg]}}
        
        print(f"[SaveAudioMP3] Successfully saved MP3 to: '{full_mp3_path_to_save_on_disk}'")
        
        results_list = [{
            "filename": mp3_final_filename_on_disk,
            "subfolder": actual_ui_subfolder,
            "type": self.type
        }]
        
        ui_payload = {"audio": results_list}
        # print(f"[SaveAudioMP3] Returning UI data for player: { {'ui': ui_payload} }")
        return {"ui": ui_payload}
