# ComfyUI_SaveAudioMP3/save_audio_mp3_node.py

import os
import torch
import numpy as np
from pydub import AudioSegment
import folder_paths 
import re
import comfy.utils # For ComfyUI-specific pattern resolution like %date:...%

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
                # These are needed for comfy.utils.format_filename to resolve patterns like [seed], [steps], etc.
                "prompt": ("PROMPT",), 
                "extra_pnginfo": ("EXTRA_PNGINFO",)
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_audio_as_mp3"
    OUTPUT_NODE = True
    CATEGORY = "audio/save"

    def _sanitize_filename_component(self, name_string):
        if not name_string: return ""
        name_string = re.sub(r'[\s/:*?"<>|]+', '_', name_string)
        sanitized_name = re.sub(r'[^a-zA-Z0-9_.-]', '', name_string)
        return sanitized_name[:100]

    def save_audio_as_mp3(self, audio_data, filename_prefix, bitrate, 
                          song_name=None, prompt=None, extra_pnginfo=None):
        if not isinstance(audio_data, dict): # Basic input validation
            # ... (error handling as before)
            print(f"[SaveAudioMP3] Error: Expected a dictionary for audio_data. Got: {type(audio_data)}")
            return {"ui": {"text": ["Error: Invalid audio data format (expected dict)."]}}
        audio_dict = audio_data
        if "waveform" not in audio_dict or "sample_rate" not in audio_dict:
            print(f"[SaveAudioMP3] Error: Audio dictionary missing 'waveform' or 'sample_rate' key. Keys: {audio_dict.keys()}")
            return {"ui": {"text": ["Error: Audio dictionary incomplete."]}}
        # ... (rest of audio_dict and source_waveform_data extraction)

        source_waveform_data = audio_dict["waveform"]
        sample_rate = audio_dict["sample_rate"]
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
        # 1. Create the initial filename template string from inputs
        current_filename_template = filename_prefix 
        if song_name and song_name.strip():
            sanitized_song_name = self._sanitize_filename_component(song_name.strip())
            if sanitized_song_name:
                if current_filename_template.endswith(('/', '\\')):
                    current_filename_template = f"{current_filename_template}{sanitized_song_name}"
                else:
                    dir_part, base_part = os.path.split(current_filename_template)
                    if base_part: current_filename_template = os.path.join(dir_part, f"{base_part}_{sanitized_song_name}")
                    else: current_filename_template = os.path.join(dir_part, sanitized_song_name)
        
        print(f"[SaveAudioMP3] Initial filename template: '{current_filename_template}'")

        # 2. Resolve ComfyUI-specific patterns (e.g., %date:yyyy-MM-dd%, [seed])
        #    using comfy.utils.format_filename.
        #    This function expects prompt and extra_pnginfo for some patterns.
        prefix_after_comfy_patterns = comfy.utils.format_filename(
            current_filename_template, 
            prompt,               # Pass the prompt object
            extra_pnginfo         # Pass the extra_pnginfo object
        )
        print(f"[SaveAudioMP3] Prefix after comfy.utils.format_filename (for get_save_image_path): '{prefix_after_comfy_patterns}'")

        # 3. Use folder_paths.get_save_image_path to handle Python strftime,
        #    directory creation, and unique counter.
        #    It takes 2 to 4 arguments: filename_prefix, output_dir, (optional) width, (optional) height
        num_samples_for_path = int(audio_segment.duration_seconds * audio_segment.frame_rate)
        num_channels_for_path = audio_segment.channels # Using channels as a proxy for "height" if needed by util

        actual_disk_output_folder, actual_disk_base_filename, counter, actual_ui_subfolder, _ = folder_paths.get_save_image_path(
            prefix_after_comfy_patterns, # This should now have ComfyUI patterns resolved
            self.output_dir,
            num_samples_for_path,  # Can be used as "width"
            num_channels_for_path  # Can be used as "height"
        )
        
        mp3_final_filename_on_disk = f"{actual_disk_base_filename}_{counter:05}.mp3"
        full_mp3_path_to_save_on_disk = os.path.join(actual_disk_output_folder, mp3_final_filename_on_disk)
        # --- End Filename and Path Generation ---

        export_parameters = {}
        if bitrate == "VBR (q4 default)": export_parameters['parameters'] = ["-q:a", "4"] 
        else: export_parameters['bitrate'] = bitrate
        
        try:
            print(f"[SaveAudioMP3] Attempting to save to actual disk path: '{full_mp3_path_to_save_on_disk}' with parameters: {export_parameters}")
            audio_segment.export(full_mp3_path_to_save_on_disk, format="mp3", **export_parameters)
        except Exception as e:
            # ... (error logging as before) ...
            msg = f"Error exporting MP3 to '{full_mp3_path_to_save_on_disk}': {e}."
            print(f"[SaveAudioMP3] {msg}")
            print(f"[SaveAudioMP3] Details for debugging path issue:")
            print(f"    - Original filename_prefix input: '{filename_prefix}'")
            print(f"    - Optional song_name input: '{song_name}'")
            print(f"    - Template passed to comfy.utils.format_filename: '{current_filename_template}'")
            print(f"    - Prefix after comfy.utils.format_filename (passed to get_save_image_path): '{prefix_after_comfy_patterns}'")
            print(f"    - Outputs from get_save_image_path: folder='{actual_disk_output_folder}', base_filename='{actual_disk_base_filename}', subfolder='{actual_ui_subfolder}'")
            return {"ui": {"text": [msg]}}
        
        print(f"[SaveAudioMP3] Successfully saved MP3 to: '{full_mp3_path_to_save_on_disk}'")
        
        results_list = [{"filename": mp3_final_filename_on_disk, "subfolder": actual_ui_subfolder, "type": self.type}]
        ui_payload = {"audio": results_list}
        # print(f"[SaveAudioMP3] Returning UI data for player: { {'ui': ui_payload} }")
        return {"ui": ui_payload}
