# ComfyUI_SaveAudioMP3/save_audio_mp3_node.py

import os
import torch
import torchaudio # For saving WAV
import numpy as np
from pydub import AudioSegment
import folder_paths
import re

class SaveAudioMP3:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.temp_dir = folder_paths.get_temp_directory() # For temporary WAV
        # self.type for the main output (MP3) is "output"
        # self.type for the UI preview (WAV) will be "temp"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_data": ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "audio/comfy_audio"}), # For MP3
                "bitrate": (["128k", "192k", "256k", "320k", "VBR (q4 default)"], {"default": "320k"}),
            },
            "optional": { 
                "song_name": ("STRING", {"default": ""}), # For MP3 filename
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_audio_and_preview" # Renamed for clarity
    OUTPUT_NODE = True
    CATEGORY = "audio/save"

    def _sanitize_filename_component(self, name_string):
        if not name_string: return ""
        name_string = re.sub(r'[\s/:*?"<>|]+', '_', name_string)
        sanitized_name = re.sub(r'[^a-zA-Z0-9_.-]', '', name_string)
        return sanitized_name[:100]

    def save_audio_and_preview(self, audio_data, filename_prefix, bitrate, song_name=None):
        # --- Basic Input Validation (as before) ---
        if not isinstance(audio_data, dict):
            print(f"[SaveAudioMP3] Error: Expected a dictionary for audio_data. Got: {type(audio_data)}")
            return {"ui": {"text": ["Error: Invalid audio data format (expected dict)."]}}
        audio_dict = audio_data 
        if "waveform" not in audio_dict or "sample_rate" not in audio_dict:
            print(f"[SaveAudioMP3] Error: Audio dictionary missing 'waveform' or 'sample_rate' key. Keys: {audio_dict.keys()}")
            return {"ui": {"text": ["Error: Audio dictionary incomplete."]}}

        source_waveform_data = audio_dict["waveform"] # This is a torch.Tensor
        sample_rate = audio_dict["sample_rate"]
        
        # print(f"[SaveAudioMP3] Received audio. Waveform data type: {type(source_waveform_data)}, Sample rate: {sample_rate}")

        # --- 1. Save the primary MP3 to output directory ---
        mp3_saved_successfully = False
        try:
            # --- MP3 Filename and Path Generation (using self.output_dir) ---
            mp3_filename_prefix = filename_prefix
            if song_name and song_name.strip():
                sanitized_song_name = self._sanitize_filename_component(song_name.strip())
                if sanitized_song_name:
                    if mp3_filename_prefix.endswith(('/', '\\')): mp3_filename_prefix = f"{mp3_filename_prefix}{sanitized_song_name}"
                    else:
                        dir_part, base_part = os.path.split(mp3_filename_prefix)
                        if base_part: mp3_filename_prefix = os.path.join(dir_part, f"{base_part}_{sanitized_song_name}")
                        else: mp3_filename_prefix = os.path.join(dir_part, sanitized_song_name)
            
            # For MP3, use folder_paths.get_save_image_path with self.output_dir
            # We assume no ComfyUI wildcards for now, focusing on player
            mp3_output_folder, mp3_base_filename, mp3_counter, _, _ = folder_paths.get_save_image_path(
                mp3_filename_prefix, self.output_dir, 
                int(source_waveform_data.shape[-1]), # num_samples as width proxy
                int(source_waveform_data.shape[-2])  # num_channels as height proxy
            )
            actual_mp3_filename = f"{mp3_base_filename}_{mp3_counter:05}.mp3"
            full_mp3_path_to_save = os.path.join(mp3_output_folder, actual_mp3_filename)

            # --- AudioSegment Creation for MP3 (from Tensor) ---
            audio_segment_for_mp3 = None
            if isinstance(source_waveform_data, torch.Tensor):
                waveform_tensor = source_waveform_data
                waveform_single_item = waveform_tensor[0].cpu() if waveform_tensor.ndim == 3 and waveform_tensor.shape[0] > 0 else waveform_tensor.cpu()
                waveform_np = waveform_single_item.numpy().T 
                audio_data_int16 = (waveform_np * 32767).astype(np.int16)
                num_channels = audio_data_int16.shape[1] if audio_data_int16.ndim > 1 else 1
                audio_segment_for_mp3 = AudioSegment(data=audio_data_int16.tobytes(), sample_width=2, frame_rate=sample_rate, channels=num_channels)
            
            if audio_segment_for_mp3:
                export_parameters = {}
                if bitrate == "VBR (q4 default)": export_parameters['parameters'] = ["-q:a", "4"] 
                else: export_parameters['bitrate'] = bitrate
                
                print(f"[SaveAudioMP3] Saving MP3 to: '{full_mp3_path_to_save}'")
                audio_segment_for_mp3.export(full_mp3_path_to_save, format="mp3", **export_parameters)
                print(f"[SaveAudioMP3] Successfully saved MP3: '{full_mp3_path_to_save}'")
                mp3_saved_successfully = True
            else:
                print(f"[SaveAudioMP3] Error: Could not create AudioSegment for MP3 conversion.")
        
        except Exception as e:
            print(f"[SaveAudioMP3] Error during MP3 saving: {e}")
            # Continue to try and save WAV for preview if possible, but report MP3 error.
            # For now, let's return an error if MP3 saving fails, as it's the primary goal.
            return {"ui": {"text": [f"Error saving MP3: {e}"]}}


        # --- 2. Save a temporary WAV for UI preview ---
        # The input `source_waveform_data` is the raw waveform tensor.
        # The input `sample_rate` is its sample rate.
        ui_results = []
        try:
            # Use a generic prefix for the temp WAV file
            temp_wav_prefix = "preview_audio_mp3node" 
            
            # Use folder_paths.get_save_image_path with self.temp_dir
            wav_temp_folder, wav_temp_base_filename, wav_temp_counter, wav_temp_subfolder, _ = folder_paths.get_save_image_path(
                temp_wav_prefix, self.temp_dir,
                int(source_waveform_data.shape[-1]), 
                int(source_waveform_data.shape[-2]) 
            )
            actual_wav_filename = f"{wav_temp_base_filename}_{wav_temp_counter:05}.wav" # Save as WAV
            full_wav_path_for_preview = os.path.join(wav_temp_folder, actual_wav_filename)

            # The waveform in audio_dict["waveform"] is often (1, channels, samples)
            # torchaudio.save expects (channels, samples) or (samples) for mono
            waveform_to_save_wav = source_waveform_data.squeeze(0) # Remove batch dim if present

            print(f"[SaveAudioMP3] Saving temporary WAV for preview to: '{full_wav_path_for_preview}'")
            torchaudio.save(full_wav_path_for_preview, waveform_to_save_wav, sample_rate, format="wav")
            
            ui_results.append({
                "filename": actual_wav_filename,
                "subfolder": wav_temp_subfolder, # Subfolder relative to temp_dir
                "type": "temp"                   # Crucial: type is "temp"
            })
            print(f"[SaveAudioMP3] Temporary WAV for preview saved.")

        except Exception as e:
            print(f"[SaveAudioMP3] Error saving temporary WAV for preview: {e}")
            # If WAV preview fails, we can still return an empty UI or a text message
            # but the primary MP3 might have saved. For now, we'll just not have a preview.
            pass # Continue, as MP3 might be saved.

        # Return UI data pointing to the temporary WAV if created,
        # otherwise, just indicate MP3 was saved if that succeeded.
        if ui_results:
            return {"ui": {"audio": ui_results}}
        elif mp3_saved_successfully: # MP3 saved but no WAV preview
             return {"ui": {"text": [f"MP3 saved. Preview not generated: {actual_mp3_filename}"]}}
        else: # Should have been caught by MP3 error return earlier
            return {"ui": {"text": ["Audio saving failed."] }}
