# ComfyUI_SaveAudioMP3/__init__.py

# Import the node class from your node file
from .save_audio_mp3_node import SaveAudioMP3

# A dictionary that ComfyUI uses to map node CLASS_NAME to node object
NODE_CLASS_MAPPINGS = {
    "SaveAudioMP3": SaveAudioMP3
}

# A dictionary that ComfyUI uses to map node CLASS_NAME to a display name for the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveAudioMP3": "Save Audio (MP3)"
}

# Optional: print a message to the console when the extension is loaded
print("----------------------------------------")
print("ComfyUI_SaveAudioMP3: Custom node loaded")
print("----------------------------------------")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
