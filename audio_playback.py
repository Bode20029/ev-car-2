import pygame
import os

def play_audio(file_paths):
    pygame.mixer.init()
    for file_path in file_paths:
        if os.path.exists(file_path):
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        else:
            print(f"File not found: {file_path}")

# Usage
script_dir = os.path.dirname(os.path.abspath(__file__))
audio_files = [
    os.path.join(script_dir, "alert.mp3"),
    os.path.join(script_dir, "Warning.mp3")
]
play_audio(audio_files)