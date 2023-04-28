from pygame import mixer
import random
import os

def play():
    """Plays sound file that is randomly selected from the sound folder.
    Sound files must be in .wav format."""

    # specify directory where sounds are located
    sound_dir = "./sounds/"
    sound_files = os.listdir(sound_dir)
    sound = random.choice(sound_files)

    mixer.init()
    mixer.music.load(sound_dir + sound)
    mixer.music.play()
    while mixer.music.get_busy():
        continue

if __name__ == "__main__":
    play()
