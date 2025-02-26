import os
import subprocess

def convert_folder_mp3_to_wav(input_folder, output_folder):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all MP3 files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".mp3"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".wav")
            
            # Run ffmpeg command
            subprocess.run(
                ["ffmpeg", "-i", input_path, output_path],
                stdout=subprocess.DEVNULL,  # Suppress console output
                stderr=subprocess.DEVNULL
            )
            print(f"Converted: {input_path} -> {output_path}")

# Example usage
input_folder = "/ocean/projects/cis240055p/liuyuex/hg/mmug/audio_only"
output_folder = "/ocean/projects/cis240055p/liuyuex/hg/mmug/audio_only_wav"
convert_folder_mp3_to_wav(input_folder, output_folder)
