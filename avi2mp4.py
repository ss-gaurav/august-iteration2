import subprocess

def convert_avi_to_mp4(input_file, output_file):
    # Construct the ffmpeg command
    command = [
        'ffmpeg',
        '-i', input_file,
        '-vcodec', 'libx264',
        '-crf', '20',
        '-preset', 'slow',
        '-acodec', 'aac',
        '-b:a', '192k',
        output_file
    ]

    # Run the command
    try:
        subprocess.run(command, check=True)
        print(f"Conversion successful: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")

if __name__ == "__main__":
    input_file = "video/dumdum1.avi"   # Replace with your input AVI file
    output_file = "output.mp4" # Replace with your desired output MP4 file
    convert_avi_to_mp4(input_file, output_file)
