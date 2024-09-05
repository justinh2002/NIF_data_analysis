
import glob as glob
import os
import subprocess

 # Close the plot to avoid memory issues
png_files = sorted(glob.glob("jps_*.png"))

# Rename PNG files to a sequential pattern
for i, png_file in enumerate(png_files):
    new_name = f"jps{i:04d}.png"
    os.rename(png_file, new_name)

# FFmpeg command
ffmpeg_command = [
    "ffmpeg",
    "-framerate", "30",
    "-i", "jps%04d.png",
    "-c:v", "mpeg4",
    "-pix_fmt", "yuv420p",
    "dens.mp4"
]

# Run the FFmpeg command
subprocess.run(ffmpeg_command)

print("Video created successfully!")