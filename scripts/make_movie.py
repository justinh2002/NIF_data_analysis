import glob
from cfpack import stop, print
import timeit
import flashplotlib as fpl
import subprocess
import os

png_files = sorted(glob.glob("jps_*.pdf"))

# Rename PNG files to a sequential pattern
for i, png_file in enumerate(png_files):
    new_name = f"frame_{i:04d}.png"
    os.rename(png_file, new_name)

# FFmpeg command
ffmpeg_command = [
    "ffmpeg",
    "-framerate", "30",
    "-i", "frame_%04d.png",
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    "output.mp4"
]

# Run the FFmpeg command
subprocess.run(ffmpeg_command)

print("Video created successfully!")