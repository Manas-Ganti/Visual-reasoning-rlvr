#!/bin/bash

# Define paths
VIDEO_DIR="/Users/manasganti/VLM-RL-aicontent-detection/data/raw_data/OpenAI_Sora"
OUTPUT_DIR="/Users/manasganti/VLM-RL-aicontent-detection/data/raw_data/fake_images"

# Create output folder if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop through all MP4 videos in the folder
for video in "$VIDEO_DIR"/*.mp4; do
    # Check if files actually exist to avoid running on an empty glob
    [ -e "$video" ] || continue
    
    # Get the base name of the video file without the extension
    base_name=$(basename "$video" .mp4)
    
    echo "Processing video: $base_name"
    
    # FFmpeg command:
    # -i: Input file
    # -vf "fps=1": Extract exactly 1 frame per second of video
    # -q:v 2: High quality flag
    ffmpeg -i "$video" -vf "fps=1" -q:v 2 "$OUTPUT_DIR/${base_name}_frame_%04d.png" -loglevel warning
done

echo "Frame extraction complete!"