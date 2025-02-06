import os
import decord
import numpy as np

def check_video_stats(folder_path):
    """Check .mp4 files in the folder, compute total video length, and collect statistics."""
    corrupt_files = []
    total_frames = 0
    total_duration = 0.0  # Total duration in seconds
    total_videos = 0
    fps_list = []
    durations = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".mp4"):
            file_path = os.path.join(folder_path, file_name)
            try:
                vr = decord.VideoReader(file_path)
                first_frame = vr[0].numpy()  # Load first frame to check for corruption
                fps = vr.get_avg_fps()  # Get frames per second
                num_frames = len(vr)  # Total number of frames
                duration = num_frames / fps  # Compute duration in seconds
                
                total_frames += num_frames
                total_duration += duration
                fps_list.append(fps)
                durations.append(duration)
                total_videos += 1
                
                print(f"{file_name}: {duration:.2f} sec ({duration/60:.2f} min), FPS: {fps:.2f}")

            except Exception as e:
                print(f"Error loading {file_name}: {e}")
                corrupt_files.append(file_name)

    # Convert total duration to hours and minutes
    total_hours = int(total_duration // 3600)
    total_minutes = int((total_duration % 3600) // 60)

    # Compute additional statistics
    num_corrupt = len(corrupt_files)
    avg_duration = np.mean(durations) if durations else 0
    median_duration = np.median(durations) if durations else 0
    std_dev_duration = np.std(durations) if durations else 0
    percentiles = np.percentile(durations, [25, 50, 75]) if durations else [0, 0, 0]
    avg_fps = np.mean(fps_list) if fps_list else 0
    shortest_video = min(durations) if durations else 0
    longest_video = max(durations) if durations else 0

    # Print summary stats
    print("\nSummary:")
    print(f"Total Videos Processed: {total_videos}")
    print(f"Corrupt Videos: {num_corrupt}")

    if num_corrupt:
        print("\nThe following videos could not be loaded:")
        for file in corrupt_files:
            print(file)
    else:
        print("All videos loaded successfully.")

    print(f"\nTotal Duration: {total_duration:.2f} seconds")
    print(f"Total Duration: {total_hours} hours, {total_minutes} minutes")
    print(f"Average Video Duration: {avg_duration:.2f} sec ({avg_duration/60:.2f} min)")
    print(f"Median Video Duration: {median_duration:.2f} sec ({median_duration/60:.2f} min)")
    print(f"Standard Deviation: {std_dev_duration:.2f} sec")
    print(f"Percentiles - 25th: {percentiles[0]:.2f} sec, 50th: {percentiles[1]:.2f} sec, 75th: {percentiles[2]:.2f} sec")
    print(f"Shortest Video: {shortest_video:.2f} sec ({shortest_video/60:.2f} min)")
    print(f"Longest Video: {longest_video:.2f} sec ({longest_video/60:.2f} min)")
    print(f"Total Frames Processed: {total_frames}")
    print(f"Average FPS: {avg_fps:.2f}")

    return {
        "corrupt_videos": corrupt_files,
        "total_videos": total_videos,
        "num_corrupt": num_corrupt,
        "total_hours": total_hours,
        "total_minutes": total_minutes,
        "total_duration_sec": total_duration,
        "avg_duration_sec": avg_duration,
        "median_duration_sec": median_duration,
        "std_dev_duration_sec": std_dev_duration,
        "percentiles": {
            "25th": percentiles[0],
            "50th": percentiles[1],
            "75th": percentiles[2],
        },
        "shortest_video_sec": shortest_video,
        "longest_video_sec": longest_video,
        "total_frames": total_frames,
        "avg_fps": avg_fps
    }

# Example usage
folder_path = "/home/liuyuex/.cache/huggingface/mmug/vid"
corrupt_videos = check_video_stats(folder_path)