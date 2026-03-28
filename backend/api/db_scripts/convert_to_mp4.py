import os
import subprocess
from pathlib import Path
import imageio_ffmpeg

DATASETS_PATH = Path("../../../../Datasets")

available_dataset = {
    "AI4RiSK": {
        "path": DATASETS_PATH / "AI4RiSK",
        "non_violence_dirs": ['0'],
        "violence_dirs": ['1', '2', '3', '4'],
    },
    "Movies": {
        "path": DATASETS_PATH / "Movies",
        "violence_dirs": ['Violence'],
        "non_violence_dirs": ['NonViolence'],
    },
    "Hockey": {
        "path": DATASETS_PATH / "Hockey",
        "violence_dirs": ['Violence'],
        "non_violence_dirs": ['NonViolence'],
    },
    "Crowd": {
        "path": DATASETS_PATH / "Crowd",
        "violence_dirs": ['Violence'],
        "non_violence_dirs": ['NonViolence'],
    },
}

def convert_to_mp4():
    valid_extensions = {'.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.mpeg', '.mpg'}
    for dataset_key, dataset_value in available_dataset.items():
        try:
            for dir_name in dataset_value.get("violence_dirs", []) + dataset_value.get("non_violence_dirs", []):
                dir_path = dataset_value["path"] / dir_name
                if not dir_path.exists():
                    print(f"Directory {dir_path} does not exist. Skipping.")
                    continue
                    
                for video_file in dir_path.glob("*"):
                    if video_file.is_file() and video_file.suffix.lower() in valid_extensions:
                        new_file_path = video_file.with_suffix('.mp4')
                        print(f"Converting {video_file.name} to MP4...")
                        
                        try:
                            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                            
                            # Using bundled ffmpeg to convert to mp4
                            result = subprocess.run([
                                ffmpeg_exe, '-y', '-i', str(video_file),
                                '-vcodec', 'libx264', '-crf', '23',
                                str(new_file_path)
                            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                            
                            if result.returncode == 0:
                                # Remove original file after successful conversion
                                os.remove(video_file)
                                print(f"Successfully converted to {new_file_path.name} and removed original.")
                            else:
                                print(f"Failed to convert {video_file.name}. Error:\n{result.stderr.decode('utf-8', errors='ignore')}")
                                if new_file_path.exists():
                                    os.remove(new_file_path) # Clean up partial file
                        except Exception as e:
                            print(f"Failed to convert {video_file.name}: {e}")
                            if new_file_path.exists():
                                os.remove(new_file_path) # Clean up partial file
        except Exception as e:
            print(f"Error processing dataset {dataset_key}: {e}")

if __name__ == "__main__":
    convert_to_mp4()
