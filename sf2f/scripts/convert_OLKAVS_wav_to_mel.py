import warnings
warnings.filterwarnings("ignore")  # 경고 메시지 무시 설정

import argparse
import os
import shutil
import gc
import time
import pickle
import numpy as np
import pandas as pd
import multiprocessing as mp
from tensorflow.io import gfile
import sys
sys.path.append('./')
from utils.wav2mel import wav_to_mel
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils.filter_pickle import filtering_pickle

DATA_DIR = '/home/data/data_OLKAVS/OLKAVS/data'
OUTPUT_DIR = '/home/mz01/voice2face-modeling/newmel'

class WavConverter:
    def __init__(self):
        self.get_wav_dirs()
        self.create_output_dirs()

    def convert_identity(self, wav_dir, mel_home_dir):
        '''
        Convert WAV files in a speaker directory to MEL spectrograms and save as pickle files.

        Arguments:
            1. wav_dir (str): Path to the identity's raw_wav folder.
            2. mel_home_dir (str): Path to the output directory where MEL pickle files will be saved.
        '''
        spkid = os.path.basename(os.path.dirname(wav_dir))
        
        # Create speaker directory if it does not exist
        speaker_dir = os.path.join(mel_home_dir, spkid)
        os.makedirs(speaker_dir, exist_ok=True)

        mel_dir = os.path.join(speaker_dir, 'audio')
        os.makedirs(mel_dir, exist_ok=True)

        wav_files = os.listdir(wav_dir)
        for wav_file in wav_files:
            if not wav_file.endswith('.wav'):
                continue
            # Read and process the wav
            wav_path = os.path.join(wav_dir, wav_file)
            try:
                wavid = wav_file.replace('.wav', '')
                # Adjusted pickle name format
                pickle_name = f"{wavid}.pickle"
                pickle_path = os.path.join(mel_dir, pickle_name)
                if os.path.exists(pickle_path):
                    # Skip if exists
                    continue
                log_mel = wav_to_mel(wav_path)
                pickle_dict = {
                    'LogMel_Features': log_mel,
                    'spkid': spkid,
                    'wavid': wavid
                }
                with open(pickle_path, "wb") as f:
                    pickle.dump(pickle_dict, f)
                print(f'Saved pickle file: {pickle_path}')
            except Exception as e:
                print(f'Error processing {wav_file}: {e}')
        gc.collect()

    def get_wav_dirs(self):
        '''
        Get paths to the wav_dir of all speakers.
        '''
        wav_dirs = []
        for subdir in os.listdir(DATA_DIR):
            audio_dir = os.path.join(DATA_DIR, subdir, 'audio')
            if os.path.isdir(audio_dir):
                wav_dirs.append(audio_dir)
        self.wav_dirs = wav_dirs

    def create_output_dirs(self):
        '''
        Create the output directory if it does not exist.
        '''
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def _worker(self, job_id, infos):
        '''
        Worker function for parallel processing.
        '''
        for i, info in enumerate(infos):
            self.convert_identity(info[0], info[1])
            print(f'Job #{job_id} processed {i+1}/{len(infos)} directories: {info[0]}')

    def convert_wav_to_mel(self, n_jobs=1):
        '''
        Convert WAV files to MEL spectrograms using multiple processes.

        Arguments:
            - n_jobs (int): Number of parallel processes to use.
        '''
        infos = [(wav_dir, OUTPUT_DIR) for wav_dir in self.wav_dirs]
        print(f"Total directories to process: {len(infos)}")
        
        n_wav_dirs = len(infos)
        n_jobs = min(n_jobs, n_wav_dirs)
        n_wav_dirs_per_job = n_wav_dirs // n_jobs
        
        process_index = []
        for ii in range(n_jobs):
            start_idx = ii * n_wav_dirs_per_job
            end_idx = (ii + 1) * n_wav_dirs_per_job
            process_index.append([start_idx, end_idx])
        
        # Adjust the last job's end index to accommodate any remaining directories
        if n_jobs * n_wav_dirs_per_job != n_wav_dirs:
            process_index[-1][1] = n_wav_dirs
        
        futures = set()
        with ProcessPoolExecutor() as executor:
            for job_id, (start, end) in enumerate(process_index):
                future = executor.submit(self._worker, job_id, infos[start:end])
                futures.add(future)
                print(f'Submitted job {job_id}: processing directories {start}-{end-1}')
            
            # Wait for all jobs to complete
            for future in as_completed(futures):
                try:
                    future.result()  # Ensure any exceptions are raised here
                except Exception as e:
                    print(f'Error in job: {e}')
        
        print("Conversion completed.")

def main():
    parser = argparse.ArgumentParser(description="Convert WAV files to MEL spectrograms.")
    parser.add_argument('--n_jobs', '-n', type=int, default=5,
                        help='Number of parallel jobs to run.')
    args = parser.parse_args()
    
    wav_converter = WavConverter()
    wav_converter.convert_wav_to_mel(args.n_jobs)
    
    # Filter the created pickle files
    dir_path = OUTPUT_DIR
    filtering_pickle(dir_path)

if __name__ == '__main__':
    main()
