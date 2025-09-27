#!/usr/bin/env python3
"""
AI Multi-Camera Video Editor with Professional Cutting and Lip Sync
Combines 3 video sources with 1 audio track using intelligent editing algorithms
"""

import os
import sys
import numpy as np
import cv2
from moviepy.editor import *
from moviepy.video.fx import resize
import librosa
import soundfile as sf
from scipy import signal
from scipy.spatial.distance import euclidean
import json
from datetime import datetime, timedelta
import random
import argparse
import logging
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, concatenate_videoclips


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiCameraEditor:
    def __init__(self, video_paths, audio_path, output_path):
        self.video_paths = video_paths
        self.audio_path = audio_path
        self.output_path = output_path
        self.videos = []
        self.audio_clip = None
        self.audio_data = None
        self.sample_rate = None
        
        # Professional editing parameters
        self.min_cut_duration = 0.5  # Minimum clip duration in seconds
        self.max_cut_duration = 8.0  # Maximum clip duration in seconds
        self.fade_duration = 0.1     # Crossfade duration
        self.silence_threshold = 0.02 # Threshold for detecting silence
        
    def load_media_files(self):
        """Load all video and audio files"""
        logger.info("Loading media files...")
        
        # Load videos
        for i, video_path in enumerate(self.video_paths):
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            video = VideoFileClip(video_path)
            self.videos.append(video)
            logger.info(f"Loaded video {i+1}: {video_path} (Duration: {video.duration:.2f}s)")
        
        # Load audio
        if not os.path.exists(self.audio_path):
            raise FileNotFoundError(f"Audio file not found: {self.audio_path}")
        
        self.audio_clip = AudioFileClip(self.audio_path)
        self.audio_data, self.sample_rate = librosa.load(self.audio_path, sr=None)
        logger.info(f"Loaded audio: {self.audio_path} (Duration: {self.audio_clip.duration:.2f}s)")
    
    def synchronize_videos(self):
        """Synchronize video start times with audio using cross-correlation"""
        logger.info("Synchronizing videos with audio...")
        
        synced_videos = []
        audio_for_sync = self.audio_data[:int(10 * self.sample_rate)]  # First 10 seconds
        
        for i, video in enumerate(self.videos):
            if video.audio is None:
                logger.warning(f"Video {i+1} has no audio track, using as-is")
                synced_videos.append(video)
                continue
            
            # Extract video audio for synchronization
            video_audio_array = video.audio.to_soundarray(fps=self.sample_rate)
            if len(video_audio_array.shape) > 1:
                video_audio_array = np.mean(video_audio_array, axis=1)
            
            # Use first 10 seconds for sync
            video_audio_sync = video_audio_array[:min(len(video_audio_array), 
                                                    int(10 * self.sample_rate))]
            
            # Cross-correlation to find offset
            correlation = signal.correlate(audio_for_sync, video_audio_sync, mode='full')
            offset_samples = correlation.argmax() - len(video_audio_sync) + 1
            offset_seconds = offset_samples / self.sample_rate
            
            logger.info(f"Video {i+1} sync offset: {offset_seconds:.3f} seconds")
            
            # Apply offset
            if offset_seconds > 0:
                synced_video = video.subclip(offset_seconds)
            else:
                synced_video = CompositeVideoClip([video.set_start(-offset_seconds)])
            
            synced_videos.append(synced_video)
        
        self.videos = synced_videos
    
    def detect_speech_segments(self):
        """Detect speech segments and word boundaries in audio"""
        logger.info("Analyzing audio for speech segments...")
        
        # Voice Activity Detection using energy and spectral features
        frame_length = int(0.025 * self.sample_rate)  # 25ms frames
        frame_shift = int(0.010 * self.sample_rate)   # 10ms shift
        
        frames = librosa.util.frame(self.audio_data, 
                                  frame_length=frame_length, 
                                  hop_length=frame_shift, 
                                  axis=0)
        
        # Calculate energy and spectral centroid for each frame
        energy = np.sum(frames**2, axis=1)
        spectral_centroids = []
        
        for frame in frames:
            if np.sum(frame**2) > 0:
                fft = np.fft.fft(frame)
                magnitude = np.abs(fft[:len(fft)//2])
                freqs = np.fft.fftfreq(len(fft), 1/self.sample_rate)[:len(fft)//2]
                centroid = np.sum(freqs * magnitude) / np.sum(magnitude) if np.sum(magnitude) > 0 else 0
                spectral_centroids.append(centroid)
            else:
                spectral_centroids.append(0)
        
        spectral_centroids = np.array(spectral_centroids)
        
        # Normalize features
        energy_norm = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-8)
        spectral_norm = (spectral_centroids - np.min(spectral_centroids)) / \
                       (np.max(spectral_centroids) - np.min(spectral_centroids) + 1e-8)
        
        # Combine features for voice activity detection
        voice_activity = (energy_norm > 0.1) & (spectral_norm > 0.1)
        
        # Convert frame indices to time
        time_per_frame = frame_shift / self.sample_rate
        speech_segments = []
        
        in_speech = False
        start_time = 0
        
        for i, is_voice in enumerate(voice_activity):
            current_time = i * time_per_frame
            
            if is_voice and not in_speech:
                start_time = current_time
                in_speech = True
            elif not is_voice and in_speech:
                if current_time - start_time > 0.1:  # Minimum segment length
                    speech_segments.append((start_time, current_time))
                in_speech = False
        
        # Handle case where speech continues to end
        if in_speech:
            speech_segments.append((start_time, len(self.audio_data) / self.sample_rate))
        
        logger.info(f"Detected {len(speech_segments)} speech segments")
        return speech_segments
    
    def detect_word_boundaries(self, speech_segments):
        """Detect approximate word boundaries within speech segments"""
        logger.info("Detecting word boundaries...")
        
        word_boundaries = []
        
        for start_time, end_time in speech_segments:
            # Extract audio segment
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            segment = self.audio_data[start_sample:end_sample]
            
            # Detect pauses within speech (potential word boundaries)
            frame_size = int(0.020 * self.sample_rate)  # 20ms frames
            hop_size = int(0.010 * self.sample_rate)    # 10ms hop
            
            frames = librosa.util.frame(segment, 
                                      frame_length=frame_size, 
                                      hop_length=hop_size, 
                                      axis=0)
            
            energy = np.sum(frames**2, axis=1)
            energy_smooth = signal.medfilt(energy, kernel_size=5)
            
            # Find local minima (potential word boundaries)
            threshold = np.percentile(energy_smooth, 25)  # Bottom 25% energy
            minima = signal.find_peaks(-energy_smooth, 
                                     height=-threshold,
                                     distance=int(0.1 * self.sample_rate / hop_size))[0]
            
            # Convert to absolute time
            for minimum in minima:
                boundary_time = start_time + (minimum * hop_size / self.sample_rate)
                if start_time + 0.1 < boundary_time < end_time - 0.1:  # Avoid edges
                    word_boundaries.append(boundary_time)
        
        word_boundaries.sort()
        logger.info(f"Detected {len(word_boundaries)} potential word boundaries")
        return word_boundaries
    
    def calculate_visual_complexity(self, video_clip, start_time, duration):
        """Calculate visual complexity score for a video segment"""
        try:
            # Sample a few frames from the segment
            sample_times = np.linspace(start_time, start_time + duration, 5)
            complexity_scores = []
            
            for t in sample_times:
                if t < video_clip.duration:
                    frame = video_clip.get_frame(t)
                    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    
                    # Calculate Laplacian variance (edge detection)
                    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                    
                    # Calculate histogram entropy
                    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                    hist_norm = hist / hist.sum()
                    entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-8))
                    
                    complexity_scores.append(laplacian_var + entropy * 10)
            
            return np.mean(complexity_scores)
        except:
            return 0
    
    def generate_cut_plan(self, speech_segments, word_boundaries):
        """Generate intelligent cutting plan for multi-camera editing"""
        logger.info("Generating intelligent cut plan...")
        
        cut_plan = []
        current_time = 0
        total_duration = self.audio_clip.duration
        
        # Create potential cut points (word boundaries + speech segment boundaries)
        cut_points = set(word_boundaries)
        for start, end in speech_segments:
            cut_points.add(start)
            cut_points.add(end)
        
        cut_points = sorted(list(cut_points))
        cut_points = [0] + [cp for cp in cut_points if 0 < cp < total_duration] + [total_duration]
        
        while current_time < total_duration:
            # Find next possible cut points
            valid_cuts = [cp for cp in cut_points 
                         if current_time + self.min_cut_duration <= cp <= current_time + self.max_cut_duration]
            
            if not valid_cuts:
                # If no valid cuts found, use maximum duration or end
                next_cut = min(current_time + self.max_cut_duration, total_duration)
            else:
                # Weighted random selection favoring cuts at word boundaries
                weights = []
                for cut_point in valid_cuts:
                    weight = 1.0
                    # Prefer cuts at word boundaries
                    if cut_point in word_boundaries:
                        weight *= 2.0
                    # Prefer cuts at speech segment boundaries
                    for start, end in speech_segments:
                        if abs(cut_point - start) < 0.1 or abs(cut_point - end) < 0.1:
                            weight *= 1.5
                            break
                    weights.append(weight)
                
                # Normalize weights
                weights = np.array(weights)
                weights = weights / weights.sum()
                
                # Select cut point
                next_cut = np.random.choice(valid_cuts, p=weights)
            
            duration = next_cut - current_time
            
            # Select camera based on visual complexity and variety
            camera_scores = []
            for i, video in enumerate(self.videos):
                if current_time < video.duration:
                    complexity = self.calculate_visual_complexity(video, current_time, duration)
                    # Add variety bonus (penalize recently used cameras)
                    variety_bonus = 1.0
                    if cut_plan and len([cp for cp in cut_plan[-3:] if cp['camera'] == i]) > 1:
                        variety_bonus = 0.5
                    
                    camera_scores.append(complexity * variety_bonus)
                else:
                    camera_scores.append(0)
            
            # Select camera with highest score (with some randomness)
            if max(camera_scores) > 0:
                camera_probs = np.array(camera_scores)
                camera_probs = camera_probs / camera_probs.sum()
                # Add some randomness while favoring higher scores
                camera_probs = camera_probs ** 0.7  # Soften the distribution
                camera_probs = camera_probs / camera_probs.sum()
                selected_camera = np.random.choice(len(self.videos), p=camera_probs)
            else:
                selected_camera = random.randint(0, len(self.videos) - 1)
            
            cut_plan.append({
                'start_time': current_time,
                'end_time': next_cut,
                'duration': duration,
                'camera': selected_camera
            })
            
            current_time = next_cut
        
        logger.info(f"Generated {len(cut_plan)} cuts")
        return cut_plan
    
    def create_final_video(self, cut_plan):
        """Create the final edited video"""
        logger.info("Creating final video...")
        
        video_clips = []
        
        for i, cut in enumerate(cut_plan):
            camera_idx = cut['camera']
            start_time = cut['start_time']
            end_time = cut['end_time']
            
            if start_time >= self.videos[camera_idx].duration:
                # If this camera doesn't have enough footage, use another one
                available_cameras = [j for j, v in enumerate(self.videos) 
                                   if start_time < v.duration]
                if available_cameras:
                    camera_idx = random.choice(available_cameras)
                else:
                    continue
            
            # Extract video clip
            video_segment = self.videos[camera_idx].subclip(start_time, 
                                                          min(end_time, self.videos[camera_idx].duration))
            
            # Add crossfades (except for first and last clips)
            if i > 0:
                video_segment = video_segment.crossfadein(self.fade_duration)
            if i < len(cut_plan) - 1:
                video_segment = video_segment.crossfadeout(self.fade_duration)
            
            video_clips.append(video_segment)
            logger.info(f"Added clip {i+1}: Camera {camera_idx+1}, {start_time:.2f}-{end_time:.2f}s")
        
        if not video_clips:
            raise ValueError("No valid video clips generated")
        
        # Concatenate all clips
        final_video = concatenate_videoclips(video_clips, method="compose")
        
        # Set the audio to the original audio track
        final_video = final_video.set_audio(self.audio_clip)
        
        # Ensure consistent resolution
        target_height = 1080
        final_video = final_video.resize(height=target_height)
        
        return final_video
    
    def export_cut_log(self, cut_plan):
        """Export cutting plan as JSON for reference"""
        log_path = self.output_path.rsplit('.', 1)[0] + '_cut_log.json'
        
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'input_files': {
                'videos': self.video_paths,
                'audio': self.audio_path
            },
            'parameters': {
                'min_cut_duration': self.min_cut_duration,
                'max_cut_duration': self.max_cut_duration,
                'fade_duration': self.fade_duration
            },
            'cuts': cut_plan
        }
        
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        logger.info(f"Cut log exported to: {log_path}")
    
    def process(self):
        """Main processing pipeline"""
        try:
            # Load media files
            self.load_media_files()
            
            # Synchronize videos with audio
            self.synchronize_videos()
            
            # Analyze audio for intelligent cutting
            speech_segments = self.detect_speech_segments()
            word_boundaries = self.detect_word_boundaries(speech_segments)
            
            # Generate cutting plan
            cut_plan = self.generate_cut_plan(speech_segments, word_boundaries)
            
            # Export cut log
            self.export_cut_log(cut_plan)
            
            # Create final video
            final_video = self.create_final_video(cut_plan)
            
            # Export final video
            logger.info(f"Exporting final video to: {self.output_path}")
            final_video.write_videofile(
                self.output_path,
                fps=30,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                verbose=False,
                logger=None
            )
            
            logger.info("Video editing completed successfully!")
            
        except Exception as e:
            logger.error(f"Error during processing: {str(e)}")
            raise
        
        finally:
            # Clean up
            for video in self.videos:
                video.close()
            if self.audio_clip:
                self.audio_clip.close()

def main():
    parser = argparse.ArgumentParser(description="AI Multi-Camera Video Editor")
    parser.add_argument('--videos', nargs=3, required=True,
                       help='Paths to 3 video files')
    parser.add_argument('--audio', required=True,
                       help='Path to audio file')
    parser.add_argument('--output', required=True,
                       help='Output video path')
    parser.add_argument('--min-duration', type=float, default=0.5,
                       help='Minimum cut duration in seconds')
    parser.add_argument('--max-duration', type=float, default=8.0,
                       help='Maximum cut duration in seconds')
    
    args = parser.parse_args()
    
    # Validate input files
    for video_path in args.videos:
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            sys.exit(1)
    
    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found: {args.audio}")
        sys.exit(1)
    
    # Create editor and process
    editor = MultiCameraEditor(args.videos, args.audio, args.output)
    editor.min_cut_duration = args.min_duration
    editor.max_cut_duration = args.max_duration
    
    try:
        editor.process()
        print(f"\nSuccess! Final video saved to: {args.output}")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()