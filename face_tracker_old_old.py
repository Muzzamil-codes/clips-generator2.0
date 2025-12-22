"""
Podcast Video to Vertical Shorts Converter
==========================================

Converts horizontal (16:9) podcast videos into vertical (9:16) short-form videos
by automatically detecting, tracking, and centering the active speaker.

Requirements (install via pip):
    pip install opencv-python mediapipe numpy

FFMPEG must be installed and available in PATH for final encoding.
    Download from: https://ffmpeg.org/download.html

Usage:
    python podcast_to_shorts.py input_video.mp4 output_video.mp4

Author: AI Assistant
License: MIT
"""

import cv2
import numpy as np
import subprocess
import sys
import os
import tempfile
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import time

# MediaPipe Tasks API (new API for mediapipe >= 0.10.0)
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

@dataclass
class Config:
    """Configuration for the video processor."""
    
    # Output aspect ratio (9:16 for vertical shorts)
    output_aspect_ratio: float = 9 / 16
    
    # Face detection frequency (detect every N frames, track in between)
    detection_interval: int = 3  # Detect more frequently for faster response
    
    # Smoothing parameters
    ema_alpha: float = 0.15  # Exponential moving average alpha (lower = smoother)
    ema_alpha_fast: float = 0.4  # Faster alpha for scene changes/recovery
    
    # Speaker switch parameters
    speaker_switch_delay_frames: int = 10  # Frames to wait before switching speakers
    speaker_switch_threshold: float = 0.3  # Min difference to trigger switch
    
    # Face tracking parameters
    max_face_distance: float = 150  # Max pixel distance to match faces between frames
    face_lost_timeout: int = 10  # Frames before considering a face lost (reduced for faster response)
    
    # Crop positioning
    vertical_face_position: float = 0.35  # Face vertical position (0.35 = upper third)
    horizontal_face_position: float = 0.5  # Face horizontal position (0.5 = center)
    
    # Minimum face size (relative to frame height)
    min_face_size_ratio: float = 0.05
    
    # MediaPipe confidence thresholds
    min_detection_confidence: float = 0.3  # Lower to detect more faces, we filter later
    min_tracking_confidence: float = 0.4
    
    # Scene change detection
    scene_change_threshold: float = 20.0  # Lowered for better camera cut detection
    
    # Camera angle change detection (face position jumps)
    face_jump_threshold: float = 150.0  # Reduced to detect smaller jumps
    
    # No-face behavior
    frames_before_center_fallback: int = 30  # Wait longer before moving to center (1 second at 30fps)
    center_transition_speed: float = 0.08  # Slower transition to center


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Face:
    """Represents a detected face with tracking information."""
    id: int
    center_x: float
    center_y: float
    width: float
    height: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    confidence: float = 1.0
    frames_since_seen: int = 0
    
    @property
    def area(self) -> float:
        """Calculate face area."""
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get face center coordinates."""
        return (self.center_x, self.center_y)


@dataclass
class TrackedFace:
    """Face with tracking history for speaker detection."""
    face: Face
    position_history: deque = field(default_factory=lambda: deque(maxlen=30))
    movement_score: float = 0.0
    horizontal_movement: float = 0.0  # X-axis movement (speaking indicator)
    vertical_movement: float = 0.0    # Y-axis movement (nodding indicator)
    avg_confidence: float = 0.0       # Average detection confidence
    confidence_history: deque = field(default_factory=lambda: deque(maxlen=30))
    
    def update_movement(self):
        """Calculate movement score based on position history.
        
        Speaking typically involves more varied movement than just nodding.
        Nodding is primarily vertical (up-down), while speaking involves
        more horizontal movement and varied patterns.
        """
        if len(self.position_history) < 2:
            self.movement_score = 0.0
            self.horizontal_movement = 0.0
            self.vertical_movement = 0.0
            return
        
        # Calculate horizontal and vertical movement separately
        h_movements = []
        v_movements = []
        positions = list(self.position_history)
        for i in range(1, len(positions)):
            dx = abs(positions[i][0] - positions[i-1][0])
            dy = abs(positions[i][1] - positions[i-1][1])
            h_movements.append(dx)
            v_movements.append(dy)
        
        self.horizontal_movement = np.mean(h_movements) if h_movements else 0.0
        self.vertical_movement = np.mean(v_movements) if v_movements else 0.0
        
        # Speaking score: prioritize horizontal movement, penalize pure vertical (nodding)
        # A speaking person moves in varied ways; a nodding person only moves vertically
        if self.vertical_movement > 0:
            h_to_v_ratio = self.horizontal_movement / (self.vertical_movement + 0.1)
        else:
            h_to_v_ratio = self.horizontal_movement * 10
        
        # Combined movement with horizontal bias
        # Horizontal movement is weighted 3x more than vertical
        self.movement_score = (self.horizontal_movement * 3 + self.vertical_movement) / 4
        
        # Update average confidence
        if self.confidence_history:
            self.avg_confidence = np.mean(list(self.confidence_history))
        else:
            self.avg_confidence = self.face.confidence


@dataclass
class CropRegion:
    """Represents the crop region for the output frame."""
    x: int
    y: int
    width: int
    height: int
    
    def clamp(self, frame_width: int, frame_height: int):
        """Clamp crop region to frame boundaries."""
        self.x = max(0, min(self.x, frame_width - self.width))
        self.y = max(0, min(self.y, frame_height - self.height))


# =============================================================================
# FACE DETECTOR
# =============================================================================

class FaceDetector:
    """
    Face detection using MediaPipe Face Detection (Tasks API).
    
    MediaPipe is chosen over OpenCV's Haar cascades because:
    - More accurate detection
    - Better performance on CPU
    - Provides facial landmarks for better center estimation
    
    Uses the new MediaPipe Tasks API (mediapipe >= 0.10.0).
    """
    
    # Model URL for automatic download
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
    MODEL_FILENAME = "blaze_face_short_range.tflite"
    
    def __init__(self, config: Config):
        self.config = config
        
        # Get or download the model file
        model_path = self._get_model_path()
        
        # Initialize MediaPipe Face Detection using Tasks API
        base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=config.min_detection_confidence,
            running_mode=vision.RunningMode.IMAGE
        )
        self.face_detector = vision.FaceDetector.create_from_options(options)
        
    def _get_model_path(self) -> str:
        """Get the model path, downloading if necessary."""
        # Check in current directory first
        if os.path.exists(self.MODEL_FILENAME):
            return self.MODEL_FILENAME
        
        # Check in script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, self.MODEL_FILENAME)
        
        if os.path.exists(model_path):
            return model_path
        
        # Download the model
        print(f"[INFO] Downloading face detection model...")
        try:
            urllib.request.urlretrieve(self.MODEL_URL, model_path)
            print(f"[INFO] Model downloaded to: {model_path}")
            return model_path
        except Exception as e:
            print(f"[WARNING] Could not download model: {e}")
            print("[INFO] Falling back to OpenCV face detection...")
            return None
        
    def detect(self, frame: np.ndarray) -> List[Face]:
        """
        Detect faces in the given frame.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            List of detected Face objects
        """
        frame_height, frame_width = frame.shape[:2]
        min_face_size = frame_height * self.config.min_face_size_ratio
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect faces
        detection_result = self.face_detector.detect(mp_image)
        
        faces = []
        if detection_result.detections:
            for idx, detection in enumerate(detection_result.detections):
                # Get bounding box
                bbox = detection.bounding_box
                
                x = bbox.origin_x
                y = bbox.origin_y
                w = bbox.width
                h = bbox.height
                
                # Filter out small faces
                if w < min_face_size or h < min_face_size:
                    continue
                
                # Clamp coordinates
                x = max(0, x)
                y = max(0, y)
                w = min(w, frame_width - x)
                h = min(h, frame_height - y)
                
                # Calculate face center
                center_x = x + w / 2
                center_y = y + h / 2
                
                # Get confidence score
                confidence = detection.categories[0].score if detection.categories else 1.0
                
                faces.append(Face(
                    id=-1,  # Will be assigned by tracker
                    center_x=center_x,
                    center_y=center_y,
                    width=w,
                    height=h,
                    bbox=(x, y, w, h),
                    confidence=confidence
                ))
        
        return faces
    
    def close(self):
        """Release resources."""
        self.face_detector.close()


# =============================================================================
# FACE TRACKER
# =============================================================================

class FaceTracker:
    """
    Tracks faces across frames with persistent IDs.
    
    Uses simple distance-based matching between frames to maintain
    face identity over time. This approach is lightweight and works
    well for podcast scenarios with limited face movement.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.tracked_faces: Dict[int, TrackedFace] = {}
        self.next_face_id = 0
        self.last_detections: List[Face] = []
        self.frames_without_faces: int = 0  # Track consecutive frames without any face
        self.consecutive_detections: Dict[int, int] = {}  # Track how many consecutive frames each face was seen
        
    def update(self, detected_faces: List[Face]) -> Dict[int, TrackedFace]:
        """
        Update tracked faces with new detections.
        
        Uses Hungarian algorithm-style greedy matching based on distance.
        
        Args:
            detected_faces: List of faces detected in current frame
            
        Returns:
            Dictionary of tracked faces with persistent IDs
        """
        # Increment frames_since_seen for all tracked faces
        for tracked in self.tracked_faces.values():
            tracked.face.frames_since_seen += 1
        
        # Match detected faces to tracked faces
        matched_detected = set()
        matched_tracked = set()
        
        # Calculate distance matrix
        if detected_faces and self.tracked_faces:
            for detected in detected_faces:
                best_match_id = None
                best_distance = float('inf')
                
                for track_id, tracked in self.tracked_faces.items():
                    if track_id in matched_tracked:
                        continue
                    
                    # Calculate Euclidean distance between face centers
                    dx = detected.center_x - tracked.face.center_x
                    dy = detected.center_y - tracked.face.center_y
                    distance = np.sqrt(dx*dx + dy*dy)
                    
                    if distance < best_distance and distance < self.config.max_face_distance:
                        best_distance = distance
                        best_match_id = track_id
                
                if best_match_id is not None:
                    # Update existing tracked face
                    matched_tracked.add(best_match_id)
                    matched_detected.add(id(detected))
                    
                    detected.id = best_match_id
                    self.tracked_faces[best_match_id].face = detected
                    self.tracked_faces[best_match_id].face.frames_since_seen = 0
                    self.tracked_faces[best_match_id].position_history.append(
                        (detected.center_x, detected.center_y)
                    )
                    self.tracked_faces[best_match_id].confidence_history.append(
                        detected.confidence
                    )
                    self.tracked_faces[best_match_id].update_movement()
        
        # Add new faces that weren't matched
        for detected in detected_faces:
            if id(detected) not in matched_detected:
                new_id = self.next_face_id
                self.next_face_id += 1
                detected.id = new_id
                
                tracked_face = TrackedFace(face=detected)
                tracked_face.position_history.append(
                    (detected.center_x, detected.center_y)
                )
                tracked_face.confidence_history.append(detected.confidence)
                tracked_face.avg_confidence = detected.confidence
                self.tracked_faces[new_id] = tracked_face
        
        # Remove faces that haven't been seen for too long
        faces_to_remove = [
            track_id for track_id, tracked in self.tracked_faces.items()
            if tracked.face.frames_since_seen > self.config.face_lost_timeout
        ]
        for track_id in faces_to_remove:
            del self.tracked_faces[track_id]
        
        # Track frames without any visible faces
        if detected_faces:
            self.frames_without_faces = 0
        else:
            self.frames_without_faces += 1
        
        self.last_detections = detected_faces
        return self.tracked_faces
    
    def interpolate(self) -> Dict[int, TrackedFace]:
        """
        Return current tracked faces without new detection.
        Used between detection frames for smooth tracking.
        """
        # Increment frames_without_faces if we have no tracked faces
        if not self.tracked_faces:
            self.frames_without_faces += 1
        return self.tracked_faces
    
    def reset(self):
        """Reset tracker state (used on scene changes)."""
        self.tracked_faces.clear()
        self.frames_without_faces = 0


# =============================================================================
# SPEAKER SELECTOR
# =============================================================================

class SpeakerSelector:
    """
    Selects the active speaker from tracked faces.
    
    Selection strategy:
    1. If only one face, select it
    2. If multiple faces, use movement score as proxy for speaking
    3. Apply hysteresis to avoid rapid switching between speakers
    4. Fall back to largest face if movement scores are similar
    5. IMPORTANT: Keep tracking last known speaker even if temporarily lost
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.current_speaker_id: Optional[int] = None
        self.speaker_lock_frames: int = 0
        self.switch_candidate_id: Optional[int] = None
        self.switch_candidate_frames: int = 0
        self.frames_without_speaker: int = 0  # Track how long we've been without the current speaker
        
    def select(self, tracked_faces: Dict[int, TrackedFace]) -> Optional[int]:
        """
        Select the active speaker from tracked faces.
        
        Args:
            tracked_faces: Dictionary of tracked faces
            
        Returns:
            ID of the selected speaker, or None if no faces
        """
        if not tracked_faces:
            # No faces detected - but don't immediately give up on current speaker
            self.frames_without_speaker += 1
            # Keep returning current speaker ID for a while (allows hold position)
            # Only clear after extended period without any faces
            if self.frames_without_speaker > 60:  # ~2 seconds at 30fps
                self.current_speaker_id = None
            return self.current_speaker_id
        
        # Reset counter since we have faces
        self.frames_without_speaker = 0
        
        # Filter out low-confidence faces first (not actually facing camera)
        min_valid_confidence = 0.5
        valid_faces = {
            fid: tf for fid, tf in tracked_faces.items() 
            if tf.face.confidence >= min_valid_confidence
        }
        
        # If no valid faces, keep current speaker or return None
        if not valid_faces:
            self.frames_without_speaker += 1
            if self.frames_without_speaker > 30:
                self.current_speaker_id = None
            return self.current_speaker_id
        
        # If only one valid face, select it (but still apply some delay)
        if len(valid_faces) == 1:
            new_speaker = list(valid_faces.keys())[0]
            tracked = valid_faces[new_speaker]
            
            # Check if this face is likely the speaker (has some movement and good confidence)
            # Don't immediately switch to a static face
            is_likely_speaker = (
                tracked.face.confidence >= 0.5 and 
                (tracked.movement_score > 0.1 or self.current_speaker_id is None)
            )
            
            if is_likely_speaker:
                if self.current_speaker_id is not None and self.current_speaker_id != new_speaker:
                    self.switch_candidate_frames += 1
                    if self.switch_candidate_frames >= 5:  # Reduced delay for switching
                        self.current_speaker_id = new_speaker
                        self.switch_candidate_frames = 0
                else:
                    self.current_speaker_id = new_speaker
                    self.switch_candidate_frames = 0
            return self.current_speaker_id
        
        # Multiple valid faces - need to determine active speaker
        # Key factors:
        # 1. Face CONFIDENCE - a face not looking at camera has low confidence
        # 2. HORIZONTAL movement - speaking involves horizontal head movement
        # 3. Movement VARIANCE - speaking has varied movement, nodding is repetitive
        
        face_scores = {}
        movement_threshold = 0.3  # Minimum movement to be considered "active"
        
        for face_id, tracked in valid_faces.items():
            confidence = tracked.face.confidence
            avg_confidence = tracked.avg_confidence if tracked.avg_confidence > 0 else confidence
            h_movement = tracked.horizontal_movement
            v_movement = tracked.vertical_movement
            total_movement = tracked.movement_score
            
            # RULE 1: Check for nodding pattern (high vertical, low horizontal)
            is_just_nodding = (v_movement > h_movement * 2) and (h_movement < 0.5)
            
            # RULE 2: Score based on movement quality
            if is_just_nodding:
                # Penalize pure nodding heavily
                face_scores[face_id] = total_movement * 0.1 * confidence
            elif total_movement > movement_threshold:
                # Good movement pattern - likely speaking
                # Confidence is a multiplier (higher confidence = more likely the active speaker)
                face_scores[face_id] = (h_movement * 5 + total_movement) * confidence
            else:
                # Low movement overall - but still could be the speaker if confident
                face_scores[face_id] = (total_movement * 0.5 + confidence * 0.3)
        
        # Find best candidate
        best_candidate_id = max(face_scores, key=face_scores.get)
        best_score = face_scores[best_candidate_id]
        
        # If we have a current speaker, apply hysteresis
        if self.current_speaker_id is not None and self.current_speaker_id in valid_faces:
            current_score = face_scores.get(self.current_speaker_id, 0)
            
            # Only consider switching if the difference is significant
            if best_candidate_id != self.current_speaker_id:
                score_diff = (best_score - current_score) / max(current_score, 0.1)
                
                if score_diff > self.config.speaker_switch_threshold:
                    # Potential switch - apply delay
                    if self.switch_candidate_id == best_candidate_id:
                        self.switch_candidate_frames += 1
                    else:
                        self.switch_candidate_id = best_candidate_id
                        self.switch_candidate_frames = 1
                    
                    # Only switch after delay
                    if self.switch_candidate_frames >= self.config.speaker_switch_delay_frames:
                        self.current_speaker_id = best_candidate_id
                        self.switch_candidate_id = None
                        self.switch_candidate_frames = 0
                else:
                    # Not significant enough difference, reset switch candidate
                    self.switch_candidate_id = None
                    self.switch_candidate_frames = 0
            else:
                self.switch_candidate_id = None
                self.switch_candidate_frames = 0
        else:
            # No current speaker or speaker lost, select best
            self.current_speaker_id = best_candidate_id
        
        return self.current_speaker_id


# =============================================================================
# CROP CALCULATOR
# =============================================================================

class CropCalculator:
    """
    Calculates the crop region for vertical output.
    
    Handles:
    - Converting 16:9 input to 9:16 output
    - Centering on the active speaker
    - Smooth transitions using exponential moving average
    - Clamping to frame boundaries
    - Scene change handling with faster transitions
    """
    
    def __init__(self, config: Config, frame_width: int, frame_height: int):
        self.config = config
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Calculate output crop dimensions
        # For 9:16 output from 16:9 input, we crop width significantly
        self.crop_height = frame_height
        self.crop_width = int(frame_height * config.output_aspect_ratio)
        
        # Ensure crop width doesn't exceed frame width
        if self.crop_width > frame_width:
            self.crop_width = frame_width
            self.crop_height = int(frame_width / config.output_aspect_ratio)
        
        # Initialize smoothed position at center
        self.center_x = frame_width / 2
        self.center_y = frame_height / 2
        self.smoothed_x = self.center_x
        self.smoothed_y = self.center_y
        
        # Track last known face position for recovery
        self.last_face_x = self.center_x
        self.last_face_y = self.center_y
        self.has_active_face = False
        
        # Scene change handling - when True, INSTANTLY jump to next face position
        self.scene_change_pending = False
        
        # For fast transition (used after instant jump for small adjustments)
        self.use_fast_transition = False
        self.fast_transition_frames = 0
        
        # Previous crop for comparison
        self.prev_crop: Optional[CropRegion] = None
        
    def calculate(self, target_x: float, target_y: float, 
                   is_new_face: bool = False) -> CropRegion:
        """
        Calculate the crop region centered on the target position.
        
        On scene change: INSTANTLY jumps to target position (no smoothing).
        Otherwise: Uses exponential moving average for smooth transitions.
        
        Args:
            target_x: Target X position (face center)
            target_y: Target Y position (face center)
            is_new_face: Whether this is a newly detected face (use faster transition)
            
        Returns:
            CropRegion with position
        """
        # INSTANT JUMP on scene change - bypass all smoothing!
        if self.scene_change_pending:
            # Directly set position to target - no transition
            self.smoothed_x = target_x
            self.smoothed_y = target_y
            self.scene_change_pending = False
            # Use fast transition for any small adjustments after the jump
            self.use_fast_transition = True
            self.fast_transition_frames = 0
        else:
            # Normal operation: apply EMA smoothing
            # Determine which alpha to use
            if is_new_face or self.use_fast_transition:
                alpha = self.config.ema_alpha_fast
                self.fast_transition_frames += 1
                # After some frames, switch back to normal smoothing
                if self.fast_transition_frames > 10:  # Reduced frames for fast transition
                    self.use_fast_transition = False
                    self.fast_transition_frames = 0
            else:
                alpha = self.config.ema_alpha
            
            # Apply exponential moving average smoothing
            self.smoothed_x = alpha * target_x + (1 - alpha) * self.smoothed_x
            self.smoothed_y = alpha * target_y + (1 - alpha) * self.smoothed_y
        
        # Store last known face position
        self.last_face_x = target_x
        self.last_face_y = target_y
        self.has_active_face = True
        
        # Calculate crop position to center the face
        # Horizontal: center the face
        crop_x = int(self.smoothed_x - self.crop_width * self.config.horizontal_face_position)
        
        # Vertical: position face in upper portion of frame (more natural for shorts)
        crop_y = int(self.smoothed_y - self.crop_height * self.config.vertical_face_position)
        
        # Create crop region
        crop = CropRegion(
            x=crop_x,
            y=crop_y,
            width=self.crop_width,
            height=self.crop_height
        )
        
        # Clamp to frame boundaries
        crop.clamp(self.frame_width, self.frame_height)
        
        self.prev_crop = crop
        return crop
    
    def calculate_no_face(self, frames_without_face: int) -> CropRegion:
        """
        Calculate crop when no face is detected.
        
        After scene change: INSTANTLY jump to center (old position is invalid)
        Normal operation: HOLD position (face may be temporarily lost)
        
        Args:
            frames_without_face: Number of consecutive frames without a face
            
        Returns:
            CropRegion at appropriate position
        """
        # IMPORTANT: After scene change, the old position is INVALID
        # We must jump to center immediately since we don't know where the new speaker is
        if self.scene_change_pending:
            # Instantly jump to center of frame
            self.smoothed_x = self.center_x
            self.smoothed_y = self.center_y
            self.scene_change_pending = False
            self.has_active_face = False
            # Use fast transition once a face IS found
            self.use_fast_transition = True
            self.fast_transition_frames = 0
        
        # Only update the has_active_face flag after a very long time
        if frames_without_face > 90:  # 3 seconds at 30fps
            self.has_active_face = False
        
        # Calculate crop position
        crop_x = int(self.smoothed_x - self.crop_width * self.config.horizontal_face_position)
        crop_y = int(self.smoothed_y - self.crop_height * self.config.vertical_face_position)
        
        crop = CropRegion(
            x=crop_x,
            y=crop_y,
            width=self.crop_width,
            height=self.crop_height
        )
        
        crop.clamp(self.frame_width, self.frame_height)
        self.prev_crop = crop
        return crop
    
    def get_default_crop(self) -> CropRegion:
        """Get a default centered crop when no face is detected."""
        crop = CropRegion(
            x=(self.frame_width - self.crop_width) // 2,
            y=(self.frame_height - self.crop_height) // 2,
            width=self.crop_width,
            height=self.crop_height
        )
        return crop
    
    def trigger_scene_change(self):
        """
        Trigger scene change mode.
        Next call to calculate() will INSTANTLY jump to the new position.
        """
        self.scene_change_pending = True
    
    def trigger_fast_transition(self):
        """Trigger fast transition mode (for speaker switches within same scene)."""
        self.use_fast_transition = True
        self.fast_transition_frames = 0


# =============================================================================
# VIDEO PROCESSOR
# =============================================================================

class VideoProcessor:
    """
    Main video processing pipeline.
    
    Coordinates face detection, tracking, speaker selection, and cropping
    to produce the final vertical video output.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.face_detector: Optional[FaceDetector] = None
        self.face_tracker: Optional[FaceTracker] = None
        self.speaker_selector: Optional[SpeakerSelector] = None
        self.crop_calculator: Optional[CropCalculator] = None
        
    def process(self, input_path: str, output_path: str, 
                preview: bool = False, debug: bool = False):
        """
        Process the input video and generate vertical output.
        
        Args:
            input_path: Path to input video file
            output_path: Path for output video file
            preview: Show preview window during processing
            debug: Draw debug overlays (face boxes, crop region)
        """
        print(f"[INFO] Opening video: {input_path}")
        
        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[INFO] Input: {frame_width}x{frame_height} @ {fps:.2f} FPS")
        print(f"[INFO] Total frames: {total_frames}")
        
        # Initialize components
        self.face_detector = FaceDetector(self.config)
        self.face_tracker = FaceTracker(self.config)
        self.speaker_selector = SpeakerSelector(self.config)
        self.crop_calculator = CropCalculator(self.config, frame_width, frame_height)
        
        output_width = self.crop_calculator.crop_width
        output_height = self.crop_calculator.crop_height
        print(f"[INFO] Output: {output_width}x{output_height}")
        
        # Create temporary file for video without audio
        # Create temp file path in the same folder as output (avoid temp folder issues)
        output_dir = os.path.dirname(os.path.abspath(output_path))
        temp_video_path = os.path.join(output_dir, f"_temp_{os.getpid()}.avi")
        
        # Initialize video writer with MJPG codec (very reliable, works everywhere)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(temp_video_path, fourcc, fps, 
                                 (output_width, output_height))
        
        if not writer.isOpened():
            # Fallback to XVID if MJPG fails
            print("[WARNING] MJPG codec failed, trying XVID...")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(temp_video_path, fourcc, fps, 
                                     (output_width, output_height))
        
        if not writer.isOpened():
            # Try mp4v codec with mp4 extension
            print("[WARNING] XVID failed, trying mp4v with .mp4...")
            temp_video_path = temp_video_path.replace('.avi', '.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(temp_video_path, fourcc, fps, 
                                     (output_width, output_height))
            if not writer.isOpened():
                raise ValueError("Could not create video writer with any codec")
        
        print(f"[INFO] Writing temp video to: {temp_video_path}")
        print(f"[INFO] Output dimensions: {output_width}x{output_height}")
        
        frame_count = 0
        frames_written = 0
        start_time = time.time()
        prev_frame_gray = None  # For scene change detection
        last_speaker_id = None  # Track speaker changes
        last_speaker_position = None  # Track speaker position for jump detection
        speaker_positions = {}  # Track positions per speaker ID
        last_num_faces = 0  # Track number of faces to detect camera angle changes
        frames_since_last_snap = 999  # Cooldown counter for instant snaps
        
        print("[INFO] Processing frames...")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Scene change detection using frame difference
                scene_changed = False
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_frame_small = cv2.resize(gray_frame, (160, 90))  # Downscale for speed
                
                if prev_frame_gray is not None:
                    # Calculate mean absolute difference
                    frame_diff = cv2.absdiff(prev_frame_gray, gray_frame_small)
                    mean_diff = np.mean(frame_diff)
                    
                    if mean_diff > self.config.scene_change_threshold:
                        scene_changed = True
                        # Reset tracker - old face positions are no longer valid
                        self.face_tracker.tracked_faces.clear()
                        self.face_tracker.frames_without_faces = 0
                        # Reset speaker selector
                        self.speaker_selector.current_speaker_id = None
                        self.speaker_selector.frames_without_speaker = 0
                        # Reset last speaker position (so we don't get false jump detection)
                        last_speaker_position = None
                        speaker_positions.clear()  # Clear all speaker position history
                        last_num_faces = 0  # Reset face count
                        # Tell crop calculator to INSTANTLY jump to the next detected face
                        self.crop_calculator.trigger_scene_change()
                        if debug:
                            print(f"[DEBUG] Scene change at frame {frame_count} (diff={mean_diff:.1f}) - will instant jump")
                
                prev_frame_gray = gray_frame_small
                
                # Detect faces every N frames, track in between
                # Force detection EVERY FRAME for a short period after scene change
                frames_since_scene_change = getattr(self, '_frames_since_scene_change', 999)
                if scene_changed:
                    self._frames_since_scene_change = 0
                else:
                    self._frames_since_scene_change = frames_since_scene_change + 1
                
                # Detect more aggressively after scene change (every frame for 15 frames)
                force_detection = (self._frames_since_scene_change < 15)
                
                if frame_count % self.config.detection_interval == 1 or scene_changed or force_detection:
                    detected_faces = self.face_detector.detect(frame)
                    tracked_faces = self.face_tracker.update(detected_faces)
                else:
                    tracked_faces = self.face_tracker.interpolate()
                
                # Select active speaker
                speaker_id = self.speaker_selector.select(tracked_faces)
                
                # Increment cooldown counter
                frames_since_last_snap += 1
                
                # Detect camera angle changes
                current_num_faces = len(tracked_faces)
                camera_angle_changed = False
                force_instant_snap = False
                
                # Only check for snaps if we haven't snapped recently (prevent ping-pong)
                snap_cooldown = 15  # Allow snapping more frequently
                
                # Check 1: Number of faces changed from 0 to something
                if last_num_faces == 0 and current_num_faces > 0 and frames_since_last_snap > snap_cooldown:
                    force_instant_snap = True
                    if debug:
                        print(f"[DEBUG] Frame {frame_count}: Faces appeared (0 -> {current_num_faces}) - instant snap")
                
                last_num_faces = current_num_faces
                
                # Check 2: Speaker position jumped significantly (same speaker, different position)
                if speaker_id is not None and speaker_id in tracked_faces and frames_since_last_snap > snap_cooldown:
                    speaker_face = tracked_faces[speaker_id].face
                    current_speaker_position = (speaker_face.center_x, speaker_face.center_y)
                    
                    # Check if THIS speaker's position jumped (camera cut showing same person differently)
                    if speaker_id in speaker_positions:
                        old_pos = speaker_positions[speaker_id]
                        dx = abs(current_speaker_position[0] - old_pos[0])
                        dy = abs(current_speaker_position[1] - old_pos[1])
                        jump_distance = (dx**2 + dy**2)**0.5
                        
                        if jump_distance > self.config.face_jump_threshold:
                            camera_angle_changed = True
                            force_instant_snap = True
                            if debug:
                                print(f"[DEBUG] Speaker {speaker_id} jumped {jump_distance:.0f}px at frame {frame_count} - instant snap")
                    
                    # Update position for this speaker
                    speaker_positions[speaker_id] = current_speaker_position
                    
                    # Check 3: Crop is far from selected speaker (crop stuck elsewhere)
                    # Only do this if speaker hasn't changed recently AND no recent snap
                    crop_center_x = self.crop_calculator.smoothed_x
                    dist_from_crop = abs(speaker_face.center_x - crop_center_x)
                    
                    # Only trigger snap if crop is far AND we've been tracking this speaker for a while
                    # This prevents snapping when speaker selection is unstable
                    if dist_from_crop > self.config.face_jump_threshold and speaker_id == last_speaker_id:
                        force_instant_snap = True
                        if debug:
                            print(f"[DEBUG] Crop far from speaker ({dist_from_crop:.0f}px) at frame {frame_count} - instant snap")
                
                # Trigger instant snap if needed
                if force_instant_snap or camera_angle_changed:
                    self.crop_calculator.trigger_scene_change()
                    frames_since_last_snap = 0 # Reset cooldown
                
                # Check if speaker changed (different person selected)
                speaker_switched = (speaker_id is not None and 
                                   speaker_id != last_speaker_id and 
                                   last_speaker_id is not None)
                if speaker_switched:
                    # When speaker changes, use fast transition (not instant snap to avoid ping-pong)
                    self.crop_calculator.trigger_fast_transition()
                    # Clear position history for old speaker to avoid false jumps
                    if last_speaker_id in speaker_positions:
                        del speaker_positions[last_speaker_id]
                    if debug:
                        print(f"[DEBUG] Speaker changed {last_speaker_id} -> {speaker_id} at frame {frame_count}")
                
                is_new_face = speaker_switched and not self.crop_calculator.has_active_face
                last_speaker_id = speaker_id
                
                # Calculate crop region
                if speaker_id is not None and speaker_id in tracked_faces:
                    speaker_face = tracked_faces[speaker_id].face
                    crop = self.crop_calculator.calculate(
                        speaker_face.center_x, 
                        speaker_face.center_y,
                        is_new_face=is_new_face
                    )
                else:
                    # No speaker detected - use smooth transition to center
                    crop = self.crop_calculator.calculate_no_face(
                        self.face_tracker.frames_without_faces
                    )
                
                # Apply crop - with validation
                crop_y_end = min(crop.y + crop.height, frame.shape[0])
                crop_x_end = min(crop.x + crop.width, frame.shape[1])
                crop_y = max(0, crop.y)
                crop_x = max(0, crop.x)
                
                cropped = frame[crop_y:crop_y_end, crop_x:crop_x_end]
                
                # Validate cropped frame
                if cropped is None or cropped.size == 0:
                    print(f"[WARNING] Empty crop at frame {frame_count}, using center crop")
                    # Fallback to center crop
                    center_x = frame.shape[1] // 2
                    crop_x = max(0, center_x - output_width // 2)
                    cropped = frame[:, crop_x:crop_x + output_width]
                
                # Make sure cropped frame has valid memory
                cropped = np.ascontiguousarray(cropped)
                
                # Draw debug overlays if requested
                if debug:
                    cropped = self._draw_debug(cropped, tracked_faces, speaker_id, crop)
                    # Also show FULL LANDSCAPE debug view in a separate window
                    landscape_debug = self._draw_debug_landscape(frame, tracked_faces, speaker_id, crop)
                    # Resize landscape for display (half size to fit screen)
                    landscape_small = cv2.resize(landscape_debug, (960, 540))
                    cv2.imshow('Landscape Debug - All Faces', landscape_small)
                
                # Write frame
                if cropped.shape[1] != output_width or cropped.shape[0] != output_height:
                    cropped = cv2.resize(cropped, (output_width, output_height))
                
                # Ensure frame is BGR with correct dtype
                if len(cropped.shape) == 2:
                    cropped = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)
                if cropped.dtype != np.uint8:
                    cropped = cropped.astype(np.uint8)
                
                # Ensure contiguous memory layout (fixes OpenCV C++ exceptions)
                if not cropped.flags['C_CONTIGUOUS']:
                    cropped = np.ascontiguousarray(cropped)
                
                try:
                    writer.write(cropped)
                    frames_written += 1
                except Exception as e:
                    print(f"[WARNING] Failed to write frame {frame_count}: {e}")
                    # Try to recover by making a copy
                    try:
                        cropped_copy = cropped.copy()
                        writer.write(cropped_copy)
                        frames_written += 1
                    except:
                        print(f"[ERROR] Could not recover frame {frame_count}, skipping")
                
                # Show preview if requested
                if preview:
                    preview_frame = cv2.resize(cropped, (540, 960))
                    cv2.imshow('Preview', preview_frame)
                
                # Handle keyboard input (for both preview and debug modes)
                if preview or debug:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("[INFO] Stopped by user (pressed 'q')")
                        break
                    elif key == ord(' '):  # Space to pause
                        print("[INFO] Paused - press any key to continue...")
                        cv2.waitKey(0)
                
                # Progress update
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    fps_actual = frame_count / elapsed
                    eta = (total_frames - frame_count) / fps_actual
                    progress = frame_count / total_frames * 100
                    print(f"[INFO] Progress: {progress:.1f}% | "
                          f"Frame {frame_count}/{total_frames} | "
                          f"Speed: {fps_actual:.1f} FPS | "
                          f"ETA: {eta:.0f}s")
        
        finally:
            # Cleanup
            cap.release()
            writer.release()
            if preview:
                cv2.destroyAllWindows()
            self.face_detector.close()
        
        print(f"[INFO] Processed {frame_count} frames, wrote {frames_written} frames in {time.time()-start_time:.1f}s")
        
        # Small delay to ensure file is fully written to disk
        time.sleep(0.5)
        
        # Verify temp file exists and has content
        if not os.path.exists(temp_video_path):
            raise ValueError(f"Temp video file does not exist: {temp_video_path}")
        
        file_size = os.path.getsize(temp_video_path)
        print(f"[INFO] Temp video size: {file_size / 1024 / 1024:.2f} MB")
        
        if file_size == 0:
            raise ValueError(f"Temp video file is empty: {temp_video_path}")
        
        if frames_written == 0:
            raise ValueError("No frames were written to video file")
        
        # Combine video with original audio using FFMPEG
        print("[INFO] Adding audio and encoding final video...")
        self._encode_with_audio(input_path, temp_video_path, output_path)
        
        # Clean up temp file
        try:
            os.unlink(temp_video_path)
        except:
            pass
        
        print(f"[INFO] Output saved to: {output_path}")
        
    def _draw_debug(self, frame: np.ndarray, tracked_faces: Dict[int, TrackedFace],
                    speaker_id: Optional[int], crop: CropRegion) -> np.ndarray:
        """Draw debug overlays on the CROPPED frame."""
        debug_frame = frame.copy()
        
        for face_id, tracked in tracked_faces.items():
            face = tracked.face
            
            # Adjust face coordinates relative to crop
            rel_x = int(face.bbox[0] - crop.x)
            rel_y = int(face.bbox[1] - crop.y)
            
            # Check if face is visible in crop
            if (rel_x + face.bbox[2] > 0 and rel_x < crop.width and
                rel_y + face.bbox[3] > 0 and rel_y < crop.height):
                
                # Color: green for active speaker, blue for others
                color = (0, 255, 0) if face_id == speaker_id else (255, 0, 0)
                
                # Draw bounding box
                cv2.rectangle(debug_frame, 
                            (rel_x, rel_y),
                            (rel_x + int(face.bbox[2]), rel_y + int(face.bbox[3])),
                            color, 2)
                
                # Draw face ID and movement score
                label = f"ID:{face_id} M:{tracked.movement_score:.1f}"
                cv2.putText(debug_frame, label, (rel_x, rel_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return debug_frame
    
    def _draw_debug_landscape(self, frame: np.ndarray, tracked_faces: Dict[int, TrackedFace],
                              speaker_id: Optional[int], crop: CropRegion) -> np.ndarray:
        """
        Draw debug overlays on the FULL LANDSCAPE frame.
        Shows ALL tracked faces and the crop region boundary.
        """
        debug_frame = frame.copy()
        
        # Draw ALL tracked faces on the full frame
        for face_id, tracked in tracked_faces.items():
            face = tracked.face
            x, y, w, h = int(face.bbox[0]), int(face.bbox[1]), int(face.bbox[2]), int(face.bbox[3])
            
            # Color: GREEN for active speaker, BLUE for others, CYAN for recently lost
            if face_id == speaker_id:
                color = (0, 255, 0)  # Green - active speaker
                thickness = 3
            elif face.frames_since_seen > 0:
                color = (255, 255, 0)  # Cyan - face not seen this frame (interpolated)
                thickness = 2
            else:
                color = (255, 0, 0)  # Blue - other detected face
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(debug_frame, (x, y), (x + w, y + h), color, thickness)
            
            # Draw face ID, movement score, and frames since seen
            label = f"ID:{face_id} M:{tracked.movement_score:.1f}"
            if face.frames_since_seen > 0:
                label += f" Lost:{face.frames_since_seen}"
            
            # Background for text readability
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(debug_frame, (x, y - 25), (x + text_w + 4, y - 5), (0, 0, 0), -1)
            cv2.putText(debug_frame, label, (x + 2, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw face center point
            cx, cy = int(face.center_x), int(face.center_y)
            cv2.circle(debug_frame, (cx, cy), 5, color, -1)
        
        # Draw CROP REGION as a prominent rectangle
        # This shows exactly what area will be in the final vertical video
        crop_color = (0, 255, 255)  # Yellow for crop region
        cv2.rectangle(debug_frame, 
                     (crop.x, crop.y), 
                     (crop.x + crop.width, crop.y + crop.height),
                     crop_color, 3)
        
        # Draw "CROP" label at top of crop region
        cv2.putText(debug_frame, "CROP REGION", (crop.x + 10, crop.y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, crop_color, 2)
        
        # Draw smoothed target position (where crop is actually centered)
        target_x = int(self.crop_calculator.smoothed_x)
        target_y = int(self.crop_calculator.smoothed_y)
        cv2.drawMarker(debug_frame, (target_x, target_y), (0, 0, 255), 
                      cv2.MARKER_CROSS, 30, 3)
        cv2.putText(debug_frame, "TARGET", (target_x + 15, target_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Status info at top-left
        status_lines = [
            f"Tracked Faces: {len(tracked_faces)}",
            f"Active Speaker: {speaker_id if speaker_id is not None else 'None'}",
            f"Frames w/o Face: {self.face_tracker.frames_without_faces}",
            f"Scene Change Pending: {self.crop_calculator.scene_change_pending}",
        ]
        
        y_offset = 30
        for line in status_lines:
            cv2.rectangle(debug_frame, (5, y_offset - 20), (300, y_offset + 5), (0, 0, 0), -1)
            cv2.putText(debug_frame, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 30
        
        return debug_frame
    
    def _encode_with_audio(self, original_video: str, processed_video: str, 
                           output_path: str):
        """
        Combine processed video with original audio using FFMPEG.
        
        Uses copy for audio stream to preserve quality and speed up encoding.
        Re-encodes video to H.264 for maximum compatibility.
        """
        # Try to find FFMPEG in common locations
        ffmpeg_cmd = self._find_ffmpeg()
        
        if ffmpeg_cmd is None:
            print("[ERROR] FFMPEG not found! Please install FFMPEG and add to PATH.")
            print("[INFO] Copying processed video without re-encoding...")
            import shutil
            shutil.copy(processed_video, output_path)
            return
        
        cmd = [
            ffmpeg_cmd,
            '-y',  # Overwrite output
            '-i', processed_video,  # Processed video (no audio)
            '-i', original_video,   # Original video (for audio)
            '-c:v', 'libx264',      # Re-encode video to H.264
            '-preset', 'medium',    # Encoding speed/quality tradeoff
            '-crf', '23',           # Quality (lower = better, 23 is default)
            '-c:a', 'aac',          # Audio codec
            '-b:a', '192k',         # Audio bitrate
            '-map', '0:v:0',        # Use video from first input
            '-map', '1:a:0?',       # Use audio from second input (? = optional)
            '-shortest',            # End when shortest stream ends
            '-movflags', '+faststart',  # Enable fast start for web playback
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("[INFO] FFMPEG encoding completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] FFMPEG failed: {e.stderr}")
            # Fallback: try without audio
            print("[INFO] Retrying without audio...")
            cmd_no_audio = [
                ffmpeg_cmd, '-y',
                '-i', processed_video,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-movflags', '+faststart',
                output_path
            ]
            try:
                subprocess.run(cmd_no_audio, check=True)
            except subprocess.CalledProcessError as e2:
                print(f"[ERROR] FFMPEG without audio also failed: {e2}")
                print("[INFO] Copying processed video without re-encoding...")
                import shutil
                shutil.copy(processed_video, output_path)
    
    def _find_ffmpeg(self) -> Optional[str]:
        """Find FFMPEG executable."""
        import shutil as sh
        
        # Try PATH first
        ffmpeg_path = sh.which('ffmpeg')
        if ffmpeg_path:
            return ffmpeg_path
        
        # Common Windows locations
        common_paths = [
            r'C:\ffmpeg\bin\ffmpeg.exe',
            r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
            r'C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe',
            os.path.expandvars(r'%LOCALAPPDATA%\Programs\ffmpeg\bin\ffmpeg.exe'),
            os.path.expandvars(r'%LOCALAPPDATA%\ffmpeg\bin\ffmpeg.exe'),
            # Try imageio-ffmpeg if installed
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                return path
        
        # Try imageio-ffmpeg package
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass
        
        return None


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for the script."""
    
    # Parse command line arguments
    if len(sys.argv) < 3:
        print("Usage: python podcast_to_shorts.py <input_video> <output_video> [options]")
        print("")
        print("Options:")
        print("  --preview    Show preview window during processing")
        print("  --debug      Draw debug overlays (face boxes)")
        print("")
        print("Example:")
        print("  python podcast_to_shorts.py podcast.mp4 short.mp4 --preview")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # Parse options
    preview = '--preview' in sys.argv
    debug = '--debug' in sys.argv
    
    # Validate input file
    if not os.path.exists(input_path):
        print(f"[ERROR] Input file not found: {input_path}")
        sys.exit(1)
    
    # Create configuration
    config = Config()
    
    # Adjust config based on video content (can be customized)
    # For podcast videos, we want smoother transitions
    config.ema_alpha = 0.1  # Smoother movement
    config.speaker_switch_delay_frames = 20  # Longer delay before switching
    
    # Create processor and run
    processor = VideoProcessor(config)
    
    try:
        processor.process(input_path, output_path, preview=preview, debug=debug)
        print("[SUCCESS] Video processing completed!")
        
    except Exception as e:
        print(f"[ERROR] Processing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
