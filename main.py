"""
Podcast to Shorts Pipeline
==========================

Complete pipeline that:
1. Takes a random video from input folder
2. Converts to vertical format using face tracking
3. Transcribes using AssemblyAI with word timestamps
4. Uses Gemini AI to highlight important words
5. Generates glow text images for subtitles (random fonts for highlights)
6. Generates viral hooks using CrewAI agents with SUBJECT/OBJECT highlights
7. Overlays subtitles and hook on video and exports
8. Optionally uploads to YouTube with AI-generated title/description

Requirements:
    pip install opencv-python mediapipe numpy assemblyai python-dotenv pillow langchain langchain-google-genai crewai crewai-tools google-auth google-auth-oauthlib google-api-python-client

Usage:
    python main.py
    python main.py --input video.mp4  # Process specific video
    python main.py --debug            # Show debug overlays
    python main.py --no-hook          # Disable hook generation
    python main.py --upload           # Upload to YouTube after processing
    python main.py --upload --privacy unlisted  # Upload as unlisted
"""

import os
import sys
import random
import json
import shutil
import tempfile
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from dotenv import load_dotenv
import cv2
import numpy as np

# Load environment variables from .env file
load_dotenv()

# Import from our modules
from face_tracker import VideoProcessor, Config
from subgen import create_glow_text_image
from hook_generator import HookGeneratorCrew, HookRenderer, HookResult
from youtube_uploader import YouTubeUploader, YouTubeConfig, VideoMetadataGenerator


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the full pipeline."""
    
    # Folders
    input_folder: str = "input"
    output_folder: str = "output"
    temp_folder: str = "temp"
    word_images_folder: str = "word_images"
    fonts_folder: str = "fonts"  # Folder for highlight fonts
    
    # Video settings
    supported_extensions: tuple = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    
    # Subtitle settings
    font_path: str = "BebasNeue-Regular.ttf"  # Default font for normal words
    font_size: int = 28  # Font size for SUBTITLES (smaller)
    text_color: tuple = (255, 255, 255)  # White
    glow_color: tuple = (255, 255, 255)  # White glow
    highlight_color: tuple = (255, 255, 0)  # Yellow for highlighted words
    subtitle_position: float = 0.6  # 60% from top (shifted down 10%)
    
    # Words to display at once (for subtitle grouping)
    words_per_subtitle: int = 3
    
    # Word spacing
    word_spacing: int = 1  # Reduced spacing between words
    
    # Max width for subtitles before wrapping to next line
    max_subtitle_width: int = 560  # Pixels (for ~607px wide vertical video)
    line_spacing: int = 8  # Vertical spacing between lines
    
    # Hook settings
    enable_hook_generation: bool = True  # Enable/disable hook generation
    hook_font_size: int = 48  # Font size for HOOK text (bigger than subtitles)
    hook_position_ratio: float = 0.10  # Position from top (within gradient)
    
    # YouTube upload settings
    enable_youtube_upload: bool = False  # Enable/disable YouTube upload
    youtube_client_secret: str = "virtualrealm_ytdata_api_client_secret.json"
    youtube_privacy: str = "public"  # public, private, or unlisted
    

# =============================================================================
# ASSEMBLYAI TRANSCRIPTION
# =============================================================================

class Transcriber:
    """
    Transcribes audio using AssemblyAI API.
    Returns word-level timestamps for precise subtitle placement.
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        
        # Import assemblyai here to allow graceful failure if not installed
        try:
            import assemblyai as aai
            aai.settings.api_key = api_key
            self.aai = aai
        except ImportError:
            raise ImportError("Please install assemblyai: pip install assemblyai")
    
    def transcribe(self, video_path: str) -> Dict:
        """
        Transcribe video and return word-level timestamps.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with 'text' and 'words' (list of word objects with timestamps)
        """
        print(f"[INFO] Uploading video for transcription...")
        
        config = self.aai.TranscriptionConfig(
            speech_model=self.aai.SpeechModel.best,
        )
        
        transcriber = self.aai.Transcriber(config=config)
        transcript = transcriber.transcribe(video_path)
        
        if transcript.status == self.aai.TranscriptStatus.error:
            raise Exception(f"Transcription failed: {transcript.error}")
        
        print(f"[INFO] Transcription completed!")
        
        # Extract word-level data
        words_data = []
        if transcript.words:
            for word in transcript.words:
                words_data.append({
                    'text': word.text,
                    'start': word.start,  # milliseconds
                    'end': word.end,      # milliseconds
                    'confidence': word.confidence
                })
        
        result = {
            'text': transcript.text,
            'words': words_data,
            'duration_ms': transcript.audio_duration * 1000 if transcript.audio_duration else 0
        }
        
        return result


# =============================================================================
# TEXT CLEANING UTILITIES
# =============================================================================

def clean_word(text: str) -> str:
    """
    Clean a word by removing punctuation except apostrophes.
    
    Args:
        text: Original word text
        
    Returns:
        Cleaned word text
    """
    # Remove all punctuation except apostrophe
    # Keep letters, numbers, and apostrophes
    cleaned = re.sub(r"[^\w\s']", "", text)
    return cleaned.strip()


def clean_transcription_words(words: List[Dict]) -> List[Dict]:
    """
    Clean all words in transcription by removing punctuation.
    
    Args:
        words: List of word dictionaries
        
    Returns:
        List of cleaned word dictionaries (empty words removed)
    """
    cleaned_words = []
    for word_data in words:
        cleaned_text = clean_word(word_data['text'])
        if cleaned_text:  # Only keep non-empty words
            cleaned_word = word_data.copy()
            cleaned_word['text'] = cleaned_text
            cleaned_words.append(cleaned_word)
    return cleaned_words


# =============================================================================
# GEMINI AI HIGHLIGHTER
# =============================================================================

class GeminiHighlighter:
    """
    Uses Google Gemini via LangChain to identify important words for highlighting.
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain.schema import HumanMessage, SystemMessage
            
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-pro",
                google_api_key=api_key,
                temperature=0.1
            )
            self.HumanMessage = HumanMessage
            self.SystemMessage = SystemMessage
            
        except ImportError:
            raise ImportError("Please install: pip install langchain langchain-google-genai")
    
    def highlight_words(self, words: List[Dict]) -> List[Dict]:
        """
        Analyze transcription and mark important words for highlighting.
        
        Args:
            words: List of word dictionaries with 'text', 'start', 'end'
            
        Returns:
            Same list with 'highlight' field added to each word
        """
        # Extract just the text for analysis
        full_text = " ".join([w['text'] for w in words])
        word_list = [w['text'] for w in words]
        
        print(f"[INFO] Analyzing {len(words)} words with Gemini AI...")
        
        # Create prompt for Gemini
        system_prompt = """You are an expert at identifying important, impactful, and emotional words in transcripts.
Your task is to identify words that should be HIGHLIGHTED in video subtitles to create engagement.

Highlight words that are:
- Key nouns (important subjects, objects, names)
- Strong verbs (action words)
- Emotional words (words that convey feeling)
- Numbers and statistics
- Surprising or unexpected words
- Words that carry the main meaning of a sentence

Do NOT highlight:
- Common words like "the", "a", "is", "are", "and", "but", "or"
- Prepositions like "in", "on", "at", "to", "from"
- Pronouns like "I", "you", "he", "she", "it", "we", "they"
- Helper words like "just", "very", "really", "actually"
- Do not highlight 3 words continiously

Return ONLY a JSON array of the exact words to highlight (case-sensitive, matching the input exactly).
Return approximately 5-15% of the total words as highlights."""

        user_prompt = f"""Here is the transcript:
"{full_text}"

Here is the list of all words (return only words from this exact list):
{json.dumps(word_list)}

Return a JSON array of words to highlight. Example format: ["word1", "word2", "word3"]
Only return the JSON array, nothing else."""

        try:
            messages = [
                self.SystemMessage(content=system_prompt),
                self.HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            response_text = response.content.strip()
            
            # Parse the JSON response
            # Handle case where response might have markdown code blocks
            if "```" in response_text:
                # Extract JSON from code block
                match = re.search(r'\[.*?\]', response_text, re.DOTALL)
                if match:
                    response_text = match.group()
            
            highlighted_words = json.loads(response_text)
            print(f"[INFO] Gemini identified {len(highlighted_words)} words to highlight")
            
            # Add highlight field to each word
            highlighted_set = set(highlighted_words)
            for word_data in words:
                word_data['highlight'] = word_data['text'] in highlighted_set
            
            # Count highlights
            highlight_count = sum(1 for w in words if w.get('highlight', False))
            print(f"[INFO] Highlighted {highlight_count}/{len(words)} words ({100*highlight_count/len(words):.1f}%)")
            
        except Exception as e:
            print(f"[WARNING] Gemini highlighting failed: {e}")
            print("[INFO] Proceeding without highlights")
            # Default: no highlights
            for word_data in words:
                word_data['highlight'] = False
        
        return words


# =============================================================================
# WORD IMAGE GENERATOR
# =============================================================================

class WordImageGenerator:
    """
    Generates glow text images for each unique word in transcription.
    Uses the subgen.py create_glow_text_image function.
    For highlighted words, uses random fonts from the fonts folder.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.word_images: Dict[str, str] = {}  # word -> image_path mapping
        self.highlight_fonts: List[str] = []  # Available fonts for highlights
        self._load_highlight_fonts()
        
    def _load_highlight_fonts(self):
        """Load available fonts from the fonts folder."""
        fonts_folder = self.config.fonts_folder
        if os.path.exists(fonts_folder):
            for file in os.listdir(fonts_folder):
                if file.lower().endswith(('.ttf', '.otf')):
                    self.highlight_fonts.append(os.path.join(fonts_folder, file))
        
        if self.highlight_fonts:
            print(f"[INFO] Found {len(self.highlight_fonts)} fonts for highlights")
        else:
            print(f"[INFO] No fonts found in '{fonts_folder}', using default for highlights")
    
    def _get_random_highlight_font(self) -> str:
        """Get a random font for highlighted words."""
        if self.highlight_fonts:
            return random.choice(self.highlight_fonts)
        return self.config.font_path
        
    def generate_word_images(self, words: List[Dict], output_folder: str) -> Dict[str, str]:
        """
        Generate images for all unique words.
        Highlighted words use random fonts from fonts folder.
        
        Args:
            words: List of word dictionaries from transcription
            output_folder: Folder to save word images
            
        Returns:
            Dictionary mapping word text to image path
        """
        # Create output folder
        os.makedirs(output_folder, exist_ok=True)
        
        # Build set of highlighted words
        highlighted_words = set(w['text'] for w in words if w.get('highlight', False))
        
        # Get unique words
        unique_words = set()
        for word_data in words:
            word = word_data['text'].strip()
            if word:
                unique_words.add(word)
        
        print(f"[INFO] Generating images for {len(unique_words)} unique words...")
        print(f"[INFO] {len(highlighted_words)} words will use highlight fonts")
        
        # Check if default font exists
        if not os.path.exists(self.config.font_path):
            print(f"[WARNING] Font not found: {self.config.font_path}")
            print("[INFO] Downloading default font...")
            self._download_default_font()
        
        # Generate image for each word
        for word in unique_words:
            is_highlighted = word in highlighted_words
            
            # Create safe filename (include _hl suffix for highlighted words)
            safe_name = "".join(c if c.isalnum() else "_" for c in word)
            suffix = "_hl" if is_highlighted else ""
            image_path = os.path.join(output_folder, f"{safe_name}{suffix}.png")
            
            # Skip if already generated
            if os.path.exists(image_path):
                self.word_images[word] = image_path
                continue
            
            # Choose font and color based on highlight status
            if is_highlighted:
                font_path = self._get_random_highlight_font()
                text_color = self.config.highlight_color
                glow_color = self.config.highlight_color
            else:
                font_path = self.config.font_path
                text_color = self.config.text_color
                glow_color = self.config.glow_color
            
            try:
                create_glow_text_image(
                    text=word,
                    font_path=font_path,
                    output_path=image_path,
                    text_color=text_color,
                    glow_color=glow_color,
                    font_size=self.config.font_size
                )
                self.word_images[word] = image_path
            except Exception as e:
                print(f"[WARNING] Failed to generate image for '{word}': {e}")
        
        print(f"[INFO] Generated {len(self.word_images)} word images")
        return self.word_images
    
    def _download_default_font(self):
        """Download a default font if not available."""
        import urllib.request
        
        # Use a free Google Font
        font_url = "https://github.com/google/fonts/raw/main/ofl/montserrat/Montserrat-Black.ttf"
        
        try:
            urllib.request.urlretrieve(font_url, self.config.font_path)
            print(f"[INFO] Font downloaded to: {self.config.font_path}")
        except Exception as e:
            print(f"[ERROR] Could not download font: {e}")
            print("[INFO] Please manually download Montserrat-Black.ttf")


# =============================================================================
# SUBTITLE OVERLAY
# =============================================================================

class SubtitleOverlay:
    """
    Overlays word images as subtitles on video frames.
    Syncs subtitle display with word timestamps.
    """
    
    def __init__(self, config: PipelineConfig, word_images: Dict[str, str]):
        self.config = config
        self.word_images = word_images
        self.loaded_images: Dict[str, np.ndarray] = {}
        
    def load_word_image(self, word: str) -> Optional[np.ndarray]:
        """Load and cache word image."""
        if word in self.loaded_images:
            return self.loaded_images[word]
        
        if word not in self.word_images:
            return None
        
        image_path = self.word_images[word]
        if not os.path.exists(image_path):
            return None
        
        # Load with alpha channel
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            self.loaded_images[word] = img
        
        return img
    
    def get_words_at_time(self, words: List[Dict], time_ms: float, 
                          num_words: int = 3) -> List[Dict]:
        """
        Get words that should be displayed at a given time.
        Groups words together for better readability.
        
        Args:
            words: List of word dictionaries with timestamps
            time_ms: Current time in milliseconds
            num_words: Number of words to show at once
            
        Returns:
            List of word dictionaries currently active
        """
        # Find the current word index
        current_idx = -1
        for i, word in enumerate(words):
            if word['start'] <= time_ms <= word['end']:
                current_idx = i
                break
            # Also check if we're between words (after one ended, before next starts)
            if i > 0 and words[i-1]['end'] < time_ms < word['start']:
                current_idx = i - 1
                break
        
        if current_idx == -1:
            return []
        
        # Calculate the group of words to show
        # Show words in groups of num_words, advancing as a group
        group_idx = current_idx // num_words
        start_idx = group_idx * num_words
        end_idx = min(len(words), start_idx + num_words)
        
        # Return the group of words
        return words[start_idx:end_idx]
    
    def overlay_subtitle(self, frame: np.ndarray, words: List[Dict], 
                         current_word_idx: int) -> np.ndarray:
        """
        Overlay subtitle text on frame.
        Handles multi-line when words are too wide for screen.
        
        Args:
            frame: Video frame (BGR)
            words: List of active word dictionaries
            current_word_idx: Index of currently spoken word (for highlighting)
            
        Returns:
            Frame with subtitle overlay
        """
        if not words:
            return frame
        
        frame_height, frame_width = frame.shape[:2]
        max_line_width = min(frame_width - 40, self.config.max_subtitle_width)  # Max width with margin
        
        # Load all word images
        word_imgs = []
        for word_data in words:
            word = word_data['text']
            img = self.load_word_image(word)
            if img is not None:
                word_imgs.append((img, word_data))
        
        if not word_imgs:
            # Fallback to simple text rendering
            return self._render_fallback_text(frame, words)
        
        # Calculate spacing
        spacing = self.config.word_spacing
        
        # Split words into lines based on max width
        lines = []
        current_line = []
        current_line_width = 0
        
        for img, word_data in word_imgs:
            img_width = img.shape[1]
            
            # Check if adding this word exceeds max width
            new_width = current_line_width + img_width
            if current_line:
                new_width += spacing  # Add spacing if not first word in line
            
            if new_width > max_line_width and current_line:
                # Start a new line
                lines.append(current_line)
                current_line = [(img, word_data)]
                current_line_width = img_width
            else:
                current_line.append((img, word_data))
                current_line_width = new_width
        
        # Don't forget the last line
        if current_line:
            lines.append(current_line)
        
        # Calculate total height of all lines
        line_heights = []
        for line in lines:
            max_height = max(img.shape[0] for img, _ in line)
            line_heights.append(max_height)
        
        line_spacing = self.config.line_spacing  # Vertical spacing between lines
        total_height = sum(line_heights) + line_spacing * (len(lines) - 1)
        
        # Calculate starting y position (center vertically)
        y_center = int(frame_height * self.config.subtitle_position)
        y_start = y_center - total_height // 2
        
        # Render each line
        y_current = y_start
        for line_idx, line in enumerate(lines):
            # Calculate line width
            line_width = sum(img.shape[1] for img, _ in line) + spacing * (len(line) - 1)
            
            # Center horizontally
            x_start = (frame_width - line_width) // 2
            x_current = x_start
            
            max_height = line_heights[line_idx]
            
            for img, word_data in line:
                # Center each word image vertically within the line
                y_offset = (max_height - img.shape[0]) // 2
                frame = self._overlay_image(frame, img, x_current, y_current + y_offset)
                x_current += img.shape[1] + spacing
            
            y_current += max_height + line_spacing
        
        return frame
    
    def _render_fallback_text(self, frame: np.ndarray, words: List[Dict]) -> np.ndarray:
        """Fallback text rendering when word images are not available."""
        frame_height, frame_width = frame.shape[:2]
        subtitle_text = " ".join([w['text'] for w in words])
        y_position = int(frame_height * self.config.subtitle_position)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3
        
        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(
            subtitle_text, font, font_scale, thickness
        )
        
        # Center text
        x = (frame_width - text_width) // 2
        y = y_position
        
        # Draw shadow
        cv2.putText(frame, subtitle_text, (x + 2, y + 2), font, 
                   font_scale, (0, 0, 0), thickness + 2)
        
        # Draw text
        cv2.putText(frame, subtitle_text, (x, y), font, 
                   font_scale, (255, 255, 255), thickness)
        
        return frame
    
    def apply_top_gradient(self, frame: np.ndarray, hook_image: Any = None, 
                           y_position: int = 30, padding: int = 20) -> np.ndarray:
        """
        Apply a dark black to transparent gradient at the top of the frame.
        Gradient size is based on hook image height if provided.
        
        Args:
            frame: Video frame (BGR)
            hook_image: Hook image to determine gradient height (BGRA with shape)
            y_position: Y position where hook starts
            padding: Extra padding below hook for gradient fade
            
        Returns:
            Frame with gradient overlay
        """
        frame_height, frame_width = frame.shape[:2]
        
        # Calculate gradient height based on hook image
        if hook_image is not None:
            hook_height = hook_image.shape[0]
            # Gradient covers: y_position + hook_height + padding for fade
            gradient_height = y_position + hook_height + padding
        else:
            # Fallback: 20% of frame height
            gradient_height = int(frame_height * 0.20)
        
        # Ensure gradient doesn't exceed frame height
        gradient_height = min(gradient_height, frame_height)
        
        # Solid black portion at very top (first 40% of gradient area)
        solid_black_height = int(gradient_height * 0.4)
        
        # Apply solid black to top portion
        frame[:solid_black_height] = 0  # Fully black
        
        # Create gradient for remaining portion
        # Use power curve (exponent < 1) to keep it darker longer before fading
        fade_height = gradient_height - solid_black_height
        if fade_height > 0:
            for y in range(solid_black_height, gradient_height):
                # Calculate normalized position within fade zone (0 to 1)
                fade_progress = (y - solid_black_height) / fade_height
                # Power curve: stays darker longer, then fades more quickly at the end
                alpha = (1.0 - fade_progress) ** 0.5
                # Apply darkening to the row
                frame[y] = (frame[y] * (1 - alpha)).astype(np.uint8)
        
        return frame
    
    def _overlay_image(self, background: np.ndarray, overlay: np.ndarray,
                       x: int, y: int) -> np.ndarray:
        """
        Overlay an image with alpha channel onto background.
        
        Args:
            background: Background image (BGR)
            overlay: Overlay image (BGRA with alpha)
            x, y: Position to place overlay
            
        Returns:
            Combined image
        """
        if overlay is None:
            return background
        
        bg_h, bg_w = background.shape[:2]
        ov_h, ov_w = overlay.shape[:2]
        
        # Clamp position
        if x < 0:
            overlay = overlay[:, -x:]
            ov_w = overlay.shape[1]
            x = 0
        if y < 0:
            overlay = overlay[-y:, :]
            ov_h = overlay.shape[0]
            y = 0
        
        # Clamp size
        if x + ov_w > bg_w:
            overlay = overlay[:, :bg_w - x]
            ov_w = overlay.shape[1]
        if y + ov_h > bg_h:
            overlay = overlay[:bg_h - y, :]
            ov_h = overlay.shape[0]
        
        if ov_w <= 0 or ov_h <= 0:
            return background
        
        # Extract alpha channel if present
        if overlay.shape[2] == 4:
            alpha = overlay[:, :, 3] / 255.0
            alpha = np.stack([alpha] * 3, axis=-1)
            
            overlay_rgb = overlay[:, :, :3]
            
            # Blend
            roi = background[y:y+ov_h, x:x+ov_w]
            blended = (alpha * overlay_rgb + (1 - alpha) * roi).astype(np.uint8)
            background[y:y+ov_h, x:x+ov_w] = blended
        else:
            background[y:y+ov_h, x:x+ov_w] = overlay
        
        return background


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class PodcastToShortsPipeline:
    """
    Main pipeline orchestrating the entire conversion process.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        
        # Get API keys from environment
        self.assemblyai_key = os.getenv("ASSEMBLYAI_API_KEY")
        self.gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        
        if not self.assemblyai_key:
            print("[WARNING] ASSEMBLYAI_API_KEY not found in .env file")
            print("[INFO] Transcription will be skipped")
        
        if not self.gemini_key:
            print("[WARNING] GOOGLE_API_KEY not found in .env file")
            print("[INFO] Word highlighting will be skipped")
        
        # Create necessary folders
        os.makedirs(self.config.input_folder, exist_ok=True)
        os.makedirs(self.config.output_folder, exist_ok=True)
        os.makedirs(self.config.temp_folder, exist_ok=True)
        os.makedirs(self.config.word_images_folder, exist_ok=True)
        os.makedirs(self.config.fonts_folder, exist_ok=True)
    
    def get_random_video(self) -> Optional[str]:
        """Get a random video file from the input folder."""
        videos = []
        
        for file in os.listdir(self.config.input_folder):
            if file.lower().endswith(self.config.supported_extensions):
                videos.append(os.path.join(self.config.input_folder, file))
        
        if not videos:
            print(f"[ERROR] No video files found in '{self.config.input_folder}' folder")
            return None
        
        selected = random.choice(videos)
        print(f"[INFO] Selected video: {selected}")
        return selected
    
    def process(self, input_video: Optional[str] = None, debug: bool = False):
        """
        Run the full pipeline.
        
        Args:
            input_video: Optional specific video path. If None, picks random from input folder
            debug: Show debug overlays during face tracking
        """
        # Step 1: Get input video
        if input_video is None:
            input_video = self.get_random_video()
        
        if input_video is None or not os.path.exists(input_video):
            print("[ERROR] No valid input video")
            return
        
        video_name = Path(input_video).stem
        
        # =============================================================================
        # TESTING MODE: Load from previously stored files instead of making API calls
        # =============================================================================
        USE_CACHED_DATA = False  # Set to False to use real API calls
        ALWAYS_REPROCESS_VIDEO = True  # Set to True to always re-run face tracking
        
        if USE_CACHED_DATA:
            print("\n" + "="*60)
            print("TESTING MODE: Loading cached data from temp folder")
            print("="*60)
            
            # Video processing
            vertical_video_path = os.path.join(
                self.config.temp_folder, 
                f"{video_name}_vertical.mp4"
            )
            
            if ALWAYS_REPROCESS_VIDEO or not os.path.exists(vertical_video_path):
                print("[INFO] Running face tracking...")
                face_config = Config()
                face_config.ema_alpha = 0.1
                face_config.speaker_switch_delay_frames = 20
                processor = VideoProcessor(face_config)
                processor.process(input_video, vertical_video_path, debug=debug)
            else:
                print(f"[INFO] Using cached vertical video: {vertical_video_path}")
            
            # Load transcription with highlights
            transcription_data = None
            highlighted_path = os.path.join(self.config.temp_folder, f"{video_name}_transcription_highlighted.json")
            regular_path = os.path.join(self.config.temp_folder, f"{video_name}_transcription.json")
            
            if os.path.exists(highlighted_path):
                print(f"[INFO] Loading cached highlighted transcription: {highlighted_path}")
                with open(highlighted_path, 'r', encoding='utf-8') as f:
                    transcription_data = json.load(f)
            elif os.path.exists(regular_path):
                print(f"[INFO] Loading cached transcription: {regular_path}")
                with open(regular_path, 'r', encoding='utf-8') as f:
                    transcription_data = json.load(f)
                # Add highlight=False if not present
                for word in transcription_data.get('words', []):
                    if 'highlight' not in word:
                        word['highlight'] = False
            else:
                print("[WARNING] No cached transcription found!")
            
            if transcription_data:
                print(f"[INFO] Loaded {len(transcription_data.get('words', []))} words")
            
            # Load hook data
            hook_result = None
            hook_image = None
            hook_path = os.path.join(self.config.temp_folder, f"{video_name}_hook.json")
            
            if os.path.exists(hook_path) and self.config.enable_hook_generation:
                print(f"[INFO] Loading cached hook: {hook_path}")
                with open(hook_path, 'r', encoding='utf-8') as f:
                    hook_data = json.load(f)
                
                hook_result = HookResult(
                    hook_text=hook_data.get('hook_text', ''),
                    subject_word=hook_data.get('subject_word'),
                    object_word=hook_data.get('object_word'),
                    scores=hook_data.get('scores', {}),
                    reasoning=hook_data.get('reasoning', '')
                )
                print(f"[INFO] Loaded hook: {hook_result.hook_text}")
                
                # Create hook image
                hook_renderer = HookRenderer(
                    font_path=self.config.font_path,
                    font_size=self.config.hook_font_size
                )
                hook_image = hook_renderer.create_hook_image(
                    hook_result,
                    max_width=380
                )
                print("[INFO] Hook image created from cached data")
            else:
                print("[INFO] No cached hook found or hook generation disabled")
            
            # Generate word images (still needed even in test mode)
            print("\n[INFO] Generating word images...")
            word_images = {}
            if transcription_data and transcription_data.get('words'):
                generator = WordImageGenerator(self.config)
                word_images = generator.generate_word_images(
                    transcription_data['words'],
                    self.config.word_images_folder
                )
        
        else:
            # =============================================================================
            # PRODUCTION MODE: Real API calls (commented out for testing)
            # =============================================================================
            
            # Step 2: Convert to vertical format using face tracker
            print("\n" + "="*60)
            print("STEP 1: Converting to vertical format (face tracking)")
            print("="*60)
            
            vertical_video_path = os.path.join(
                self.config.temp_folder, 
                f"{video_name}_vertical.mp4"
            )
            
            face_config = Config()
            face_config.ema_alpha = 0.1
            face_config.speaker_switch_delay_frames = 20
            
            processor = VideoProcessor(face_config)
            processor.process(input_video, vertical_video_path, debug=debug)
            
            print(f"[INFO] Vertical video saved to: {vertical_video_path}")
            
            # Step 3: Transcribe with AssemblyAI
            print("\n" + "="*60)
            print("STEP 2: Transcribing audio with AssemblyAI")
            print("="*60)
            
            transcription_data = None
            
            if self.assemblyai_key:
                try:
                    transcriber = Transcriber(self.assemblyai_key)
                    transcription_data = transcriber.transcribe(vertical_video_path)
                    
                    # Clean words (remove punctuation except apostrophe)
                    print("[INFO] Cleaning transcription (removing punctuation)...")
                    transcription_data['words'] = clean_transcription_words(transcription_data['words'])
                    
                    # Save transcription to JSON
                    transcription_path = os.path.join(
                        self.config.temp_folder,
                        f"{video_name}_transcription.json"
                    )
                    with open(transcription_path, 'w', encoding='utf-8') as f:
                        json.dump(transcription_data, f, indent=2, ensure_ascii=False)
                    
                    print(f"[INFO] Transcription saved to: {transcription_path}")
                    print(f"[INFO] Found {len(transcription_data['words'])} words")
                    
                except Exception as e:
                    print(f"[ERROR] Transcription failed: {e}")
                    transcription_data = None
            else:
                print("[INFO] Skipping transcription (no API key)")
            
            # Step 4: Highlight important words with Gemini AI
            print("\n" + "="*60)
            print("STEP 3: Highlighting important words with Gemini AI")
            print("="*60)
            
            if transcription_data and transcription_data['words'] and self.gemini_key:
                try:
                    highlighter = GeminiHighlighter(self.gemini_key)
                    transcription_data['words'] = highlighter.highlight_words(transcription_data['words'])
                    
                    # Save updated transcription with highlights
                    transcription_path = os.path.join(
                        self.config.temp_folder,
                        f"{video_name}_transcription_highlighted.json"
                    )
                    with open(transcription_path, 'w', encoding='utf-8') as f:
                        json.dump(transcription_data, f, indent=2, ensure_ascii=False)
                    
                    print(f"[INFO] Highlighted transcription saved to: {transcription_path}")
                    
                except Exception as e:
                    print(f"[ERROR] Highlighting failed: {e}")
                    # Continue without highlights
                    for word in transcription_data.get('words', []):
                        word['highlight'] = False
            else:
                if transcription_data and transcription_data['words']:
                    print("[INFO] Skipping highlighting (no Gemini API key)")
                    for word in transcription_data['words']:
                        word['highlight'] = False
            
            # Step 5: Generate word images
            print("\n" + "="*60)
            print("STEP 4: Generating word images")
            print("="*60)
            
            word_images = {}
            
            if transcription_data and transcription_data['words']:
                generator = WordImageGenerator(self.config)
                word_images = generator.generate_word_images(
                    transcription_data['words'],
                    self.config.word_images_folder
                )
            else:
                print("[INFO] Skipping word image generation (no transcription)")
            
            # Step 6: Generate hook with CrewAI
            print("\n" + "="*60)
            print("STEP 5: Generating hook with CrewAI agents")
            print("="*60)
            
            hook_result = None
            hook_image = None
            
            if self.config.enable_hook_generation and transcription_data and self.gemini_key:
                try:
                    hook_crew = HookGeneratorCrew(self.gemini_key)
                    hook_result = hook_crew.generate_hook(transcription_data['text'])
                    
                    print(f"[INFO] Generated hook: {hook_result.hook_text}")
                    print(f"[INFO] Subject (Yellow): {hook_result.subject_word}")
                    print(f"[INFO] Object (Purple): {hook_result.object_word}")
                    
                    # Create hook image with smaller width for proper centering
                    hook_renderer = HookRenderer(
                        font_path=self.config.font_path,
                        font_size=self.config.hook_font_size
                    )
                    hook_image = hook_renderer.create_hook_image(
                        hook_result,
                        max_width=380  # Smaller width for vertical video centering
                    )
                    print("[INFO] Hook image created successfully")
                    
                    # Save hook info to JSON
                    hook_path = os.path.join(
                        self.config.temp_folder,
                        f"{video_name}_hook.json"
                    )
                    with open(hook_path, 'w', encoding='utf-8') as f:
                        json.dump({
                            'hook_text': hook_result.hook_text,
                            'subject_word': hook_result.subject_word,
                            'object_word': hook_result.object_word,
                            'scores': hook_result.scores,
                            'reasoning': hook_result.reasoning
                        }, f, indent=2, ensure_ascii=False)
                    print(f"[INFO] Hook saved to: {hook_path}")
                    
                except Exception as e:
                    print(f"[WARNING] Hook generation failed: {e}")
                    import traceback
                    traceback.print_exc()
                    hook_result = None
                    hook_image = None
            else:
                if not self.config.enable_hook_generation:
                    print("[INFO] Hook generation disabled in config")
                elif not transcription_data:
                    print("[INFO] Skipping hook generation (no transcription)")
                else:
                    print("[INFO] Skipping hook generation (no Gemini API key)")
        
        # =============================================================================
        # END OF TESTING/PRODUCTION MODE SWITCH
        # =============================================================================
        
        # Step 7: Overlay subtitles and hook, then export
        print("\n" + "="*60)
        print("STEP 6: Overlaying subtitles, hook, and exporting")
        print("="*60)
        
        output_video_path = os.path.join(
            self.config.output_folder,
            f"{video_name}_final.mp4"
        )
        
        if transcription_data and transcription_data['words']:
            self._add_subtitles_to_video(
                vertical_video_path,
                output_video_path,
                transcription_data['words'],
                word_images,
                hook_image=hook_image
            )
        else:
            # No transcription - just copy the vertical video
            print("[INFO] No transcription available, copying vertical video as final")
            shutil.copy(vertical_video_path, output_video_path)
        
        # Cleanup: Delete word images folder contents
        self._cleanup_word_images()
        
        # Step 8: Upload to YouTube (optional)
        youtube_result = None
        if self.config.enable_youtube_upload:
            print("\n" + "="*60)
            print("STEP 7: Uploading to YouTube")
            print("="*60)
            
            youtube_result = self._upload_to_youtube(
                output_video_path,
                transcription_data,
                hook_result
            )
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETE!")
        print("="*60)
        print(f"[SUCCESS] Final video saved to: {output_video_path}")
        
        if youtube_result and youtube_result.get('success'):
            print(f"[SUCCESS] YouTube URL: {youtube_result.get('url')}")
        
        return output_video_path
    
    def _upload_to_youtube(self, video_path: str, transcription_data: Optional[Dict],
                            hook_result: Optional['HookResult']) -> Optional[Dict]:
        """
        Upload the final video to YouTube with AI-generated metadata.
        
        Args:
            video_path: Path to the final video
            transcription_data: Transcription data with text
            hook_result: Hook generation result
            
        Returns:
            YouTube upload result dictionary
        """
        try:
            # Get transcript text
            transcript = ""
            if transcription_data:
                transcript = transcription_data.get('text', '')
            
            # Get hook text if available
            hook_text = None
            if hook_result:
                hook_text = hook_result.hook_text
            
            # Configure YouTube uploader
            yt_config = YouTubeConfig(
                client_secret_file=self.config.youtube_client_secret,
                privacy_status=self.config.youtube_privacy
            )
            
            uploader = YouTubeUploader(yt_config)
            
            # Upload with AI-generated metadata
            result = uploader.upload_short(
                video_path=video_path,
                transcript=transcript,
                hook=hook_text,
                gemini_api_key=self.gemini_key,
                privacy_status=self.config.youtube_privacy
            )
            
            return result
            
        except Exception as e:
            print(f"[ERROR] YouTube upload failed: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    def _cleanup_word_images(self):
        """Delete all images from the word_images folder."""
        word_images_folder = self.config.word_images_folder
        if os.path.exists(word_images_folder):
            try:
                for file in os.listdir(word_images_folder):
                    file_path = os.path.join(word_images_folder, file)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                print(f"[INFO] Cleaned up word images from '{word_images_folder}'")
            except Exception as e:
                print(f"[WARNING] Failed to cleanup word images: {e}")
    
    def _add_subtitles_to_video(self, input_path: str, output_path: str,
                                 words: List[Dict], word_images: Dict[str, str],
                                 hook_image: Optional[np.ndarray] = None):
        """
        Add subtitle overlays and hook to video.
        
        Args:
            input_path: Input video path
            output_path: Output video path
            words: Word list with timestamps
            word_images: Dictionary mapping words to image paths
            hook_image: Optional hook image to overlay at top (BGRA numpy array)
        """
        print(f"[INFO] Adding subtitles to video...")
        if hook_image is not None:
            print(f"[INFO] Hook will be displayed at top of video")
        
        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[INFO] Video: {frame_width}x{frame_height} @ {fps:.2f} FPS")
        
        # Create temp file in output directory (avoid temp folder permission issues)
        output_dir = os.path.dirname(os.path.abspath(output_path))
        temp_video_path = os.path.join(output_dir, f"_temp_subtitle_{os.getpid()}.avi")
        
        # Initialize video writer with MJPG codec (very reliable)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(temp_video_path, fourcc, fps, 
                                 (frame_width, frame_height))
        
        if not writer.isOpened():
            # Fallback to XVID
            print("[WARNING] MJPG codec failed, trying XVID...")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(temp_video_path, fourcc, fps, 
                                     (frame_width, frame_height))
            if not writer.isOpened():
                raise ValueError("Could not create video writer")
        
        print(f"[INFO] Writing temp video to: {temp_video_path}")
        
        # Initialize subtitle overlay
        overlay = SubtitleOverlay(self.config, word_images)
        
        # Initialize hook renderer for overlay
        hook_renderer = HookRenderer(
            font_path=self.config.font_path,
            font_size=self.config.hook_font_size
        ) if hook_image is not None else None
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Calculate current time in milliseconds
            current_time_ms = (frame_count / fps) * 1000
            
            # Get words to display at this time
            active_words = overlay.get_words_at_time(
                words, current_time_ms, 
                self.config.words_per_subtitle
            )
            
            # Apply top gradient (sized to match hook image)
            frame = overlay.apply_top_gradient(
                frame, 
                hook_image=hook_image,
                y_position=30,
                padding=25  # Extra fade below hook
            )
            
            # Overlay hook at top (fixed 30px from top, within gradient area)
            if hook_image is not None:
                frame = hook_renderer.overlay_hook_on_frame(
                    frame, hook_image, 
                    y_position=30  # Fixed pixel position from top
                )
            
            # Overlay subtitle
            if active_words:
                frame = overlay.overlay_subtitle(frame, active_words, 0)
            
            writer.write(frame)
            
            # Progress update
            if frame_count % 100 == 0:
                progress = frame_count / total_frames * 100
                print(f"[INFO] Subtitle overlay progress: {progress:.1f}%")
        
        cap.release()
        writer.release()
        
        # Small delay to ensure file is fully written to disk
        import time
        time.sleep(0.5)
        
        # Verify temp file exists and has content
        if not os.path.exists(temp_video_path) or os.path.getsize(temp_video_path) == 0:
            raise ValueError(f"Temp video file is empty or missing: {temp_video_path}")
        
        print(f"[INFO] Adding audio track...")
        
        # Combine with audio using FFMPEG
        self._encode_with_audio(input_path, temp_video_path, output_path)
        
        # Cleanup
        try:
            os.unlink(temp_video_path)
        except:
            pass
    
    def _encode_with_audio(self, original_video: str, processed_video: str,
                           output_path: str):
        """Combine processed video with original audio using FFMPEG."""
        import subprocess
        
        # Try to find FFMPEG
        ffmpeg_cmd = self._find_ffmpeg()
        
        if ffmpeg_cmd is None:
            print("[WARNING] FFMPEG not found, copying without re-encoding...")
            shutil.copy(processed_video, output_path)
            return
        
        cmd = [
            ffmpeg_cmd,
            '-y',
            '-i', processed_video,
            '-i', original_video,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-map', '0:v:0',
            '-map', '1:a:0?',
            '-shortest',
            '-movflags', '+faststart',
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("[INFO] FFMPEG encoding completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"[WARNING] FFMPEG failed: {e.stderr}")
            print("[INFO] Copying without audio...")
            shutil.copy(processed_video, output_path)
    
    def _find_ffmpeg(self) -> Optional[str]:
        """Find FFMPEG executable."""
        # Try PATH first
        ffmpeg_path = shutil.which('ffmpeg')
        if ffmpeg_path:
            return ffmpeg_path
        
        # Common Windows locations
        common_paths = [
            r'C:\ffmpeg\bin\ffmpeg.exe',
            r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
            r'C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe',
            os.path.expandvars(r'%LOCALAPPDATA%\Programs\ffmpeg\bin\ffmpeg.exe'),
            os.path.expandvars(r'%LOCALAPPDATA%\ffmpeg\bin\ffmpeg.exe'),
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
# ENTRY POINT
# =============================================================================

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert podcast videos to vertical shorts with subtitles"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input video path (if not specified, picks random from input folder)"
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Show debug overlays during face tracking"
    )
    parser.add_argument(
        "--font",
        type=str,
        default="BebasNeue-Regular.ttf",
        help="Path to font file for subtitles"
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=40,
        help="Font size for subtitles (default: 40)"
    )
    parser.add_argument(
        "--hook-font-size",
        type=int,
        default=48,
        help="Font size for hook text (default: 48)"
    )
    parser.add_argument(
        "--no-hook",
        action="store_true",
        help="Disable hook generation"
    )
    parser.add_argument(
        "--upload", "-u",
        action="store_true",
        help="Upload final video to YouTube"
    )
    parser.add_argument(
        "--privacy",
        type=str,
        choices=['public', 'private', 'unlisted'],
        default='public',
        help="YouTube video privacy status (default: public)"
    )
    parser.add_argument(
        "--youtube-secret",
        type=str,
        default="virtualrealm_ytdata_api_client_secret.json",
        help="Path to YouTube OAuth client secret file"
    )
    
    args = parser.parse_args()
    
    # Create config
    config = PipelineConfig()
    config.font_path = args.font
    config.font_size = args.font_size
    config.hook_font_size = args.hook_font_size
    config.enable_hook_generation = not args.no_hook
    config.enable_youtube_upload = args.upload
    config.youtube_privacy = args.privacy
    config.youtube_client_secret = args.youtube_secret
    
    # Create and run pipeline
    pipeline = PodcastToShortsPipeline(config)
    
    try:
        pipeline.process(input_video=args.input, debug=args.debug)
    except KeyboardInterrupt:
        print("\n[INFO] Processing interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
