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
from speaker_overlay import SpeakerInfo, create_speaker_info_ass, apply_speaker_overlay_ffmpeg
from instagram_uploader import InstagramUploader, InstagramConfig
from hook_discord_selector import run_music_selector_bot, MusicSelectionResult


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
    hook_width_ratio: float = 0.90  # Hook covers 90% of video width
    hook_prefix: str = "$8B Fintech CEO: "  # Prefix to add before the hook text
    use_discord_for_hooks: bool = True  # Use Discord for hook selection
    discord_hook_timeout: int = 1200  # Seconds to wait for Discord response (20 min)
    
    # Logo settings
    logos_folder: str = "logos"  # Folder containing logo images
    logo_white: str = "logo-white.png"  # White logo for dark backgrounds
    logo_black: str = "logo-black.png"  # Black/dark logo for light backgrounds
    logo_width_ratio: float = 0.45  # Logo covers 30% of video width
    logo_margin_right: int = 20  # Pixels from right edge
    logo_margin_bottom: int = 20  # Pixels from bottom edge
    
    # Speaker info overlay settings (animated text at bottom of video)
    enable_speaker_overlay: bool = False  # DISABLED - speaker info overlay
    speaker_name: str = "Jack Zhang"  # Speaker's name
    speaker_title: str = "Co-founder of Airwallex"  # Speaker's title/role
    speaker_net_worth: str = "($8,000,000,000 Net Worth)"  # Net worth info ($ will be red)
    speaker_font_size_ratio: float = 0.038 # Font size as 1.6% of video width (reduced by 80%)
    speaker_position_from_bottom: float = 0.22  # 25% from bottom
    speaker_start_time: float = 0.5  # When the animation starts (seconds)
    speaker_display_duration: float = 9999.0  # Stay visible forever (entire video duration)
    
    # YouTube upload settings
    enable_youtube_upload: bool = False  # Enable/disable YouTube upload
    youtube_client_secret: str = "virtualrealm_ytdata_api_client_secret.json"
    youtube_privacy: str = "public"  # public, private, or unlisted
    youtube_mention: str = "@airwallex"  # Mention to add to YouTube title
    
    # Instagram upload settings
    enable_instagram_upload: bool = False  # Enable/disable Instagram upload
    instagram_tags: list = None  # Tags for Instagram caption
    instagram_audio_name: str = "Original Audio"  # Audio attribution
    instagram_thumb_offset: int = 500  # Thumbnail offset in ms
    
    # Background music settings
    enable_bg_music: bool = True  # Enable/disable background music
    default_bg_music: str = "bg_music.mp3"  # Default background music file
    bg_music_volume: float = 0.48  # Background music volume (0.0 to 1.0)
    use_discord_for_music: bool = True  # Use Discord for music selection
    discord_music_timeout: int = 1200  # Seconds to wait for Discord music selection
    
    # Smart audio sync settings (sync music drop with voice peak)
    enable_smart_sync: bool = True  # Enable intelligent music-voice sync
    smart_sync_voice_peak_time: float = None  # Manual override for voice peak time (seconds), None = auto-detect
    enable_audio_ducking: bool = True  # Lower music during speech, boost at climax
    ducking_low_volume: float = 0.15  # Music volume during speech
    ducking_high_volume: float = 0.6  # Music volume at climax/drops
    ducking_attack_ms: int = 50  # How fast to duck (ms)
    ducking_release_ms: int = 300  # How fast to unduck (ms)
    
    def __post_init__(self):
        if self.instagram_tags is None:
            self.instagram_tags = ["@airwallex", "@awxblackjz"]
    

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
        Uses smooth blending to avoid visible cutoff lines.
        
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
            # Add extra fade zone for smooth blending
            gradient_height = y_position + hook_height + padding + 60  # Extra smooth fade
        else:
            # Fallback: 25% of frame height for better fade
            gradient_height = int(frame_height * 0.25)
        
        # Ensure gradient doesn't exceed frame height
        gradient_height = min(gradient_height, frame_height)
        
        # Solid black portion at very top (first 30% of gradient area - reduced for smoother blend)
        solid_black_height = int(gradient_height * 0.30)
        
        # Apply solid black to top portion
        frame[:solid_black_height] = 0  # Fully black
        
        # Create gradient for remaining portion
        # Use a smoother sigmoid-like curve for natural fade
        fade_height = gradient_height - solid_black_height
        if fade_height > 0:
            for y in range(solid_black_height, gradient_height):
                # Calculate normalized position within fade zone (0 to 1)
                fade_progress = (y - solid_black_height) / fade_height
                
                # Use a smooth curve that starts slow, accelerates, then slows down
                # This creates a more natural blend without harsh cutoff
                # Using a combination of smooth-step and ease-out for better results
                # Smooth-step: 3x^2 - 2x^3 creates smooth S-curve
                smooth_progress = fade_progress * fade_progress * (3.0 - 2.0 * fade_progress)
                
                # Calculate alpha (1.0 = full black, 0.0 = transparent)
                # Invert so we fade FROM dark TO transparent
                alpha = 1.0 - smooth_progress
                
                # Apply the darkening as a blend with black
                # This avoids the harsh line by gradually reducing the intensity
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
            self._last_random_video = input_video  # Track for deletion
        else:
            self._last_random_video = None
        
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
                
                # Create hook image - width will be set dynamically based on video width
                hook_renderer = HookRenderer(
                    font_path=self.config.font_path,
                    font_size=self.config.hook_font_size
                )
                # Placeholder max_width - will be recalculated when we know video dimensions
                hook_image = hook_renderer.create_hook_image(
                    hook_result,
                    max_width=500  # Will be adjusted later based on video width
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
                    hook_result = hook_crew.generate_hook(
                        transcription_data['text'],
                        use_discord=self.config.use_discord_for_hooks,
                        discord_timeout=self.config.discord_hook_timeout
                    )
                    
                    # Set the hook prefix from config
                    hook_result.prefix = self.config.hook_prefix
                    
                    print(f"[INFO] Generated hook: {hook_result.hook_text}")
                    print(f"[INFO] Hook prefix: {hook_result.prefix}")
                    print(f"[INFO] Subject (Yellow): {hook_result.subject_words}")
                    print(f"[INFO] Object (Purple): {hook_result.object_words}")
                    
                    # Create hook image - width will be set dynamically based on video width
                    hook_renderer = HookRenderer(
                        font_path=self.config.font_path,
                        font_size=self.config.hook_font_size
                    )
                    # Placeholder max_width - will be adjusted later based on video width
                    hook_image = hook_renderer.create_hook_image(
                        hook_result,
                        max_width=500  # Will be adjusted later based on video width
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
                            'prefix': hook_result.prefix,
                            'subject_words': hook_result.subject_words,
                            'object_words': hook_result.object_words,
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
                hook_image=hook_image,
                hook_result=hook_result  # Pass hook_result for dynamic width calculation
            )
        else:
            # No transcription - just copy the vertical video
            print("[INFO] No transcription available, copying vertical video as final")
            shutil.copy(vertical_video_path, output_video_path)
        
        # Cleanup: Delete word images folder contents
        self._cleanup_word_images()
        
        # Step 8: Add background music (optional)
        if self.config.enable_bg_music:
            print("\n" + "="*60)
            print("STEP 7: Adding Background Music")
            print("="*60)
            
            # Get music selection (Discord or default)
            music_path = self._select_background_music()
            
            if music_path and os.path.exists(music_path):
                # Create temp path for video with music
                video_with_music = os.path.join(
                    self.config.temp_folder,
                    f"{video_name}_with_music.mp4"
                )
                
                # Mix background music
                success = self._add_background_music(
                    output_video_path,
                    music_path,
                    video_with_music
                )
                
                if success:
                    # Replace original output with music version
                    shutil.move(video_with_music, output_video_path)
                    print(f"[SUCCESS] Background music added from: {music_path}")
                else:
                    print("[WARNING] Failed to add background music, using original video")
            else:
                print(f"[WARNING] Music file not found: {music_path}")
        
        # Step 9: Upload to YouTube (optional)
        youtube_result = None
        youtube_caption = None
        if self.config.enable_youtube_upload:
            print("\n" + "="*60)
            print("STEP 8: Uploading to YouTube")
            print("="*60)
            
            youtube_result = self._upload_to_youtube(
                output_video_path,
                transcription_data,
                hook_result
            )
            
            # Store caption for Instagram
            if youtube_result and youtube_result.get('metadata'):
                youtube_caption = youtube_result['metadata'].get('description', '')
        
        # Step 10: Upload to Instagram (optional)
        instagram_result = None
        if self.config.enable_instagram_upload:
            print("\n" + "="*60)
            print("STEP 9: Uploading to Instagram")
            print("="*60)
            
            # Use YouTube caption if available, otherwise create a basic one
            caption = youtube_caption or "Sharp insights that challenge conventional thinking."
            title = ""
            if youtube_result and youtube_result.get('metadata'):
                title = youtube_result['metadata'].get('title', '')
            
            instagram_result = self._upload_to_instagram(
                output_video_path,
                caption,
                title
            )
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETE!")
        print("="*60)
        print(f"[SUCCESS] Final video saved to: {output_video_path}")

        if youtube_result and youtube_result.get('success'):
            print(f"[SUCCESS] YouTube URL: {youtube_result.get('url')}")

        if instagram_result and instagram_result.get('permalink'):
            print(f"[SUCCESS] Instagram URL: {instagram_result.get('permalink')}")

        # Delete the input video if it was randomly chosen and pipeline succeeded
        if self._last_random_video and os.path.exists(self._last_random_video):
            try:
                os.remove(self._last_random_video)
                print(f"[INFO] Deleted input video: {self._last_random_video}")
            except Exception as e:
                print(f"[WARNING] Failed to delete input video: {e}")

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
            
            # Generate metadata first using Gemini
            api_key = self.gemini_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            
            if api_key:
                generator = VideoMetadataGenerator(api_key)
                metadata = generator.generate_metadata(transcript, hook_text)
            else:
                print("[WARNING] No Gemini API key, using default metadata")
                metadata = {
                    'title': "You Won't Believe This! 🔥 #Shorts",
                    'description': "Sharp insights that challenge conventional wisdom and drive growth.",
                    'tags': ['shorts', 'viral', 'trending']
                }
            
            # Add YouTube mention to title if configured
            if self.config.youtube_mention:
                original_title = metadata['title']
                metadata['title'] = f"{self.config.youtube_mention} {original_title}"
                print(f"[INFO] Added mention to title: {metadata['title']}")
            
            # Upload with custom metadata (bypassing upload_short's metadata generation)
            result = uploader.upload_video(
                video_path=video_path,
                title=metadata['title'],
                description=metadata['description'],
                tags=metadata['tags'],
                privacy_status=self.config.youtube_privacy
            )
            
            # Add metadata to result
            result['metadata'] = metadata
            
            # Save upload record
            uploader._save_upload_record(video_path, result)
            
            return result
            
        except Exception as e:
            print(f"[ERROR] YouTube upload failed: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    def _upload_to_instagram(self, video_path: str, caption: str, title: str = "") -> Optional[Dict]:
        """
        Upload the final video to Instagram Reels.
        
        Args:
            video_path: Path to the final video
            caption: Caption for the Reel (typically the YouTube description)
            title: Title for logging purposes
            
        Returns:
            Instagram upload result dictionary with media_id and permalink
        """
        try:
            # Configure Instagram uploader
            ig_config = InstagramConfig(
                thumb_offset=self.config.instagram_thumb_offset,
                audio_name=self.config.instagram_audio_name,
                tags=self.config.instagram_tags
            )
            
            uploader = InstagramUploader(ig_config)
            
            # Upload to Instagram
            result = uploader.upload(
                video_path=video_path,
                caption=caption,
                title=title
            )
            
            return result
            
        except Exception as e:
            print(f"[ERROR] Instagram upload failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _select_background_music(self) -> str:
        """
        Select background music via Discord or use default.
        
        Returns:
            Path to the music file to use
        """
        default_music = self.config.default_bg_music
        
        # Check if Discord selection is enabled
        if self.config.use_discord_for_music:
            try:
                print(f"[MUSIC] Posting music selection to Discord (timeout: {self.config.discord_music_timeout}s)...")
                
                result = run_music_selector_bot(
                    default_music_path=default_music,
                    timeout_seconds=self.config.discord_music_timeout,
                    download_folder=self.config.temp_folder
                )
                
                if result.is_custom:
                    print(f"[MUSIC] Using custom music: {result.music_path}")
                    if result.source_url:
                        print(f"[MUSIC] Source: {result.source_url}")
                elif result.timed_out:
                    print(f"[MUSIC] Discord timed out, using default: {default_music}")
                else:
                    print(f"[MUSIC] User selected default music: {default_music}")
                
                return result.music_path
                
            except Exception as e:
                print(f"[MUSIC] Discord error: {e}, using default music")
                return default_music
        else:
            print(f"[MUSIC] Discord disabled, using default: {default_music}")
            return default_music
    
    def _analyze_music_drop(self, music_path: str) -> float:
        """
        Analyze music to find the 'drop' - the moment with highest sudden increase in energy.
        
        Args:
            music_path: Path to the music file
            
        Returns:
            Time in seconds where the drop occurs
        """
        try:
            import librosa
            import numpy as np
            
            print("[SYNC] Analyzing music for the 'drop'...")
            
            # Load music audio
            y_music, sr_music = librosa.load(music_path)
            
            # Calculate 'onset strength' (sudden changes in audio)
            # This finds the "hit" of the beat better than just volume
            onset_env = librosa.onset.onset_strength(y=y_music, sr=sr_music)
            
            # Find the frame with the highest energy jump (The Drop)
            drop_frame = np.argmax(onset_env)
            music_drop_time = librosa.frames_to_time(drop_frame, sr=sr_music)
            
            print(f"[SYNC] Music drop found at: {music_drop_time:.2f} seconds")
            return music_drop_time
            
        except ImportError:
            print("[SYNC] librosa not installed. Install with: pip install librosa")
            return 0.0
        except Exception as e:
            print(f"[SYNC] Error analyzing music: {e}")
            return 0.0
    
    def _analyze_voice_peak(self, video_path: str) -> float:
        """
        Analyze voice/speech to find the 'peak' - the loudest/most intense moment.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Time in seconds where the voice peak occurs
        """
        try:
            import librosa
            import numpy as np
            
            print("[SYNC] Analyzing voice for the 'climax'...")
            
            # Load audio from video
            y_voice, sr_voice = librosa.load(video_path)
            
            # Calculate RMS energy (loudness over time)
            rms = librosa.feature.rms(y=y_voice)[0]
            
            # Find the loudest moment
            peak_frame = np.argmax(rms)
            voice_peak_time = librosa.frames_to_time(peak_frame, sr=sr_voice)
            
            print(f"[SYNC] Voice peak found at: {voice_peak_time:.2f} seconds")
            return voice_peak_time
            
        except ImportError:
            print("[SYNC] librosa not installed. Install with: pip install librosa")
            return 0.0
        except Exception as e:
            print(f"[SYNC] Error analyzing voice: {e}")
            return 0.0
    
    def _add_background_music(self, video_path: str, music_path: str, output_path: str) -> bool:
        """
        Add background music to video with smart sync and optional ducking.
        
        Smart Sync Logic:
        - Find the 'drop' in the music (highest energy spike)
        - Find the 'peak' in the voice (loudest moment)
        - Crop/offset the music so these two moments align perfectly
        
        Example: If voice peak is at 5s and music drop is at 30s,
        we skip the first 25s of music so the drop happens at 5s.
        
        Args:
            video_path: Path to the input video
            music_path: Path to the background music file
            output_path: Path for the output video
            
        Returns:
            True if successful, False otherwise
        """
        import subprocess
        
        # Find FFmpeg
        ffmpeg_cmd = "ffmpeg"
        for path in ["ffmpeg", "ffmpeg.exe", r"C:\ffmpeg\bin\ffmpeg.exe"]:
            try:
                subprocess.run([path, "-version"], capture_output=True, check=True)
                ffmpeg_cmd = path
                break
            except:
                continue
        
        try:
            # Calculate where to start the music for peak alignment
            music_seek_time = 0.0  # How many seconds to skip into the music
            music_delay_ms = 0     # How many ms to delay music start (if needed)
            
            if self.config.enable_smart_sync:
                print("\n[SYNC] ═══════════════════════════════════════════")
                print("[SYNC] SMART AUDIO SYNC - Aligning music drop with voice peak")
                print("[SYNC] ═══════════════════════════════════════════")
                
                # Step 1: Find the music drop (highest energy spike)
                music_drop_time = self._analyze_music_drop(music_path)
                
                # Step 2: Find the voice peak (loudest/most intense moment)
                if self.config.smart_sync_voice_peak_time is not None:
                    voice_peak_time = self.config.smart_sync_voice_peak_time
                    print(f"[SYNC] Voice peak (manual): {voice_peak_time:.2f}s")
                else:
                    voice_peak_time = self._analyze_voice_peak(video_path)
                
                # Step 3: Calculate how to align them
                # We want: when video reaches voice_peak_time, music should be at music_drop_time
                # 
                # If voice peak is at 5s and music drop is at 30s:
                #   - We need to skip 25s of music (start music from 25s mark)
                #   - So at video time 5s, we're at music time 30s (the drop!)
                #
                # If voice peak is at 20s and music drop is at 10s:
                #   - We can't skip negative time, so we delay music start by 10s
                #   - Music starts at video time 10s, drop happens at video time 20s
                
                offset = music_drop_time - voice_peak_time
                
                print(f"\n[SYNC] Calculation:")
                print(f"[SYNC]   Voice peak at: {voice_peak_time:.2f}s (in video)")
                print(f"[SYNC]   Music drop at: {music_drop_time:.2f}s (in music file)")
                print(f"[SYNC]   Offset needed: {offset:.2f}s")
                
                if offset >= 0:
                    # Music drop is later than voice peak
                    # Solution: Skip into the music file
                    music_seek_time = offset
                    print(f"\n[SYNC] → Cropping first {music_seek_time:.2f}s of music")
                    print(f"[SYNC] → Music drop will now align with voice peak at {voice_peak_time:.2f}s")
                else:
                    # Music drop is earlier than voice peak
                    # Solution: Delay the music start
                    music_delay_ms = int(abs(offset) * 1000)
                    print(f"\n[SYNC] → Delaying music start by {abs(offset):.2f}s")
                    print(f"[SYNC] → Music drop will align with voice peak at {voice_peak_time:.2f}s")
                
                print("[SYNC] ═══════════════════════════════════════════\n")
            
            # Build FFmpeg filter based on settings
            bg_volume = self.config.bg_music_volume
            
            if self.config.enable_audio_ducking:
                # Audio ducking: Lower music during speech, boost during silence/climax
                high_vol = self.config.ducking_high_volume
                attack = self.config.ducking_attack_ms / 1000.0
                release = self.config.ducking_release_ms / 1000.0
                
                if music_delay_ms > 0:
                    # With delay
                    filter_complex = (
                        f"[1:a]adelay={music_delay_ms}|{music_delay_ms},volume={high_vol}[music_delayed];"
                        f"[music_delayed][0:a]sidechaincompress=threshold=0.02:ratio=8:attack={attack}:release={release}:level_sc=1[ducked];"
                        f"[0:a][ducked]amix=inputs=2:duration=first:dropout_transition=2[aout]"
                    )
                else:
                    # Without delay
                    filter_complex = (
                        f"[1:a]volume={high_vol}[music_vol];"
                        f"[music_vol][0:a]sidechaincompress=threshold=0.02:ratio=8:attack={attack}:release={release}:level_sc=1[ducked];"
                        f"[0:a][ducked]amix=inputs=2:duration=first:dropout_transition=2[aout]"
                    )
                print(f"[MUSIC] Audio ducking enabled (volume: {high_vol})")
            else:
                # Simple volume mix without ducking
                if music_delay_ms > 0:
                    filter_complex = (
                        f"[1:a]adelay={music_delay_ms}|{music_delay_ms},volume={bg_volume}[music];"
                        f"[0:a][music]amix=inputs=2:duration=first:dropout_transition=2[aout]"
                    )
                else:
                    filter_complex = (
                        f"[1:a]volume={bg_volume}[music];"
                        f"[0:a][music]amix=inputs=2:duration=first:dropout_transition=2[aout]"
                    )
            
            # Build FFmpeg command
            cmd = [ffmpeg_cmd, "-y"]
            
            # Input 0: Video with voice
            cmd.extend(["-i", video_path])
            
            # Input 1: Music (with seek if needed to crop the beginning)
            if music_seek_time > 0:
                cmd.extend(["-ss", f"{music_seek_time:.3f}"])
            
            cmd.extend(["-stream_loop", "-1"])  # Loop music if shorter than video
            cmd.extend(["-i", music_path])
            
            # Apply filter and output
            cmd.extend([
                "-filter_complex", filter_complex,
                "-map", "0:v",  # Use video from first input
                "-map", "[aout]",  # Use mixed audio
                "-c:v", "copy",  # Copy video codec (no re-encoding)
                "-c:a", "aac",  # Encode audio as AAC
                "-b:a", "192k",
                "-shortest",  # End when shortest input ends
                output_path
            ])
            
            print(f"[MUSIC] Mixing background music...")
            result = subprocess.run(cmd, check=True, capture_output=True)
            
            print(f"[MUSIC] Successfully added background music with smart sync")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"[MUSIC] FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
            return False
        except Exception as e:
            print(f"[MUSIC] Error adding background music: {e}")
            import traceback
            traceback.print_exc()
            return False
    
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
    
    def _load_logos(self, frame_width: int, frame_height: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load and resize logo images for overlay.
        
        Args:
            frame_width: Video frame width
            frame_height: Video frame height
            
        Returns:
            Tuple of (white_logo, black_logo) as numpy arrays with alpha channel
        """
        logo_white = None
        logo_black = None
        
        # Calculate target logo width (30% of video width)
        target_width = int(frame_width * self.config.logo_width_ratio)
        
        # Load white logo
        white_path = os.path.join(self.config.logos_folder, self.config.logo_white)
        if os.path.exists(white_path):
            logo_white = cv2.imread(white_path, cv2.IMREAD_UNCHANGED)
            if logo_white is not None:
                # Resize maintaining aspect ratio
                h, w = logo_white.shape[:2]
                scale = target_width / w
                new_h = int(h * scale)
                logo_white = cv2.resize(logo_white, (target_width, new_h), interpolation=cv2.INTER_AREA)
                print(f"[INFO] Loaded white logo: {target_width}x{new_h}")
        else:
            print(f"[WARNING] White logo not found: {white_path}")
        
        # Load black logo
        black_path = os.path.join(self.config.logos_folder, self.config.logo_black)
        if os.path.exists(black_path):
            logo_black = cv2.imread(black_path, cv2.IMREAD_UNCHANGED)
            if logo_black is not None:
                # Resize maintaining aspect ratio
                h, w = logo_black.shape[:2]
                scale = target_width / w
                new_h = int(h * scale)
                logo_black = cv2.resize(logo_black, (target_width, new_h), interpolation=cv2.INTER_AREA)
                print(f"[INFO] Loaded black logo: {target_width}x{new_h}")
        else:
            print(f"[WARNING] Black logo not found: {black_path}")
        
        return logo_white, logo_black
    
    def _detect_background_brightness(self, frame: np.ndarray, logo_width: int, logo_height: int,
                                        margin_right: int, margin_bottom: int) -> bool:
        """
        Detect if the background where logo will be placed is dark or light.
        
        Args:
            frame: Video frame (BGR)
            logo_width: Width of the logo
            logo_height: Height of the logo
            margin_right: Right margin for logo placement
            margin_bottom: Bottom margin for logo placement
            
        Returns:
            True if background is dark (use white logo), False if light (use black logo)
        """
        frame_h, frame_w = frame.shape[:2]
        
        # Calculate logo placement region
        x_start = frame_w - logo_width - margin_right
        y_start = frame_h - logo_height - margin_bottom
        x_end = frame_w - margin_right
        y_end = frame_h - margin_bottom
        
        # Clamp to valid region
        x_start = max(0, x_start)
        y_start = max(0, y_start)
        x_end = min(frame_w, x_end)
        y_end = min(frame_h, y_end)
        
        # Extract the region where logo will be placed
        region = frame[y_start:y_end, x_start:x_end]
        
        if region.size == 0:
            return True  # Default to dark background
        
        # Convert to grayscale and calculate mean brightness
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        # Threshold: if brightness < 128, consider it dark
        return mean_brightness < 128
    
    def _overlay_logo(self, frame: np.ndarray, logo: np.ndarray,
                      margin_right: int, margin_bottom: int) -> np.ndarray:
        """
        Overlay logo on the bottom-right corner of the frame.
        
        Args:
            frame: Video frame (BGR)
            logo: Logo image with alpha channel (BGRA)
            margin_right: Pixels from right edge
            margin_bottom: Pixels from bottom edge
            
        Returns:
            Frame with logo overlay
        """
        if logo is None:
            return frame
        
        frame_h, frame_w = frame.shape[:2]
        logo_h, logo_w = logo.shape[:2]
        
        # Calculate position (bottom-right corner)
        x = frame_w - logo_w - margin_right
        y = frame_h - logo_h - margin_bottom
        
        # Clamp position
        if x < 0:
            logo = logo[:, -x:]
            logo_w = logo.shape[1]
            x = 0
        if y < 0:
            logo = logo[-y:, :]
            logo_h = logo.shape[0]
            y = 0
        
        # Clamp size
        if x + logo_w > frame_w:
            logo = logo[:, :frame_w - x]
            logo_w = logo.shape[1]
        if y + logo_h > frame_h:
            logo = logo[:frame_h - y, :]
            logo_h = logo.shape[0]
        
        if logo_w <= 0 or logo_h <= 0:
            return frame
        
        # Extract alpha channel if present
        if logo.shape[2] == 4:
            alpha = logo[:, :, 3] / 255.0
            alpha = np.stack([alpha] * 3, axis=-1)
            logo_rgb = logo[:, :, :3]
            
            # Blend
            roi = frame[y:y+logo_h, x:x+logo_w]
            blended = (alpha * logo_rgb + (1 - alpha) * roi).astype(np.uint8)
            frame[y:y+logo_h, x:x+logo_w] = blended
        else:
            frame[y:y+logo_h, x:x+logo_w] = logo
        
        return frame
    
    def _add_subtitles_to_video(self, input_path: str, output_path: str,
                                 words: List[Dict], word_images: Dict[str, str],
                                 hook_image: Optional[np.ndarray] = None,
                                 hook_result: Optional['HookResult'] = None):
        """
        Add subtitle overlays, hook, and logo to video.
        
        Args:
            input_path: Input video path
            output_path: Output video path
            words: Word list with timestamps
            word_images: Dictionary mapping words to image paths
            hook_image: Optional hook image to overlay at top (BGRA numpy array)
            hook_result: Optional hook result for regenerating hook with correct width
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
        
        # Regenerate hook image with correct width (80% of video width)
        if hook_result is not None:
            hook_max_width = int(frame_width * self.config.hook_width_ratio)
            hook_renderer = HookRenderer(
                font_path=self.config.font_path,
                font_size=self.config.hook_font_size
            )
            hook_image = hook_renderer.create_hook_image(
                hook_result,
                max_width=hook_max_width
            )
            print(f"[INFO] Hook regenerated with width: {hook_max_width}px (80% of {frame_width}px)")
        
        # Load logos
        logo_white, logo_black = self._load_logos(frame_width, frame_height)
        has_logos = logo_white is not None or logo_black is not None
        if has_logos:
            print(f"[INFO] Logo overlay enabled (30% of video width)")
        
        # Get logo dimensions for brightness detection
        logo_height = 0
        logo_width = 0
        if logo_white is not None:
            logo_height, logo_width = logo_white.shape[:2]
        elif logo_black is not None:
            logo_height, logo_width = logo_black.shape[:2]
        
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
        hook_renderer_overlay = HookRenderer(
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
            
            # Detect background brightness and select appropriate logo BEFORE other overlays
            if has_logos and logo_width > 0:
                is_dark_bg = self._detect_background_brightness(
                    frame, logo_width, logo_height,
                    self.config.logo_margin_right, self.config.logo_margin_bottom
                )
                current_logo = logo_white if is_dark_bg else logo_black
                # Fallback if one logo is missing
                if current_logo is None:
                    current_logo = logo_white or logo_black
            else:
                current_logo = None
            
            # Apply top gradient (sized to match hook image)
            frame = overlay.apply_top_gradient(
                frame, 
                hook_image=hook_image,
                y_position=30,
                padding=25  # Extra fade below hook
            )
            
            # Overlay hook at top (fixed 30px from top, within gradient area)
            if hook_image is not None and hook_renderer_overlay is not None:
                frame = hook_renderer_overlay.overlay_hook_on_frame(
                    frame, hook_image, 
                    y_position=30  # Fixed pixel position from top
                )
            
            # Overlay subtitle
            if active_words:
                frame = overlay.overlay_subtitle(frame, active_words, 0)
            
            # Overlay logo on bottom-right corner
            if current_logo is not None:
                frame = self._overlay_logo(
                    frame, current_logo,
                    self.config.logo_margin_right,
                    self.config.logo_margin_bottom
                )
            
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
        
        # Combine with audio using FFMPEG (pass video dimensions for speaker overlay)
        self._encode_with_audio(input_path, temp_video_path, output_path, 
                                video_width=frame_width, video_height=frame_height)
        
        # Cleanup
        try:
            os.unlink(temp_video_path)
        except:
            pass
    
    def _encode_with_audio(self, original_video: str, processed_video: str,
                           output_path: str, video_width: int = 0, video_height: int = 0):
        """
        Combine processed video with original audio using FFMPEG.
        Optionally adds speaker info overlay if enabled in config.
        
        Args:
            original_video: Original video for audio track
            processed_video: Processed video with overlays
            output_path: Final output path
            video_width: Video width for speaker overlay
            video_height: Video height for speaker overlay
        """
        import subprocess
        
        # Try to find FFMPEG
        ffmpeg_cmd = self._find_ffmpeg()
        
        if ffmpeg_cmd is None:
            print("[WARNING] FFMPEG not found, copying without re-encoding...")
            shutil.copy(processed_video, output_path)
            return
        
        # Check if speaker overlay is enabled and we have dimensions
        ass_file = None
        if self.config.enable_speaker_overlay and video_width > 0 and video_height > 0:
            try:
                # Create speaker info
                speaker_info = SpeakerInfo(
                    name=self.config.speaker_name,
                    title=self.config.speaker_title,
                    net_worth=self.config.speaker_net_worth
                )
                
                # Create .ass file in temp folder
                ass_file = os.path.join(
                    self.config.temp_folder,
                    f"speaker_overlay_{os.getpid()}.ass"
                )
                
                create_speaker_info_ass(
                    speaker_info=speaker_info,
                    output_filename=ass_file,
                    width=video_width,
                    height=video_height,
                    start_time_seconds=self.config.speaker_start_time,
                    display_duration=self.config.speaker_display_duration,
                    font_size_ratio=self.config.speaker_font_size_ratio,
                    position_from_bottom_ratio=self.config.speaker_position_from_bottom
                )
                print(f"[INFO] Speaker overlay .ass file created")
            except Exception as e:
                print(f"[WARNING] Failed to create speaker overlay: {e}")
                ass_file = None
        
        # Build FFmpeg command
        if ass_file and os.path.exists(ass_file):
            # Escape the ass file path for FFmpeg filter (Windows compatibility)
            ass_escaped = ass_file.replace('\\', '/').replace(':', '\\:')
            
            cmd = [
                ffmpeg_cmd,
                '-y',
                '-i', processed_video,
                '-i', original_video,
                '-vf', f"ass='{ass_escaped}'",
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
            print(f"[INFO] Encoding with speaker overlay...")
        else:
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
        finally:
            # Cleanup .ass file
            if ass_file and os.path.exists(ass_file):
                try:
                    os.unlink(ass_file)
                except:
                    pass
    
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
    parser.add_argument(
        "--no-discord",
        action="store_true",
        help="Disable Discord hook selection (use AI auto-selection)"
    )
    parser.add_argument(
        "--discord-timeout",
        type=int,
        default=1200,
        help="Seconds to wait for Discord hook selection (default: 1200 = 20 min)"
    )
    parser.add_argument(
        "--upload-ig",
        action="store_true",
        help="Upload final video to Instagram Reels"
    )
    parser.add_argument(
        "--instagram-tags",
        type=str,
        nargs='+',
        default=["@airwallex", "@awxblackjz"],
        help="Tags to add to Instagram caption (default: @airwallex @awxblackjz)"
    )
    parser.add_argument(
        "--no-music",
        action="store_true",
        help="Disable background music"
    )
    parser.add_argument(
        "--music-file",
        type=str,
        default="bg_music.mp3",
        help="Path to default background music file (default: bg_music.mp3)"
    )
    parser.add_argument(
        "--music-volume",
        type=float,
        default=0.35,
        help="Background music volume 0.0-1.0 (default: 0.35)"
    )
    parser.add_argument(
        "--no-discord-music",
        action="store_true",
        help="Disable Discord music selection (use default music directly)"
    )
    parser.add_argument(
        "--no-smart-sync",
        action="store_true",
        help="Disable smart audio sync (music drop + voice peak alignment)"
    )
    parser.add_argument(
        "--voice-peak",
        type=float,
        default=None,
        help="Manual voice peak time in seconds for smart sync (auto-detect if not set)"
    )
    parser.add_argument(
        "--no-ducking",
        action="store_true",
        help="Disable audio ducking (music volume stays constant)"
    )
    parser.add_argument(
        "--ducking-low",
        type=float,
        default=0.15,
        help="Music volume during speech when ducking (default: 0.15)"
    )
    parser.add_argument(
        "--ducking-high",
        type=float,
        default=0.6,
        help="Music volume at climax/drops when ducking (default: 0.6)"
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
    config.use_discord_for_hooks = not args.no_discord
    config.discord_hook_timeout = args.discord_timeout
    config.enable_instagram_upload = args.upload_ig
    config.instagram_tags = args.instagram_tags
    config.enable_bg_music = not args.no_music
    config.default_bg_music = args.music_file
    config.bg_music_volume = args.music_volume
    config.use_discord_for_music = not args.no_discord_music
    config.enable_smart_sync = not args.no_smart_sync
    config.smart_sync_voice_peak_time = args.voice_peak
    config.enable_audio_ducking = not args.no_ducking
    config.ducking_low_volume = args.ducking_low
    config.ducking_high_volume = args.ducking_high
    
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
