"""
YouTube Uploader Module
=======================

Uploads short-form videos to YouTube with AI-generated titles and descriptions.

Features:
- Uses LangChain + Gemini to generate engaging titles and descriptions
- OAuth2 authentication with YouTube Data API
- Uploads as YouTube Shorts (vertical video)
- Saves upload metadata for tracking

Requirements:
    pip install google-auth google-auth-oauthlib google-api-python-client langchain langchain-google-genai

Setup:
    1. Create OAuth2 credentials in Google Cloud Console
    2. Download client secret JSON file
    3. Add YOUTUBE_CLIENT_SECRET_FILE path to .env (or use default)
    4. First run will open browser for authentication
"""

import os
import json
import pickle
import re
from pathlib import Path
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class YouTubeConfig:
    """Configuration for YouTube upload."""
    
    # OAuth settings
    client_secret_file: str = "virtualrealm_ytdata_api_client_secret.json"
    token_file: str = "youtube_token.pickle"
    
    # API settings
    api_service_name: str = "youtube"
    api_version: str = "v3"
    scopes: tuple = ("https://www.googleapis.com/auth/youtube.upload",)
    
    # Upload defaults
    category_id: str = "22"  # People & Blogs (good for shorts)
    privacy_status: str = "public"  # public, private, or unlisted
    made_for_kids: bool = False
    
    # Shorts settings
    shorts_hashtags: list = None
    
    def __post_init__(self):
        if self.shorts_hashtags is None:
            self.shorts_hashtags = ["#Shorts", "#YouTubeShorts", "#Viral"]


# =============================================================================
# TITLE & DESCRIPTION GENERATOR (GEMINI AI)
# =============================================================================

class VideoMetadataGenerator:
    """
    Uses Google Gemini via LangChain to generate YouTube titles and descriptions.
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain.schema import HumanMessage, SystemMessage
            
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-pro",
                google_api_key=api_key,
                temperature=0.7  # More creative for titles
            )
            self.HumanMessage = HumanMessage
            self.SystemMessage = SystemMessage
            
        except ImportError:
            raise ImportError("Please install: pip install langchain langchain-google-genai")
    
    def generate_metadata(self, transcript: str, hook: Optional[str] = None,
                          additional_context: Optional[str] = None) -> Dict[str, str]:
        """
        Generate YouTube title and description using Gemini AI.
        
        Args:
            transcript: Full transcript of the video
            hook: Optional hook text that was used in the video
            additional_context: Any additional context about the video
            
        Returns:
            Dictionary with 'title', 'description', and 'tags'
        """
        print("[INFO] Generating YouTube metadata with Gemini AI...")
        
        system_prompt = """You are an expert YouTube Shorts content strategist specializing in viral short-form content.
Your task is to create compelling titles and descriptions that maximize views and engagement.

For TITLES:
- Keep under 70 characters (ideally 50-60)
- Use emotional triggers and curiosity gaps
- Include power words that drive clicks
- Make it feel urgent or exclusive
- Don't use clickbait that doesn't deliver
- Consider using numbers or lists when relevant
- Capitalize key words strategically

For DESCRIPTIONS:
- Start with a strong hook in the first line (this shows in search)
- Keep total description under 200 words
- Include relevant keywords naturally
- Add a call-to-action (like, subscribe, comment)
- Be conversational and engaging
- Don't stuff with hashtags (we add those separately)

For TAGS:
- Provide 8-12 relevant tags
- Mix broad and specific tags
- Include trending topics if relevant
- Think about search terms viewers would use

Return your response in this exact JSON format:
{
    "title": "Your Title Here",
    "description": "Your description here...",
    "tags": ["tag1", "tag2", "tag3"]
}"""

        context_parts = [f"TRANSCRIPT:\n{transcript[:3000]}"]  # Limit transcript length
        
        if hook:
            context_parts.append(f"\nHOOK USED IN VIDEO:\n{hook}")
        
        if additional_context:
            context_parts.append(f"\nADDITIONAL CONTEXT:\n{additional_context}")
        
        user_prompt = f"""Based on this video content, generate an optimized YouTube Shorts title, description, and tags.

{chr(10).join(context_parts)}

Remember:
- This is a YouTube SHORT (vertical, under 60 seconds)
- The title should create curiosity and drive clicks
- The description should be engaging but concise
- Tags should help with discoverability

Return ONLY the JSON object, no other text."""

        try:
            messages = [
                self.SystemMessage(content=system_prompt),
                self.HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            response_text = response.content.strip()
            
            # Extract JSON from response
            if "```" in response_text:
                match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
                if match:
                    response_text = match.group(1)
            
            # Find JSON object in response
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group()
            
            metadata = json.loads(response_text)
            
            # Validate required fields
            if 'title' not in metadata:
                metadata['title'] = "Check This Out! 🔥"
            if 'description' not in metadata:
                metadata['description'] = "Watch till the end! 👀\n\nLike and subscribe for more!"
            if 'tags' not in metadata:
                metadata['tags'] = ["shorts", "viral", "trending"]
            
            # Ensure title is not too long
            if len(metadata['title']) > 100:
                metadata['title'] = metadata['title'][:97] + "..."
            
            print(f"[INFO] Generated title: {metadata['title']}")
            print(f"[INFO] Generated {len(metadata['tags'])} tags")
            
            return metadata
            
        except Exception as e:
            print(f"[WARNING] Metadata generation failed: {e}")
            # Return defaults
            return {
                'title': "You Need To See This! 🔥 #Shorts",
                'description': "Watch this incredible moment! 👀\n\nDon't forget to like and subscribe for more amazing content!",
                'tags': ["shorts", "viral", "trending", "fyp", "mustwatch"]
            }


# =============================================================================
# YOUTUBE UPLOADER
# =============================================================================

class YouTubeUploader:
    """
    Handles YouTube video uploads using the YouTube Data API v3.
    """
    
    def __init__(self, config: Optional[YouTubeConfig] = None):
        self.config = config or YouTubeConfig()
        self.youtube = None
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with YouTube API using OAuth2."""
        try:
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build
        except ImportError:
            raise ImportError(
                "Please install: pip install google-auth google-auth-oauthlib google-api-python-client"
            )
        
        credentials = None
        
        # Check for existing token
        if os.path.exists(self.config.token_file):
            print("[INFO] Loading existing YouTube credentials...")
            with open(self.config.token_file, 'rb') as token:
                credentials = pickle.load(token)
        
        # Refresh or get new credentials
        if not credentials or not credentials.valid:
            if credentials and credentials.expired and credentials.refresh_token:
                print("[INFO] Refreshing YouTube credentials...")
                credentials.refresh(Request())
            else:
                if not os.path.exists(self.config.client_secret_file):
                    raise FileNotFoundError(
                        f"Client secret file not found: {self.config.client_secret_file}\n"
                        "Please download it from Google Cloud Console."
                    )
                
                print("[INFO] Starting OAuth2 flow...")
                print("[INFO] A browser window will open for authentication.")
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.config.client_secret_file,
                    self.config.scopes
                )
                credentials = flow.run_local_server(port=8080)
            
            # Save credentials for future use
            with open(self.config.token_file, 'wb') as token:
                pickle.dump(credentials, token)
            print("[INFO] Credentials saved for future use.")
        
        # Build YouTube API client
        self.youtube = build(
            self.config.api_service_name,
            self.config.api_version,
            credentials=credentials
        )
        print("[INFO] YouTube API client initialized successfully!")
    
    def upload_video(self, video_path: str, title: str, description: str,
                     tags: list = None, category_id: str = None,
                     privacy_status: str = None,
                     notify_subscribers: bool = True) -> Dict:
        """
        Upload a video to YouTube.
        
        Args:
            video_path: Path to the video file
            title: Video title
            description: Video description
            tags: List of tags
            category_id: YouTube category ID
            privacy_status: 'public', 'private', or 'unlisted'
            notify_subscribers: Whether to notify channel subscribers
            
        Returns:
            Dictionary with upload response data
        """
        from googleapiclient.http import MediaFileUpload
        from googleapiclient.errors import HttpError
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Add shorts hashtags to description
        hashtags = " ".join(self.config.shorts_hashtags)
        full_description = f"{description}\n\n{hashtags}"
        
        # Prepare request body
        body = {
            'snippet': {
                'title': title,
                'description': full_description,
                'tags': tags or [],
                'categoryId': category_id or self.config.category_id
            },
            'status': {
                'privacyStatus': privacy_status or self.config.privacy_status,
                'selfDeclaredMadeForKids': self.config.made_for_kids,
            },
            'notifySubscribers': notify_subscribers
        }
        
        # Create media upload object
        media = MediaFileUpload(
            video_path,
            mimetype='video/mp4',
            resumable=True,
            chunksize=1024*1024  # 1MB chunks
        )
        
        print(f"[INFO] Uploading video: {title}")
        print(f"[INFO] File: {video_path}")
        print(f"[INFO] Privacy: {body['status']['privacyStatus']}")
        
        try:
            # Execute upload
            request = self.youtube.videos().insert(
                part=','.join(body.keys()),
                body=body,
                media_body=media
            )
            
            response = None
            while response is None:
                status, response = request.next_chunk()
                if status:
                    progress = int(status.progress() * 100)
                    print(f"[INFO] Upload progress: {progress}%")
            
            video_id = response['id']
            video_url = f"https://youtube.com/shorts/{video_id}"
            
            print(f"[SUCCESS] Video uploaded successfully!")
            print(f"[INFO] Video ID: {video_id}")
            print(f"[INFO] URL: {video_url}")
            
            return {
                'success': True,
                'video_id': video_id,
                'url': video_url,
                'title': title,
                'response': response
            }
            
        except HttpError as e:
            error_content = json.loads(e.content.decode('utf-8'))
            error_reason = error_content.get('error', {}).get('errors', [{}])[0].get('reason', 'unknown')
            
            print(f"[ERROR] Upload failed: {error_reason}")
            print(f"[ERROR] Details: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'reason': error_reason
            }
    
    def upload_short(self, video_path: str, transcript: str,
                     hook: Optional[str] = None,
                     gemini_api_key: Optional[str] = None,
                     custom_title: Optional[str] = None,
                     custom_description: Optional[str] = None,
                     privacy_status: str = "public") -> Dict:
        """
        Upload a YouTube Short with AI-generated metadata.
        
        Args:
            video_path: Path to the short video file
            transcript: Video transcript for metadata generation
            hook: Optional hook text used in the video
            gemini_api_key: API key for Gemini (uses env var if not provided)
            custom_title: Override AI-generated title
            custom_description: Override AI-generated description
            privacy_status: 'public', 'private', or 'unlisted'
            
        Returns:
            Dictionary with upload result and metadata
        """
        # Get API key
        api_key = gemini_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        
        # Generate metadata with AI if not provided
        if custom_title and custom_description:
            metadata = {
                'title': custom_title,
                'description': custom_description,
                'tags': ['shorts', 'viral']
            }
        elif api_key:
            generator = VideoMetadataGenerator(api_key)
            metadata = generator.generate_metadata(transcript, hook)
        else:
            print("[WARNING] No Gemini API key, using default metadata")
            metadata = {
                'title': "You Won't Believe This! 🔥 #Shorts",
                'description': "Watch till the end! 👀\n\nLike and subscribe!",
                'tags': ['shorts', 'viral', 'trending']
            }
        
        # Use custom overrides if provided
        if custom_title:
            metadata['title'] = custom_title
        if custom_description:
            metadata['description'] = custom_description
        
        # Upload the video
        result = self.upload_video(
            video_path=video_path,
            title=metadata['title'],
            description=metadata['description'],
            tags=metadata['tags'],
            privacy_status=privacy_status
        )
        
        # Add metadata to result
        result['metadata'] = metadata
        
        # Save upload record
        self._save_upload_record(video_path, result)
        
        return result
    
    def _save_upload_record(self, video_path: str, result: Dict):
        """Save upload record to JSON file for tracking."""
        records_file = "youtube_uploads.json"
        
        # Load existing records
        records = []
        if os.path.exists(records_file):
            try:
                with open(records_file, 'r', encoding='utf-8') as f:
                    records = json.load(f)
            except:
                records = []
        
        # Add new record
        record = {
            'timestamp': datetime.now().isoformat(),
            'video_file': video_path,
            'success': result.get('success', False),
            'video_id': result.get('video_id'),
            'url': result.get('url'),
            'title': result.get('title') or result.get('metadata', {}).get('title'),
            'error': result.get('error')
        }
        records.append(record)
        
        # Save records
        with open(records_file, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        
        print(f"[INFO] Upload record saved to: {records_file}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def upload_short_to_youtube(video_path: str, transcript: str,
                            hook: Optional[str] = None,
                            privacy_status: str = "public",
                            client_secret_file: str = "virtualrealm_ytdata_api_client_secret.json") -> Dict:
    """
    Convenience function to upload a short to YouTube.
    
    Args:
        video_path: Path to the video file
        transcript: Video transcript for AI metadata generation
        hook: Optional hook text
        privacy_status: 'public', 'private', or 'unlisted'
        client_secret_file: Path to OAuth client secret file
        
    Returns:
        Upload result dictionary
    """
    config = YouTubeConfig(client_secret_file=client_secret_file)
    uploader = YouTubeUploader(config)
    
    return uploader.upload_short(
        video_path=video_path,
        transcript=transcript,
        hook=hook,
        privacy_status=privacy_status
    )


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Command-line interface for YouTube upload."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Upload videos to YouTube with AI-generated metadata"
    )
    parser.add_argument(
        "video",
        type=str,
        help="Path to video file to upload"
    )
    parser.add_argument(
        "--transcript", "-t",
        type=str,
        help="Path to transcript file (JSON with 'text' field)"
    )
    parser.add_argument(
        "--title",
        type=str,
        help="Custom title (overrides AI generation)"
    )
    parser.add_argument(
        "--description", "-d",
        type=str,
        help="Custom description (overrides AI generation)"
    )
    parser.add_argument(
        "--privacy", "-p",
        type=str,
        choices=['public', 'private', 'unlisted'],
        default='public',
        help="Privacy status (default: public)"
    )
    parser.add_argument(
        "--client-secret",
        type=str,
        default="virtualrealm_ytdata_api_client_secret.json",
        help="Path to OAuth client secret file"
    )
    
    args = parser.parse_args()
    
    # Load transcript if provided
    transcript = ""
    if args.transcript:
        if os.path.exists(args.transcript):
            with open(args.transcript, 'r', encoding='utf-8') as f:
                data = json.load(f)
                transcript = data.get('text', '')
        else:
            print(f"[WARNING] Transcript file not found: {args.transcript}")
    
    # If no transcript, try to find one based on video name
    if not transcript:
        video_name = Path(args.video).stem.replace('_final', '')
        possible_transcript = f"temp/{video_name}_transcription.json"
        if os.path.exists(possible_transcript):
            print(f"[INFO] Found transcript: {possible_transcript}")
            with open(possible_transcript, 'r', encoding='utf-8') as f:
                data = json.load(f)
                transcript = data.get('text', '')
    
    if not transcript:
        transcript = "An amazing short video clip."
        print("[INFO] No transcript found, using default metadata")
    
    # Configure and upload
    config = YouTubeConfig(client_secret_file=args.client_secret)
    uploader = YouTubeUploader(config)
    
    result = uploader.upload_short(
        video_path=args.video,
        transcript=transcript,
        custom_title=args.title,
        custom_description=args.description,
        privacy_status=args.privacy
    )
    
    if result['success']:
        print(f"\n✅ Upload successful!")
        print(f"🔗 Watch at: {result['url']}")
    else:
        print(f"\n❌ Upload failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
