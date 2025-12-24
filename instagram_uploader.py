"""
Instagram Uploader
==================

Uploads videos to Instagram Reels using the Instagram Graph API.
Uses ngrok to create a public URL for the video file.

Requirements:
    pip install pyngrok requests python-dotenv

Environment Variables:
    IG_BUSINESS_ACCOUNT_ID: Instagram Business Account ID
    INSTAGRAM_ACCESS_TOKEN: Long-lived Instagram access token
    NGROK_AUTH_TOKEN: (Optional) Ngrok auth token for longer sessions
"""

import os
import csv
import time
import threading
import requests
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Optional, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class InstagramConfig:
    """Configuration for Instagram upload."""
    business_account_id: str = ""
    access_token: str = ""
    port: int = 3000
    thumb_offset: int = 500  # Thumbnail offset in ms
    audio_name: str = "Original Audio"
    share_to_feed: bool = True
    csv_log_path: str = "instagram_posts.csv"
    # Tags to include in caption
    tags: list = None
    
    def __post_init__(self):
        if not self.business_account_id:
            self.business_account_id = os.getenv("IG_BUSINESS_ACCOUNT_ID", "")
        if not self.access_token:
            self.access_token = os.getenv("INSTAGRAM_ACCESS_TOKEN", "")
        if self.tags is None:
            self.tags = ["@airwallex", "@awxblackjz"]


class InstagramUploader:
    """
    Uploads videos to Instagram Reels using the Graph API.
    """
    
    def __init__(self, config: Optional[InstagramConfig] = None):
        self.config = config or InstagramConfig()
        self.server = None
        self.server_thread = None
        self.public_url = None
        
        # Validate credentials
        if not self.config.business_account_id:
            raise ValueError("IG_BUSINESS_ACCOUNT_ID not found in environment")
        if not self.config.access_token:
            raise ValueError("INSTAGRAM_ACCESS_TOKEN not found in environment")
        
        # Try to import ngrok
        try:
            import pyngrok
            from pyngrok import ngrok
            self.ngrok = ngrok
            
            # Set auth token if available
            auth_token = os.getenv("NGROK_AUTH_TOKEN")
            if auth_token:
                ngrok.set_auth_token(auth_token)
                
        except ImportError:
            raise ImportError("Please install pyngrok: pip install pyngrok")
    
    def _start_server(self, video_dir: str) -> str:
        """
        Start HTTP server and ngrok tunnel to serve the video.
        
        Args:
            video_dir: Directory containing the video file
            
        Returns:
            Public URL to access the video
        """
        import functools
        
        # Create a custom handler that serves from the video directory
        # This avoids issues with changing the working directory
        handler = functools.partial(SimpleHTTPRequestHandler, directory=video_dir)
        
        # Start HTTP server with the custom handler
        self.server = HTTPServer(("", self.config.port), handler)
        
        # Establish ngrok tunnel
        self.public_url = self.ngrok.connect(self.config.port)
        print(f"[INSTAGRAM] ngrok tunnel established at: {self.public_url}")
        print(f"[INSTAGRAM] Serving files from: {video_dir}")
        
        # Start server in background thread
        def serve():
            print(f"[INSTAGRAM] Serving HTTP on port {self.config.port}...")
            self.server.serve_forever()
        
        self.server_thread = threading.Thread(target=serve)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        return self.public_url.public_url
    
    def _stop_server(self):
        """Stop the HTTP server and ngrok tunnel."""
        if self.server:
            print("[INSTAGRAM] Shutting down server...")
            self.server.shutdown()
            self.server.server_close()
        
        if self.public_url:
            self.ngrok.disconnect(self.public_url)
            self.ngrok.kill()
        
        print("[INSTAGRAM] Server stopped")
    
    def _create_media_container(self, video_url: str, caption: str) -> str:
        """
        Create a media container for the Reel.
        
        Args:
            video_url: Public URL to the video
            caption: Caption for the Reel
            
        Returns:
            Creation ID for the media container
        """
        params = {
            "media_type": "REELS",
            "video_url": video_url,
            "caption": caption,
            "share_to_feed": "true" if self.config.share_to_feed else "false",
            "thumb_offset": str(self.config.thumb_offset),
            "access_token": self.config.access_token
        }
        
        if self.config.audio_name:
            params["audio_name"] = self.config.audio_name
        
        url = f"https://graph.facebook.com/v22.0/{self.config.business_account_id}/media"
        response = requests.post(url, params=params)
        data = response.json()
        
        if "error" in data:
            raise Exception(f"Failed to create media container: {data['error']}")
        
        creation_id = data["id"]
        print(f"[INSTAGRAM] Media container created: {creation_id}")
        return creation_id
    
    def _wait_for_processing(self, creation_id: str, max_wait: int = 600) -> bool:
        """
        Wait for Instagram to process the video.
        
        Args:
            creation_id: The media container ID
            max_wait: Maximum seconds to wait (default 10 minutes)
            
        Returns:
            True if processing finished, False if timed out
        """
        print("[INSTAGRAM] Waiting for video processing...")
        
        params = {
            "fields": "status_code",
            "access_token": self.config.access_token
        }
        
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            url = f"https://graph.facebook.com/v22.0/{creation_id}"
            response = requests.get(url, params=params)
            data = response.json()
            
            if "error" in data:
                print(f"[INSTAGRAM] Error checking status: {data['error']}")
                return False
            
            status = data.get("status_code", "UNKNOWN")
            print(f"[INSTAGRAM] Processing status: {status}")
            
            if status == "FINISHED":
                return True
            elif status == "ERROR":
                print("[INSTAGRAM] Processing failed!")
                return False
            
            time.sleep(5)
        
        print("[INSTAGRAM] Processing timed out!")
        return False
    
    def _publish_media(self, creation_id: str) -> Dict[str, str]:
        """
        Publish the processed media.
        
        Args:
            creation_id: The media container ID
            
        Returns:
            Dict with media_id and permalink
        """
        params = {
            "creation_id": creation_id,
            "access_token": self.config.access_token
        }
        
        url = f"https://graph.facebook.com/v22.0/{self.config.business_account_id}/media_publish"
        response = requests.post(url, params=params)
        data = response.json()
        
        if "error" in data:
            raise Exception(f"Failed to publish: {data['error']}")
        
        media_id = data["id"]
        print(f"[INSTAGRAM] Published! Media ID: {media_id}")
        
        # Get permalink
        params = {"access_token": self.config.access_token}
        url = f"https://graph.facebook.com/v22.0/{media_id}?fields=permalink"
        response = requests.get(url, params=params)
        data = response.json()
        
        permalink = data.get("permalink", "")
        print(f"[INSTAGRAM] Permalink: {permalink}")
        
        return {"media_id": media_id, "permalink": permalink}
    
    def _log_to_csv(self, permalink: str, media_id: str, title: str):
        """Log the upload to CSV file."""
        csv_path = self.config.csv_log_path
        
        # Create header if file doesn't exist
        if not os.path.exists(csv_path):
            with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["permalink", "media_id", "title", "likes", "comments", "shares", "saves", "reach", "engagement_rate"])
        
        # Append new entry
        with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([permalink, media_id, title, 0, 0, 0, 0, 0, "0.00%"])
        
        print(f"[INSTAGRAM] Logged to {csv_path}")
    
    def _format_caption(self, caption: str) -> str:
        """
        Format caption with required tags.
        
        Args:
            caption: Original caption
            
        Returns:
            Caption with tags added
        """
        # Add tags if not already present
        tags_str = " ".join(self.config.tags)
        
        if tags_str not in caption:
            caption = f"{caption}\n\n{tags_str}"
        
        return caption
    
    def upload(self, video_path: str, caption: str, title: str = "") -> Optional[Dict[str, str]]:
        """
        Upload a video to Instagram Reels.
        
        Args:
            video_path: Path to the video file
            caption: Caption for the Reel
            title: Title for logging purposes
            
        Returns:
            Dict with media_id and permalink, or None if failed
        """
        if not os.path.exists(video_path):
            print(f"[INSTAGRAM] Video not found: {video_path}")
            return None
        
        video_dir = os.path.dirname(os.path.abspath(video_path))
        video_filename = os.path.basename(video_path)
        
        # Format caption with tags
        formatted_caption = self._format_caption(caption)
        
        try:
            # Start server and get public URL
            base_url = self._start_server(video_dir)
            video_url = f"{base_url}/{video_filename}"
            print(f"[INSTAGRAM] Video URL: {video_url}")
            
            # Wait a bit for ngrok to stabilize
            time.sleep(2)
            
            # Create media container
            creation_id = self._create_media_container(video_url, formatted_caption)
            
            # Wait for processing
            if not self._wait_for_processing(creation_id):
                print("[INSTAGRAM] Upload failed - processing error")
                return None
            
            # Publish
            result = self._publish_media(creation_id)
            
            # Log to CSV
            self._log_to_csv(result["permalink"], result["media_id"], title or caption[:50])
            
            return result
            
        except Exception as e:
            print(f"[INSTAGRAM] Upload failed: {e}")
            import traceback
            traceback.print_exc()
            return None
            
        finally:
            # Always stop server
            self._stop_server()


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    print("Instagram Uploader Test")
    print("=" * 60)
    
    # Check environment variables
    required_vars = ["IG_BUSINESS_ACCOUNT_ID", "INSTAGRAM_ACCESS_TOKEN"]
    missing = [v for v in required_vars if not os.getenv(v)]
    
    if missing:
        print(f"Missing environment variables: {missing}")
        print("Please set them in your .env file")
    else:
        print("Environment variables found!")
        print(f"Business Account ID: {os.getenv('IG_BUSINESS_ACCOUNT_ID')[:10]}...")
        print(f"Access Token: {os.getenv('INSTAGRAM_ACCESS_TOKEN')[:20]}...")
        
        # Test upload (uncomment to test)
        # config = InstagramConfig()
        # uploader = InstagramUploader(config)
        # result = uploader.upload(
        #     video_path="output/test_video.mp4",
        #     caption="Test upload from automation pipeline! 🚀",
        #     title="Test Video"
        # )
        # print(f"Result: {result}")
