import os
import requests
import logging

logger = logging.getLogger(__name__)

class VoiceboxClient:
    def __init__(self):
        self.base_url = os.environ.get("VOICEBOX_BASE_URL", "").rstrip("/")
        self.profile_id = os.environ.get("VOICEBOX_PROFILE_ID", "")
        if not self.base_url:
            logger.warning("VOICEBOX_BASE_URL not set.")
        if not self.profile_id:
            logger.warning("VOICEBOX_PROFILE_ID not set.")

    def is_available(self) -> bool:
        """Health check, returns False on any exception."""
        if not self.base_url:
            return False
        try:
            response = requests.get(f"{self.base_url}/profiles", timeout=1.5)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.debug(f"Voicebox unavailable: {e}")
            return False

    def synthesize(self, text: str, output_path: str) -> str | None:
        """Generates TTS audio and saves it to output_path. Returns output_path on success, None on failure."""
        if not self.base_url or not self.profile_id:
            logger.error("Missing Voicebox configuration.")
            return None
        
        payload = {
            "text": text,
            "profile_id": self.profile_id,
            "language": "en"
        }
        try:
            response = requests.post(
                f"{self.base_url}/generate",
                json=payload,
                timeout=60.0,
                stream=True
            )
            response.raise_for_status()
            
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return output_path
        except Exception as e:
            logger.error(f"Error synthesizing speech: {e}", exc_info=True)
            return None
