# src/doubao_transcriber.py
import os
import uuid
import base64
import requests
from typing import Optional, Union, BinaryIO
import logging
import io

logger = logging.getLogger(__name__)


class DoubaoAudioTranscriber:
    """
    Doubao audio transcriber.
    Uses the Volcano Engine big-model flash audio-file recognition API.
    """

    def __init__(
            self,
            appid: Optional[str] = None,
            access_token: Optional[str] = None,
            api_url: str = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/recognize/flash",
            timeout: int = 30
    ):
        """
        Initialize the Doubao audio transcriber.

        Args:
            appid: Volcano Engine App ID
            access_token: Volcano Engine Access Token
            api_url: API endpoint
            timeout: request timeout in seconds
        """
        self.appid = appid or os.getenv("DOUBAO_APP_ID")
        self.access_token = access_token or os.getenv("DOUBAO_ACCESS_TOKEN")
        self.api_url = api_url
        self.timeout = timeout
        self.resource_id = "volc.bigasr.auc_turbo"

        self._validate_config()

    def _validate_config(self):
        """Validate that the configuration is complete."""
        if not self.appid:
            raise ValueError("Doubao API App ID is required; set it via the DOUBAO_APP_ID environment variable")
        if not self.access_token:
            raise ValueError("Doubao API Access Token is required; set it via the DOUBAO_ACCESS_TOKEN environment variable")

        logger.info(f"Doubao audio transcriber initialized, App ID: {self.appid[:8]}...")

    def _get_headers(self) -> dict:
        """Build request headers."""
        return {
            "X-Api-App-Key": self.appid,
            "X-Api-Access-Key": self.access_token,
            "X-Api-Resource-Id": self.resource_id,
            "X-Api-Request-Id": str(uuid.uuid4()),
            "X-Api-Sequence": "-1",
            "Content-Type": "application/json"
        }

    def _get_request_body(self, base64_data: str) -> dict:
        """Build request body."""
        return {
            "user": {
                "uid": self.appid
            },
            "audio": {
                "data": base64_data
            },
            "request": {
                "model_name": "bigmodel",
                "enable_punc": True,  # enable punctuation
                "enable_itn": True,  # enable inverse text normalization
                "enable_ddc": True,  # enable smart segmentation
            }
        }

    def _format_to_base64(self, audio_input: Union[BinaryIO, bytes, str]) -> str:
        """
        Convert audio input to a base64 string.

        Args:
            audio_input: Audio input: a file-like object, bytes, or a file path

        Returns:
            Base64-encoded string
        """
        try:
            if hasattr(audio_input, 'read'):
                # Treat any object with a read() method as a file stream
                audio_input.seek(0)
                audio_data = audio_input.read()
            elif isinstance(audio_input, bytes):
                audio_data = audio_input
            elif isinstance(audio_input, str) and os.path.exists(audio_input):
                with open(audio_input, 'rb') as f:
                    audio_data = f.read()
            else:
                raise ValueError(f"Unsupported audio input type: {type(audio_input)}")

            return base64.b64encode(audio_data).decode('utf-8')

        except Exception as e:
            logger.error(f"Error converting audio to base64: {str(e)}")
            raise

    def transcribe(
            self,
            audio_input: Union[BinaryIO, bytes, str],
            audio_format: Optional[str] = None
    ) -> str:
        """
        Transcribe an audio file.

        Args:
            audio_input: Audio input: a file-like object, bytes, or a file path
            audio_format: audio format hint (optional)

        Returns:
            Transcribed text
        """
        try:
            logger.debug(f"Audio input type: {type(audio_input)}, format: {audio_format}")

            base64_data = self._format_to_base64(audio_input)
            logger.debug(f"Base64 data length: {len(base64_data)} chars")

            headers = self._get_headers()
            request_body = self._get_request_body(base64_data)

            logger.debug("Sending request to Doubao API...")
            response = requests.post(
                self.api_url,
                json=request_body,
                headers=headers,
                timeout=self.timeout
            )

            logger.debug(f"API response status code: {response.status_code}")

            return self._parse_response(response)

        except requests.exceptions.Timeout:
            error_msg = "Request timed out"
            logger.error(f"{error_msg}, timeout: {self.timeout}s")
            return f"Transcription failed: {error_msg}"
        except requests.exceptions.RequestException as e:
            error_msg = f"Network request failed: {str(e)}"
            logger.error(error_msg)
            return f"Transcription failed: {error_msg}"
        except Exception as e:
            error_msg = f"Transcription error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return f"Transcription failed: {error_msg}"

    def _parse_response(self, response: requests.Response) -> str:
        """Parse the API response."""
        status_code = response.headers.get('X-Api-Status-Code')

        if status_code == '20000000':
            # Success
            result = response.json()
            text = result.get('result', {}).get('text', '')

            duration = result.get('audio_info', {}).get('duration', 0)
            log_id = response.headers.get('X-Tt-Logid', 'N/A')
            logger.info(f"Transcription succeeded - duration: {duration}ms, LogID: {log_id}, chars: {len(text)}")

            return text
        elif status_code == '20000003':
            # Silent audio
            logger.warning("Silent audio, nothing to transcribe")
            return "Silent audio, no transcription content"
        elif status_code == '45000001':
            # Invalid request parameters
            logger.error("Invalid request parameters")
            return "Transcription failed: invalid request parameters"
        elif status_code == '45000002':
            # Empty audio
            logger.error("Empty audio file")
            return "Transcription failed: empty audio file"
        elif status_code == '45000151':
            # Invalid audio format
            logger.error("Invalid audio format")
            return "Transcription failed: invalid audio format"
        else:
            # Other errors
            error_msg = response.headers.get('X-Api-Message', 'Unknown error')
            log_id = response.headers.get('X-Tt-Logid', 'N/A')
            logger.error(f"Transcription failed - status code: {status_code}, message: {error_msg}, LogID: {log_id}")

            # Try to extract error details from the response body
            try:
                error_detail = response.json()
                logger.error(f"Error details: {error_detail}")
            except:
                pass

            return f"Transcription failed: {error_msg}"

    def get_supported_formats(self) -> dict:
        """Return the supported audio formats."""
        return {
            'extensions': ['.wav', '.mp3', '.ogg', '.opus'],
            'mime_types': [
                'audio/x-wav',
                'audio/wav',
                'audio/mpeg',
                'audio/mp3',
                'audio/ogg',
                'audio/opus'
            ]
        }