import asyncio
import aiohttp
import base64
import io
import json
import pyarrow as pa
import time

from .data import Data
from .processor import Processor, CustomMetadataField, PredefinedMetadataField


class Throttler:
    def __init__(self, qps_limit=1.0):
        self.qps_limit = qps_limit
        self._min_call_interval_seconds = 1 / qps_limit
        self._last_call = None
        self._lock = asyncio.Lock()

    async def wait(self):
        async with self._lock:
            if self._last_call is not None:
                elapsed = time.time() - self._last_call
                wait_time = max(0, self._min_call_interval_seconds - elapsed)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
            self._last_call = time.time()


class Annotator(Processor):
    def __init__(self):
        super().__init__()

    async def annotate(self, data: Data) -> None:
        pass

    def close(self) -> None:
        pass


class ImageCaptionAnnotator(Annotator):
    DEFAULT_REQUEST_HEADERS = {"Content-Type": "application/json"}

    def __init__(
        self,
        api_keys: list[str],
        output_field_name: str,
        qps_limit: float = 0.8,
        max_concurrent_call=10,
    ):
        super().__init__()
        self._api_keys = [x for x in api_keys if x is not None and len(x) > 0]
        if len(self._api_keys) == 0:
            raise ValueError("api_keys is empty")
        self._throttlers = [Throttler(qps_limit=qps_limit) for _ in self._api_keys]
        self._metadata_field = CustomMetadataField(output_field_name, pa.string())
        self._output_schema.add(self._metadata_field)
        self._max_concurrent_call = max_concurrent_call
        self._current_api_key_index = 0
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(self._max_concurrent_call)

    def construct_headers(self, api_key):
        return ImageCaptionAnnotator.DEFAULT_REQUEST_HEADERS

    def construct_request(self, api_key, image_bytes, mime_type):
        return None, None

    def parse_response(self, response_json):
        return None

    async def annotate(self, data: Data) -> None:
        try:
            data.metadata[self._metadata_field.name] = None
            if data.content is None or len(data.content) == 0:
                return
            buffer = io.BytesIO(data.content)
            mime_type = f"image/{data.encode_format}"
            async with self._lock:
                api_key = self._api_keys[self._current_api_key_index]
                throttler = self._throttlers[self._current_api_key_index]
                self._current_api_key_index += 1
                if self._current_api_key_index == len(self._api_keys):
                    self._current_api_key_index = 0

            url, request_data = self.construct_request(api_key, data.content, mime_type)
            async with self._semaphore:
                await throttler.wait()
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as session:
                    async with session.post(
                        url,
                        json=request_data,
                        headers=self.construct_headers(api_key=api_key),
                    ) as response:
                        response_json = await response.json()
                        data.metadata[self._metadata_field.name] = self.parse_response(
                            response_json
                        )

        except Exception as e:
            data.metadata[PredefinedMetadataField.ERROR_MESSAGE.name] = (
                f"Caption failed, {e}"
            )


_DEFAULT_IMAGE_CAPTION_PROMPT = "Caption the image. Remember, describe the image in detail, but do not stray beyond the information contained in the image itself. Limit your description to only what you can confidently observe."


class GeminiImageCaptionAnnotator(ImageCaptionAnnotator):
    GENERATION_CONFIG = {
        "temperature": 0.9,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 1024,
    }
    SAFETY_SETTING = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    def __init__(
        self,
        gemini_api_keys: list[str],
        prompt=_DEFAULT_IMAGE_CAPTION_PROMPT,
        gemini_model_name="gemini-pro-vision",
        qps_limit: float = 0.8,
        max_concurrent_call=10,
    ):
        super().__init__(
            api_keys=gemini_api_keys,
            output_field_name="gemini_image_caption",
            qps_limit=qps_limit,
            max_concurrent_call=max_concurrent_call,
        )
        self._gemini_model_name = gemini_model_name
        self._prompt = prompt

    def construct_request(self, api_key, image_bytes, mime_type):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self._gemini_model_name}:generateContent?key={api_key}"
        encoded_img_bytes = base64.b64encode(image_bytes).decode("utf-8")
        request_dict = {
            "contents": [
                {
                    "parts": [
                        {"text": self._prompt},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": encoded_img_bytes,
                            }
                        },
                    ]
                }
            ],
            "safety_settings": GeminiImageCaptionAnnotator.SAFETY_SETTING,
        }
        return url, request_dict

    def parse_response(self, response_json):
        try:
            return response_json["candidates"][0]["content"]["parts"][0]["text"].strip()
        except:
            raise ValueError(f"Gemini caption failed: response{response_json}")


class AnthropicImageCaptionAnnotator(ImageCaptionAnnotator):
    PROMPT = "Caption the image. Remember, describe the image in detail, but do not stray beyond the information contained in the image itself. Limit your description to only what you can confidently observe."
    GENERATION_CONFIG = {
        "temperature": 0.9,
        "top_p": 1,
        "top_k": 1,
        "max_tokens": 512,
    }

    def __init__(
        self,
        anthropic_api_keys: list[str],
        prompt=_DEFAULT_IMAGE_CAPTION_PROMPT,
        anthropic_model_name="claude-3-haiku-20240307",
        qps_limit: float = 0.8,
        max_concurrent_call=10,
    ):
        super().__init__(
            api_keys=anthropic_api_keys,
            output_field_name="anthropic_image_caption",
            qps_limit=qps_limit,
            max_concurrent_call=max_concurrent_call,
        )
        self._anthropic_model_name = anthropic_model_name
        self._prompt = prompt

    def construct_headers(self, api_key):
        return ImageCaptionAnnotator.DEFAULT_REQUEST_HEADERS | {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }

    def construct_request(self, api_key, image_bytes, mime_type):
        encoded_img_bytes = base64.b64encode(image_bytes).decode("utf-8")
        request_dict = {
            "model": self._anthropic_model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": encoded_img_bytes,
                            },
                        },
                        {"type": "text", "text": self._prompt},
                    ],
                },
            ],
        } | AnthropicImageCaptionAnnotator.GENERATION_CONFIG
        return "https://api.anthropic.com/v1/messages", request_dict

    def parse_response(self, response_json):
        try:
            return " ".join(
                [x["text"] for x in response_json["content"] if x["type"] == "text"]
            )
        except:
            raise ValueError(f"Anthropic caption failed: response{response_json}")
