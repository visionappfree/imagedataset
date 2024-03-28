import albumentations as A
import cv2
import math
import numpy as np
import pandas as pd
import pyarrow as pa

from collections import namedtuple
from enum import Enum

from .data import Task, CompletedTask, Data
from .processor import Processor, PredefinedMetadataField, CustomMetadataField


def _get_attr_or_none(data: tuple, attr: str):
    try:
        return getattr(data, attr)
    except:
        return None


def pandas_dtype_to_arrow(dtype):
    if pd.api.types.is_integer_dtype(dtype):
        return pa.int64()
    elif pd.api.types.is_float_dtype(dtype):
        return pa.float64()
    elif pd.api.types.is_bool_dtype(dtype):
        return pa.bool_()
    elif pd.api.types.is_string_dtype(dtype):
        return pa.string()
    # Add more mappings as needed
    else:
        raise ValueError(f"Unsupported pandas dtype: {dtype}")


class DataFrameToTaskConverter(Processor):
    def __init__(
        self,
        data: pd.DataFrame,
        url_column: str = "url",
        columns_to_convert: set[str] | None = None,
    ) -> None:
        super().__init__()
        self._data = data
        self._num_tasks = data.shape[0]
        self._num_id_digits = math.ceil(math.log10(self._num_tasks))
        self._url_column = url_column
        self._columns_to_convert = set(
            columns_to_convert
            if columns_to_convert is not None
            else [c for c in data.columns.to_list()]
        )

        self._output_schema.update(
            [
                CustomMetadataField(name, pandas_dtype_to_arrow(data[name].dtype))
                for name in self._columns_to_convert
            ]
        )

    def _convert_task(self, input_data: namedtuple) -> Task | None:
        url = _get_attr_or_none(input_data, self._url_column)

        id = f"{input_data.Index:0{self._num_id_digits}}"
        return Task(
            url,
            id,
            {
                field: getattr(input_data, field)
                for field in input_data._fields
                if field in self._columns_to_convert
            },
        )

    def get_tasks(self) -> list[Task]:
        return [self._convert_task(d) for d in self._data.itertuples(index=True)]


_INTER_STR_TO_CV2 = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "bilinear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "bicubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
    "lanczos4": cv2.INTER_LANCZOS4,
}


class ResizeMode(Enum):
    NO_RESIZE = 0  # pylint: disable=invalid-name
    KEEP_RATIO = 1  # pylint: disable=invalid-name
    CENTER_CROP = 2  # pylint: disable=invalid-name
    PAD_BORDER = 3  # pylint: disable=invalid-name
    KEEP_RATIO_LARGEST = 4  # pylint: disable=invalid-name


def inter_str_to_cv2(inter_str):
    inter_str = inter_str.lower()
    if inter_str not in _INTER_STR_TO_CV2:
        raise ValueError(f"Invalid option for interpolation: {inter_str}")
    return _INTER_STR_TO_CV2[inter_str]


class CompletedTaskConverter(Processor):
    def __init__(self) -> None:
        super().__init__()
        self._output_schema.update(
            [
                PredefinedMetadataField.KEY,
                PredefinedMetadataField.ERROR_MESSAGE,
            ]
        )

    def __call__(self, completed_task: CompletedTask) -> Data | None:
        converted_data = Data(
            completed_task.task.id,
            None,
            self.encode_format,
            completed_task.task.metadata,
        )
        converted_data.metadata[PredefinedMetadataField.KEY.name] = completed_task.key
        converted_data.metadata[PredefinedMetadataField.ERROR_MESSAGE.name] = (
            Processor.SUCCESS_MESSAGE
            if completed_task.download_error is None
            else completed_task.download_error
        )
        return converted_data


class ImageDataConverter(CompletedTaskConverter):
    """Convert for image data from CompletedTask to Data."""

    def __init__(
        self,
        resize_mode=ResizeMode.KEEP_RATIO_LARGEST,
        image_size=512,
        encode_format="jpeg",
        encode_quality=95,
        resize_only_if_bigger=True,
        upscale_interpolation="lanczos",
        downscale_interpolation="area",
        min_image_size=0,
        max_image_area=float("inf"),
        max_aspect_ratio=float("inf"),
    ) -> None:
        super().__init__()
        self._output_schema.update(
            [
                PredefinedMetadataField.ERROR_MESSAGE,
                PredefinedMetadataField.IMAGE_WIDTH,
                PredefinedMetadataField.IMAGE_HEIGHT,
                PredefinedMetadataField.ORI_IMAGE_WIDTH,
                PredefinedMetadataField.ORI_IMAGE_HEIGHT,
            ]
        )

        self.encode_format = encode_format
        self.image_size = image_size
        self.encode_quality = encode_quality
        self.resize_only_if_bigger = resize_only_if_bigger
        self.min_image_size = min_image_size
        self.max_image_area = max_image_area
        self.max_aspect_ratio = max_aspect_ratio
        if isinstance(resize_mode, str):
            if resize_mode not in ResizeMode.__members__:
                raise ValueError(f"Invalid option for resize_mode: {resize_mode}")
            self.resize_mode = ResizeMode[resize_mode]
        else:
            self.resize_mode = resize_mode

        if self.encode_format not in ["jpeg", "png", "webp"]:
            raise ValueError(f"Invalid encode format {self.encode_format}")
        # `cv2_img_quality` is a variable that is used to store the OpenCV constant value
        # corresponding to the quality parameter for image encoding. The value stored in
        # `cv2_img_quality` is determined based on the selected encoding format (jpg, png, webp) in
        # the `ImageDataConverter` class.
        cv2_img_quality = None
        if self.encode_format == "jpeg":
            cv2_img_quality = int(cv2.IMWRITE_JPEG_QUALITY)
            self.what_ext = "jpeg"
        elif self.encode_format == "png":
            cv2_img_quality = int(cv2.IMWRITE_PNG_COMPRESSION)
            self.what_ext = "png"
        elif self.encode_format == "webp":
            cv2_img_quality = int(cv2.IMWRITE_WEBP_QUALITY)
            self.what_ext = "webp"
        if cv2_img_quality is None:
            raise ValueError(f"Invalid option for encode_format: {cv2_img_quality}")
        self.encode_params = [cv2_img_quality, encode_quality]
        self.upscale_interpolation = inter_str_to_cv2(upscale_interpolation)
        self.downscale_interpolation = inter_str_to_cv2(downscale_interpolation)

    def __call__(self, completed_task: CompletedTask) -> Data | None:
        converted_data = super().__call__(completed_task)
        if completed_task.content is None:
            return converted_data
        cv2.setNumThreads(1)
        img_stream = completed_task.content
        img_stream.seek(0)
        img_buf = np.frombuffer(img_stream.read(), np.uint8)
        img = cv2.imdecode(img_buf, cv2.IMREAD_UNCHANGED)
        if img is None:
            converted_data.metadata[PredefinedMetadataField.ERROR_MESSAGE.name] = (
                "Image decode fail"
            )
            return converted_data
        if len(img.shape) == 3 and img.shape[-1] == 4:
            # alpha matting with white background
            alpha = img[:, :, 3, np.newaxis]
            img = alpha / 255 * img[..., :3] + 255 - alpha
            img = np.rint(img.clip(min=0, max=255)).astype(np.uint8)
        original_height, original_width = img.shape[:2]
        converted_data.metadata[PredefinedMetadataField.ORI_IMAGE_WIDTH.name] = (
            original_width
        )
        converted_data.metadata[PredefinedMetadataField.ORI_IMAGE_HEIGHT.name] = (
            original_height
        )
        # check if image is too small
        if min(original_height, original_width) < self.min_image_size:
            converted_data.metadata[PredefinedMetadataField.ERROR_MESSAGE.name] = (
                "Image too small"
            )
            return converted_data
        if original_height * original_width > self.max_image_area:
            converted_data.metadata[PredefinedMetadataField.ERROR_MESSAGE.name] = (
                "Image too large"
            )
            return converted_data
        # check if wrong aspect ratio
        if (
            max(original_height, original_width) / min(original_height, original_width)
            > self.max_aspect_ratio
        ):
            converted_data.metadata[PredefinedMetadataField.ERROR_MESSAGE.name] = (
                "Image aspect ratio too large"
            )
            return converted_data
        # resizing in following conditions
        if self.resize_mode in (ResizeMode.KEEP_RATIO, ResizeMode.CENTER_CROP):
            downscale = min(original_width, original_height) > self.image_size
            if not self.resize_only_if_bigger or downscale:
                interpolation = (
                    self.downscale_interpolation
                    if downscale
                    else self.upscale_interpolation
                )
                img = A.smallest_max_size(
                    img, self.image_size, interpolation=interpolation
                )
                if self.resize_mode == ResizeMode.CENTER_CROP:
                    img = A.CENTER_CROP(img, self.image_size, self.image_size)
        elif self.resize_mode in (
            ResizeMode.PAD_BORDER,
            ResizeMode.KEEP_RATIO_LARGEST,
        ):
            downscale = max(original_width, original_height) > self.image_size
            if not self.resize_only_if_bigger or downscale:
                interpolation = (
                    self.downscale_interpolation
                    if downscale
                    else self.upscale_interpolation
                )
                img = A.longest_max_size(
                    img, self.image_size, interpolation=interpolation
                )
                if self.resize_mode == ResizeMode.PAD_BORDER:
                    img = A.pad(
                        img,
                        self.image_size,
                        self.image_size,
                        PAD_BORDER_mode=cv2.BORDER_CONSTANT,
                        value=[255, 255, 255],
                    )
        (
            converted_data.metadata[PredefinedMetadataField.IMAGE_HEIGHT.name],
            converted_data.metadata[PredefinedMetadataField.IMAGE_WIDTH.name],
        ) = img.shape[:2]

        img_str = cv2.imencode(
            f".{self.encode_format}", img, params=self.encode_params
        )[1].tobytes()
        converted_data.content = img_str
        return converted_data
