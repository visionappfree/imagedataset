import json
import os

import asyncio
import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import webdataset as wds

from abc import ABC, abstractmethod

from .data import Data


class Writer(ABC):
    @abstractmethod
    async def async_write(self, data: Data):
        pass

    @abstractmethod
    def close(self):
        pass


class WDSTarWriter(Writer):
    """A writer for WDSTar files."""

    WDS_TAR_WRITER_KEY_FIELD = "__key__"
    WDS_TAR_WRITER_METADATA_FORMAT = "json"

    def __init__(self, tar_fd):
        super().__init__()
        self.tar_writer = wds.TarWriter(tar_fd)
        self.write_lock = asyncio.Lock()

    async def async_write(self, data: Data):
        sample = {
            WDSTarWriter.WDS_TAR_WRITER_KEY_FIELD: data.key,
            data.encode_format: data.content,
        }

        sample[WDSTarWriter.WDS_TAR_WRITER_METADATA_FORMAT] = json.dumps(
            data.metadata, indent=2
        )
        async with self.write_lock:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.tar_writer.write, sample)

    def close(self):
        self.tar_writer.close()


class BufferedParquetWriter(Writer):
    """
    A writer to write samples to parquet files incrementally with a buffer.
    """

    def __init__(self, output_file, schema, buffer_size=100):
        super().__init__()
        self.buffer_size = buffer_size
        self.schema = schema
        self._reset_buffer()
        fs, output_path = fsspec.core.url_to_fs(output_file)
        self.output_fd = fs.open(output_path, "wb")
        self.parquet_writer = pq.ParquetWriter(self.output_fd, schema)
        self.write_lock = asyncio.Lock()

    def _reset_buffer(self):
        self.current_buffer_size = 0
        self.buffer = {k: [] for k in self.schema.names}

    def _add_sample_to_buffer(self, sample):
        for k in self.schema.names:
            self.buffer[k].append(sample[k])
        self.current_buffer_size += 1

    def _flush(self):
        """Write the buffer to disk"""
        if self.current_buffer_size == 0:
            return

        df = pa.Table.from_pydict(self.buffer, self.schema)
        self.parquet_writer.write_table(df)
        self._reset_buffer()

    async def _async_flush(self):
        """Write the buffer to disk"""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._flush)

    async def async_write(self, sample):
        async with self.write_lock:
            if self.current_buffer_size >= self.buffer_size:
                await self._async_flush()
            self._add_sample_to_buffer(sample)

    def close(self):
        """
        The `close` function writes the buffer to disk, closes the Parquet writer if it exists, and
        closes the output file descriptor.
        **only can be called when no additional call to async_write**
        """
        self._flush()
        if self.parquet_writer is not None:
            self.parquet_writer.close()
            self.parquet_writer = None
            self.output_fd.close()


class WebDatasetSampleWriter(Writer):
    """
    A writer to webdataset
    """

    def __init__(self, output_file, schema):
        super().__init__()
        fs, output_path = fsspec.core.url_to_fs(output_file)
        self.tar_fd = fs.open(f"{output_path}.tar", "wb")
        self.tar_writer = WDSTarWriter(self.tar_fd)
        self.buffered_parquet_writer = BufferedParquetWriter(
            f"{output_path}.parquet", schema, buffer_size=100
        )

    async def async_write(self, data: Data):
        """write sample to tars"""
        if data.content is None or data.key is None:
            return

        # some meta data may not be JSON serializable
        for k, v in data.metadata.items():
            if isinstance(v, np.ndarray):
                data.meta[k] = v.tolist()
        await self.tar_writer.async_write(data)
        await self.buffered_parquet_writer.async_write(data.metadata)

    def close(self):
        self.buffered_parquet_writer.close()
        self.tar_writer.close()
        self.tar_fd.close()
