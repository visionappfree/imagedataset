import aiohttp
import asyncio
import time
import traceback

from dataclasses import dataclass

from .annotator import Annotator
from .download_utils import async_download_content_with_retry
from .data import Task, CompletedTask
from .converter import CompletedTaskConverter
from .writer import Writer
from .processor import Processor, PredefinedMetadataField


@dataclass
class Downloader:
    """
    Manages the download tasks.
    """

    max_concurrent_downloads: int = 48
    timeout_seconds: float = 15
    max_retries: int = 3
    user_agent_token: str | None = None
    converter: CompletedTaskConverter = None
    annotators: list[Annotator] = None

    def _compute_key(self, task):
        return f"{task.id}"

    async def _download_one_task(
        self,
        semaphore: asyncio.Semaphore,
        session: aiohttp.ClientSession,
        task: Task,
        writer: Writer,
        stats: dict,
    ):
        try:
            _, content_stream, err = await async_download_content_with_retry(
                semaphore, session, task, self.max_retries, self.user_agent_token
            )
            converted_data = self.converter(
                CompletedTask(task, self._compute_key(task), content_stream, err)
            )
            for a in self.annotators:
                await a.annotate(converted_data)
            await writer.async_write(converted_data)
            err = converted_data.metadata[PredefinedMetadataField.ERROR_MESSAGE.name]
            if err not in stats:
                stats[err] = 1
            else:
                stats[err] += 1
        except Exception as e:
            err = f"{e}"
            if err not in stats:
                stats[err] = 1
            else:
                stats[err] += 1

    async def _download(
        self,
        download_tasks: list[Task],
        writer: Writer,
        stats: dict,
    ) -> None:
        semaphore = asyncio.Semaphore(self.max_concurrent_downloads)
        connector = aiohttp.TCPConnector(limit=self.max_concurrent_downloads)
        session_timeout = aiohttp.ClientTimeout(
            total=self.timeout_seconds,
            sock_connect=self.timeout_seconds * 0.3,
            ceil_threshold=2,
        )
        async with aiohttp.ClientSession(
            connector=connector, timeout=session_timeout
        ) as session:
            tasks = [
                self._download_one_task(semaphore, session, t, writer, stats)
                for t in download_tasks
            ]
            for task in asyncio.as_completed(tasks):
                await task

    def download(
        self,
        download_tasks: list[Task],
        writer: Writer,
    ) -> dict:
        stats = {}
        try:
            start_time = time.time()
            asyncio.run(self._download(download_tasks, writer, stats))
            end_time = time.time()

            duration = end_time - start_time
            total_succ = (
                stats[Processor.SUCCESS_MESSAGE]
                if Processor.SUCCESS_MESSAGE in stats
                else 0
            )
            print(
                f"download done! download time: {duration}(s), image per second: {len(download_tasks) / duration}, total_succ: {total_succ}\n"
            )
        except Exception as err:  # pylint: disable=broad-except
            traceback.print_exc()
            print(f"download shard failed with error {err}")
        return stats
