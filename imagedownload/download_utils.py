"""the utils for downloading"""

import aiohttp
import asyncio
import io

from tqdm.asyncio import tqdm

from .data import Task

TIME_OUT_ERR_MESSAGE = "Request timed out"
RETRIEABLE_STATUS = set(["Status 503", "Status 429"])
RETRY_BACKOFF_SECONDS = 5


async def async_download_url(
    session: aiohttp.ClientSession, url: str, user_agent_token: str | None = None
) -> tuple[io.BytesIO | None, str | None]:
    """Download url.
    Return content_stream, error status"""
    content_stream = None
    user_agent_string = (
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0"
    )
    if user_agent_token:
        user_agent_string += f" (compatible; {user_agent_token};"
    try:
        async with session.get(
            url, headers={"User-Agent": user_agent_string}
        ) as response:
            if response.status == 200:
                content_stream = io.BytesIO(await response.read())
                return content_stream, None
            else:
                return None, f"Status {response.status}"
    except aiohttp.ClientError as e:
        return None, f"ClientError occurred: {e}"
    except asyncio.TimeoutError:
        return None, TIME_OUT_ERR_MESSAGE
    except Exception as e:  # pylint: disable=broad-except
        if content_stream is not None:
            content_stream.close()
        return None, f"unrecognized error:{e}"


async def async_download_content_with_retry(
    semaphore: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    task: Task,
    max_retries: int = 3,
    user_agent_token: str | None = None,
) -> tuple[Task, io.BytesIO | None, str | None]:
    """Download content with retry.
    Returns url, content_stream, error status"""

    def is_retriable_err(err):
        return err in RETRIEABLE_STATUS

    async with semaphore:
        for i in range(max_retries):
            content_stream, err = await async_download_url(
                session, task.url, user_agent_token
            )
            if content_stream is not None:
                return task, content_stream, err
            if not is_retriable_err(err):
                break
            await asyncio.sleep(RETRY_BACKOFF_SECONDS)
    return task, None, err


async def run_async_download(
    url_list: list[str],
    max_concurrent_downloads: int,
    timeout_seconds: float = 5,
    max_retries: int = 3,
    user_agent_token: str | None = None,
) -> list:
    download_tasks = [Task(url, index) for index, url in enumerate(url_list)]
    semaphore = asyncio.Semaphore(max_concurrent_downloads)
    connector = aiohttp.TCPConnector(limit=max_concurrent_downloads)
    session_timeout = aiohttp.ClientTimeout(
        total=timeout_seconds, sock_connect=timeout_seconds * 0.3, ceil_threshold=2
    )
    data = []
    async with aiohttp.ClientSession(
        connector=connector, timeout=session_timeout
    ) as session:
        tasks = [
            async_download_content_with_retry(
                semaphore, session, t, max_retries, user_agent_token
            )
            for t in download_tasks
        ]
        for task in tqdm(
            asyncio.as_completed(tasks), total=len(tasks), desc="Downloading"
        ):
            url, content_stream, err = await task
            data.append((url, content_stream, err))
    return data
