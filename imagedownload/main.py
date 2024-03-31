import fire
import fsspec
import math
import os
import pyarrow as pa
import pandas as pd
import time

from multiprocessing import get_context

from .annotator import GeminiImageCaptionAnnotator, AnthropicImageCaptionAnnotator
from .converter import (
    ImageDataConverter,
    DataFrameToTaskConverter,
)
from .data import Task
from .downloader import Downloader
from .writer import WebDatasetSampleWriter


def download_one_shard(args) -> dict:
    (
        input_schema,
        shard_id,
        shard_task,
        output_file_path,
        max_concurrent_downloads_per_process,
        enable_gemini_caption,
        gemini_qps_limit,
        enable_anthropic_caption,
        anthropic_qps_limit,
    ) = args
    image_converter = ImageDataConverter()
    schema_list = image_converter.pa_metadata_schema()
    schema_list.extend(input_schema)
    annotators = []
    if enable_gemini_caption:
        gemini_caption_annotator = GeminiImageCaptionAnnotator(
            gemini_api_keys=[
                os.getenv("GEMINI_API_KEY"),
            ],
            qps_limit=gemini_qps_limit,
        )
        schema_list.extend(gemini_caption_annotator.pa_metadata_schema())
        annotators.append(gemini_caption_annotator)
    if enable_anthropic_caption:
        anthropic_annotator = AnthropicImageCaptionAnnotator(
            anthropic_api_keys=[os.getenv("ANTHROPIC_API_KEY")],
            qps_limit=anthropic_qps_limit,
        )
        schema_list.extend(anthropic_annotator.pa_metadata_schema())
        annotators.append(anthropic_annotator)
    output_schema = pa.schema(schema_list)

    writer = WebDatasetSampleWriter(output_file_path, schema=output_schema)
    image_downloader = Downloader(
        max_concurrent_downloads=max_concurrent_downloads_per_process,
        converter=image_converter,
        annotators=annotators,
    )
    stats = image_downloader.download(
        shard_id, download_tasks=shard_task, writer=writer
    )
    writer.close()
    for a in annotators:
        a.close()
    return stats


def download(
    input_csv_file_path: str,
    output_folder: str,
    input_delimiter: str = ",",
    url_column="url",
    number_sample_per_shard: int = 10000,
    processes_count=1,
    max_concurrent_downloads_per_process: int = 48,
    enable_gemini_caption=False,
    gemini_qps_limit=0.8,
    enable_anthropic_caption=False,
    anthropic_qps_limit=0.8,
):
    fs, output_folder_path = fsspec.core.url_to_fs(output_folder)
    if not fs.exists(output_folder_path):
        fs.makedir(output_folder_path)
    elif not fs.isdir(output_folder_path):
        print(f"output folder is not a dir.")
        return None
    df = pd.read_csv(input_csv_file_path, delimiter=input_delimiter)
    task_converter = DataFrameToTaskConverter(data=df, url_column=url_column)
    tasks = task_converter.get_tasks()
    input_schema = task_converter.pa_metadata_schema()
    num_tasks = df.shape[0]
    num_shards = math.ceil(num_tasks / number_sample_per_shard)
    print(f"tasks split to {num_shards} shards.")
    num_shards_digits = math.ceil(math.log10(num_shards))
    shard_tasks = [
        tasks[x : min(x + number_sample_per_shard, num_tasks)]
        for x in range(0, num_tasks, number_sample_per_shard)
    ]
    shard_tasks_with_params = [
        (
            input_schema,
            shard_id,
            shard_task,
            os.path.join(output_folder_path, f"{shard_id:0{num_shards_digits}}"),
            max_concurrent_downloads_per_process,
            enable_gemini_caption,
            gemini_qps_limit,
            enable_anthropic_caption,
            anthropic_qps_limit,
        )
        for shard_id, shard_task in enumerate(shard_tasks)
    ]

    start_time = time.time()
    ctx = get_context("spawn")
    with ctx.Pool(processes_count, maxtasksperchild=5) as process_pool:
        stats = process_pool.map(download_one_shard, shard_tasks_with_params)
        process_pool.terminate()
        process_pool.join()
        del process_pool
    duration = time.time() - start_time
    print(
        f"all download done!\ntotal download time: {duration}(s), image per second: {num_tasks / duration}\n"
    )
    final_stats = {}
    for s in stats:
        for k, v in s.items():
            if k not in final_stats:
                final_stats[k] = v
            else:
                final_stats[k] += v

    sorted_stats = [i for i in final_stats.items()]
    sorted_stats.sort(key=lambda i: i[1], reverse=True)
    for k, v in sorted_stats[:10]:
        print(f"download {k}, count: {v}")


def main():
    fire.Fire(download)


if __name__ == "__main__":
    main()
