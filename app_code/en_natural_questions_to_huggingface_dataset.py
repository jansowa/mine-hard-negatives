import argparse
import os
import random
from collections.abc import Iterable
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from decouple import config

BATCH_SIZE = 10_000
NATURAL_QUESTIONS_ID_OFFSET = 10_000_000
DEFAULT_NATURAL_QUESTIONS_DATASET = "sentence-transformers/natural-questions"
DEFAULT_NATURAL_QUESTIONS_CONFIG = "pair"
DEFAULT_MSMARCO_DATASET = "mteb/msmarco"
DEFAULT_MSMARCO_CONFIG = "corpus"
PLACEHOLDER_HF_TOKENS = {"", "hf_TOKEN", "hf_token", "none", "None", "null", "NULL"}


@dataclass
class NaturalQuestionsStats:
    processed_rows: int = 0
    written_queries: int = 0
    written_corpus_docs: int = 0
    written_relevant_pairs: int = 0
    skipped_empty_question: int = 0
    skipped_empty_answer: int = 0


@dataclass
class MsmarcoSamplingStats:
    eligible_seen: int = 0
    skipped_used: int = 0
    skipped_empty_id: int = 0
    skipped_empty_text: int = 0


def write_batch(writer: pq.ParquetWriter, batch: list[dict], schema: pa.Schema) -> None:
    if not batch:
        return
    writer.write_table(pa.Table.from_pylist(batch, schema=schema))
    batch.clear()


def natural_questions_id_from_position(position: int, id_offset: int) -> str:
    return str(id_offset + position)


def natural_questions_position_from_id(row_id: object, id_offset: int) -> int | None:
    try:
        position = int(str(row_id)) - id_offset
    except ValueError:
        return None
    if position < 0:
        return None
    return position


def normalize_hf_token(hf_token: str | None) -> str | None:
    if hf_token is None:
        return None
    hf_token = hf_token.strip()
    if hf_token in PLACEHOLDER_HF_TOKENS:
        return None
    return hf_token


def first_non_empty_text(item: dict, field_names: tuple[str, ...]) -> str:
    for field_name in field_names:
        value = item.get(field_name)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def write_natural_questions_parquets(
    rows: Iterable[dict],
    queries_writer: pq.ParquetWriter,
    corpus_writer: pq.ParquetWriter,
    relevant_writer: pq.ParquetWriter,
    queries_schema: pa.Schema,
    corpus_schema: pa.Schema,
    relevant_schema: pa.Schema,
    id_offset: int,
    batch_size: int,
    source_name: str = "Natural Questions dataset",
) -> NaturalQuestionsStats:
    stats = NaturalQuestionsStats()

    queries_batch: list[dict] = []
    corpus_batch: list[dict] = []
    relevant_batch: list[dict] = []

    for position, item in enumerate(rows):
        stats.processed_rows += 1

        question = first_non_empty_text(item, ("query", "question"))
        if not question:
            stats.skipped_empty_question += 1
            continue

        answer = first_non_empty_text(item, ("answer", "positive"))
        if not answer:
            stats.skipped_empty_answer += 1
            continue

        row_id = natural_questions_id_from_position(position, id_offset)
        queries_batch.append({"id": row_id, "text": question})
        corpus_batch.append({"id": row_id, "text": answer})
        relevant_batch.append({"query_id": row_id, "document_id": row_id})

        stats.written_queries += 1
        stats.written_corpus_docs += 1
        stats.written_relevant_pairs += 1

        if len(queries_batch) >= batch_size:
            write_batch(queries_writer, queries_batch, queries_schema)
            print(f"  Written {stats.written_queries} queries from {source_name}...")

        if len(corpus_batch) >= batch_size:
            write_batch(corpus_writer, corpus_batch, corpus_schema)
            print(f"  Written {stats.written_corpus_docs} Natural Questions corpus docs...")

        if len(relevant_batch) >= batch_size:
            write_batch(relevant_writer, relevant_batch, relevant_schema)
            print(f"  Written {stats.written_relevant_pairs} relevant pairs...")

        if stats.processed_rows % (batch_size * 10) == 0:
            print(f"  Processed {stats.processed_rows} rows from {source_name}...")

    write_batch(queries_writer, queries_batch, queries_schema)
    write_batch(corpus_writer, corpus_batch, corpus_schema)
    write_batch(relevant_writer, relevant_batch, relevant_schema)

    return stats


def reservoir_sample_msmarco_docs(
    docs: Iterable[dict],
    sample_size: int,
    used_ids: set[str],
    rng: random.Random,
) -> tuple[list[dict], MsmarcoSamplingStats]:
    stats = MsmarcoSamplingStats()
    if sample_size <= 0:
        return [], stats

    reservoir: list[dict] = []

    for item in docs:
        doc_id = str(item.get("_id") or "").strip()
        if not doc_id:
            stats.skipped_empty_id += 1
            continue

        text = str(item.get("text") or "").strip()
        if not text:
            stats.skipped_empty_text += 1
            continue

        if doc_id in used_ids:
            stats.skipped_used += 1
            continue

        row = {"id": doc_id, "text": text}
        stats.eligible_seen += 1

        if len(reservoir) < sample_size:
            reservoir.append(row)
            continue

        replace_idx = rng.randint(0, stats.eligible_seen - 1)
        if replace_idx < sample_size:
            reservoir[replace_idx] = row

    return reservoir, stats


def load_used_corpus_ids(dataset_name: str, hf_token: str | None) -> set[str]:
    try:
        ds = load_dataset(dataset_name, data_dir="corpus", split="train", token=hf_token)
    except Exception as first_exc:
        try:
            ds = load_dataset(
                "parquet",
                data_files={"train": f"hf://datasets/{dataset_name}/corpus/*.parquet"},
                split="train",
                token=hf_token,
            )
        except Exception as second_exc:
            first_exc.add_note(f"Fallback hf:// parquet load also failed: {second_exc}")
            raise RuntimeError(
                "Could not load the used-corpus dataset. Set a valid HF_TOKEN with access "
                "or rerun with --skip_used_corpus_filter."
            ) from first_exc

    used_ids: set[str] = set()
    for item in ds:
        used_ids.add(str(item["id"]))
        if len(used_ids) % 100_000 == 0:
            print(f"  Loaded {len(used_ids)} used corpus ids...")
    return used_ids


def load_hf_dataset(
    dataset_name: str,
    config_name: str | None,
    split: str,
    hf_token: str | None,
    streaming: bool,
    source_name: str,
) -> Iterable[dict]:
    try:
        if config_name:
            return load_dataset(dataset_name, config_name, split=split, token=hf_token, streaming=streaming)
        return load_dataset(dataset_name, split=split, token=hf_token, streaming=streaming)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load {source_name} dataset {dataset_name!r} (config={config_name!r}, split={split!r})."
        ) from exc


def main(
    natural_questions_dataset: str,
    natural_questions_config: str | None,
    natural_questions_split: str,
    queries_path: str,
    corpus_path: str,
    relevant_path: str,
    msmarco_dataset: str,
    msmarco_config: str | None,
    msmarco_split: str,
    msmarco_extra_docs: int,
    nq_id_offset: int,
    seed: int,
    skip_used_corpus_filter: bool,
    used_corpus_dataset: str,
    hf_token: str | None,
    batch_size: int,
    streaming: bool,
) -> None:
    hf_token = normalize_hf_token(hf_token)

    for path in (queries_path, corpus_path, relevant_path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    queries_schema = pa.schema([("id", pa.string()), ("text", pa.string())])
    corpus_schema = pa.schema([("id", pa.string()), ("text", pa.string())])
    relevant_schema = pa.schema([("query_id", pa.string()), ("document_id", pa.string())])

    queries_writer = corpus_writer = relevant_writer = None

    try:
        queries_writer = pq.ParquetWriter(queries_path, queries_schema)
        corpus_writer = pq.ParquetWriter(corpus_path, corpus_schema)
        relevant_writer = pq.ParquetWriter(relevant_path, relevant_schema)

        print(
            "Loading English Natural Questions: "
            f"{natural_questions_dataset} (config={natural_questions_config}, split={natural_questions_split})"
        )
        natural_questions_rows = load_hf_dataset(
            dataset_name=natural_questions_dataset,
            config_name=natural_questions_config,
            split=natural_questions_split,
            hf_token=hf_token,
            streaming=streaming,
            source_name="Natural Questions",
        )

        print(f"Processing Natural Questions rows with id offset {nq_id_offset}...")
        nq_stats = write_natural_questions_parquets(
            rows=natural_questions_rows,
            queries_writer=queries_writer,
            corpus_writer=corpus_writer,
            relevant_writer=relevant_writer,
            queries_schema=queries_schema,
            corpus_schema=corpus_schema,
            relevant_schema=relevant_schema,
            id_offset=nq_id_offset,
            batch_size=batch_size,
            source_name=natural_questions_dataset,
        )
        print(
            "Finished Natural Questions. "
            f"Rows: {nq_stats.processed_rows}, queries: {nq_stats.written_queries}, "
            f"corpus docs: {nq_stats.written_corpus_docs}, relevant pairs: {nq_stats.written_relevant_pairs}."
        )
        if nq_stats.skipped_empty_question or nq_stats.skipped_empty_answer:
            print(
                "Skipped Natural Questions rows. "
                f"empty_question={nq_stats.skipped_empty_question}, empty_answer={nq_stats.skipped_empty_answer}."
            )

        used_ids: set[str] = set()
        if skip_used_corpus_filter:
            print("Skipping used-corpus filter for MS MARCO docs.")
        else:
            print(f"Loading used corpus ids from {used_corpus_dataset}...")
            used_ids = load_used_corpus_ids(used_corpus_dataset, hf_token)
            print(f"Loaded {len(used_ids)} used corpus ids to exclude.")

        print(f"Loading MS MARCO corpus: {msmarco_dataset} (config={msmarco_config}, split={msmarco_split})")
        msmarco_rows = load_hf_dataset(
            dataset_name=msmarco_dataset,
            config_name=msmarco_config,
            split=msmarco_split,
            hf_token=hf_token,
            streaming=streaming,
            source_name="MS MARCO corpus",
        )

        print(f"Sampling {msmarco_extra_docs} MS MARCO docs after filtering used ids...")
        reservoir, msmarco_stats = reservoir_sample_msmarco_docs(
            docs=msmarco_rows,
            sample_size=msmarco_extra_docs,
            used_ids=used_ids,
            rng=random.Random(seed),
        )

        if len(reservoir) < msmarco_extra_docs:
            print(
                f"Warning: requested {msmarco_extra_docs} MS MARCO docs, "
                f"but only {len(reservoir)} eligible docs were available."
            )

        written_extra = 0
        start = 0
        while start < len(reservoir):
            chunk = reservoir[start : start + batch_size]
            corpus_writer.write_table(pa.Table.from_pylist(chunk, schema=corpus_schema))
            written_extra += len(chunk)
            start += batch_size
            if written_extra % (batch_size * 10) == 0:
                print(f"  Written {written_extra} sampled MS MARCO docs...")

        print(
            "Finished MS MARCO sampling. "
            f"Eligible seen: {msmarco_stats.eligible_seen}, skipped used: {msmarco_stats.skipped_used}, "
            f"skipped empty id: {msmarco_stats.skipped_empty_id}, skipped empty text: {msmarco_stats.skipped_empty_text}, "
            f"written extra: {written_extra}."
        )
    finally:
        if queries_writer:
            queries_writer.close()
        if corpus_writer:
            corpus_writer.close()
        if relevant_writer:
            relevant_writer.close()

    print("All Parquet files are ready.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert English Natural Questions plus sampled MS MARCO docs to Parquet."
    )
    parser.add_argument(
        "--natural_questions_dataset",
        type=str,
        default=config("NATURAL_QUESTIONS_DATASET", default=DEFAULT_NATURAL_QUESTIONS_DATASET),
    )
    parser.add_argument(
        "--natural_questions_config",
        type=str,
        default=config("NATURAL_QUESTIONS_CONFIG", default=DEFAULT_NATURAL_QUESTIONS_CONFIG),
    )
    parser.add_argument(
        "--natural_questions_split", type=str, default=config("NATURAL_QUESTIONS_SPLIT", default="train")
    )
    parser.add_argument("--queries_path", type=str, default=config("QUERIES_PATH"))
    parser.add_argument("--corpus_path", type=str, default=config("CORPUS_PATH"))
    parser.add_argument("--relevant_path", type=str, default=config("RELEVANT_PATH"))
    parser.add_argument(
        "--msmarco_dataset", type=str, default=config("MSMARCO_DATASET", default=DEFAULT_MSMARCO_DATASET)
    )
    parser.add_argument("--msmarco_config", type=str, default=config("MSMARCO_CONFIG", default=DEFAULT_MSMARCO_CONFIG))
    parser.add_argument("--msmarco_split", type=str, default=config("MSMARCO_SPLIT", default="corpus"))
    parser.add_argument(
        "--msmarco_extra_docs", type=int, default=config("MSMARCO_EXTRA_DOCS", cast=int, default=1_500_000)
    )
    parser.add_argument(
        "--nq_id_offset",
        type=int,
        default=config("NQ_ID_OFFSET", cast=int, default=NATURAL_QUESTIONS_ID_OFFSET),
        help="Offset added to zero-based Natural Questions row positions. Subtract it to recover the source position.",
    )
    parser.add_argument("--seed", type=int, default=config("NATURAL_QUESTIONS_SEED", cast=int, default=42))
    parser.add_argument("--skip_used_corpus_filter", dest="skip_used_corpus_filter", action="store_true")
    parser.add_argument("--use_used_corpus_filter", dest="skip_used_corpus_filter", action="store_false")
    parser.set_defaults(skip_used_corpus_filter=config("SKIP_USED_CORPUS_FILTER", cast=bool, default=False))
    parser.add_argument(
        "--used_corpus_dataset", type=str, default=config("USED_CORPUS_DATASET", default="minehard/negatives2")
    )
    parser.add_argument("--hf_token", type=str, default=config("HF_TOKEN", default=None))
    parser.add_argument(
        "--batch_size", type=int, default=config("NATURAL_QUESTIONS_BATCH_SIZE", cast=int, default=BATCH_SIZE)
    )
    parser.add_argument("--streaming", dest="streaming", action="store_true")
    parser.add_argument("--no_streaming", dest="streaming", action="store_false")
    parser.set_defaults(streaming=config("HF_DATASET_STREAMING", cast=bool, default=True))
    args = parser.parse_args()

    main(
        natural_questions_dataset=args.natural_questions_dataset,
        natural_questions_config=args.natural_questions_config or None,
        natural_questions_split=args.natural_questions_split,
        queries_path=args.queries_path,
        corpus_path=args.corpus_path,
        relevant_path=args.relevant_path,
        msmarco_dataset=args.msmarco_dataset,
        msmarco_config=args.msmarco_config or None,
        msmarco_split=args.msmarco_split,
        msmarco_extra_docs=args.msmarco_extra_docs,
        nq_id_offset=args.nq_id_offset,
        seed=args.seed,
        skip_used_corpus_filter=args.skip_used_corpus_filter,
        used_corpus_dataset=args.used_corpus_dataset,
        hf_token=args.hf_token,
        batch_size=args.batch_size,
        streaming=args.streaming,
    )
