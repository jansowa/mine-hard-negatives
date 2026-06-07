from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import datasets

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from curated_negatives.flag_embedding import ensure_parent_dir, write_jsonl_rows
else:
    from .flag_embedding import ensure_parent_dir, write_jsonl_rows

DEFAULT_DATASET_NAME = "lightonai/embeddings-fine-tuning"
DEFAULT_SPLITS = ("trivia", "hotpotqa", "nq", "msmarco", "fever", "squadv2", "fiqa")
LIGHTONAI_COMPONENTS = {"documents", "queries", "scores"}


def _select_split(dataset: datasets.Dataset | datasets.DatasetDict, split: str) -> datasets.Dataset:
    if isinstance(dataset, datasets.DatasetDict):
        return dataset[split]
    return dataset


def _build_index(dataset: datasets.Dataset, id_column: str) -> dict[Any, int]:
    index: dict[Any, int] = {}
    for row_index, item_id in enumerate(dataset[id_column]):
        index[item_id] = row_index
        index[str(item_id)] = row_index
    return index


def _lookup_text(dataset: datasets.Dataset, index: dict[Any, int], item_id: Any, text_column: str) -> str:
    try:
        row = dataset[index[item_id]]
    except KeyError:
        row = dataset[index[str(item_id)]]
    value = row[text_column]
    return "" if value is None else str(value)


class LightOnKDToFlagEmbedding:
    """Convert LightOn KD score rows to FlagEmbedding-style training rows."""

    def __init__(
        self,
        queries: datasets.Dataset | datasets.DatasetDict,
        documents: datasets.Dataset | datasets.DatasetDict,
        split: str = "train",
        num_negatives: int = 32,
        nv_threshold: float = 0.95,
        prompt: str = "",
        dataset_type: str = "retrieval",
        source_dataset: str = DEFAULT_DATASET_NAME,
        include_ids: bool = True,
    ) -> None:
        if num_negatives <= 0:
            raise ValueError("num_negatives must be greater than 0")

        self.queries = _select_split(queries, split)
        self.documents = _select_split(documents, split)
        self.num_negatives = num_negatives
        self.nv_threshold = nv_threshold
        self.prompt = prompt
        self.dataset_type = dataset_type
        self.source_dataset = source_dataset
        self.include_ids = include_ids

        self.queries_index = _build_index(self.queries, "query_id")
        self.documents_index = _build_index(self.documents, "document_id")

    def has_enough_negatives(self, example: dict[str, Any]) -> bool:
        scores = example["scores"]
        if not scores:
            return False
        positive_score = float(scores[0])
        valid_count = sum(1 for score in scores[1:] if float(score) < self.nv_threshold * positive_score)
        return valid_count >= self.num_negatives

    def map_to_flag_embedding(self, example: dict[str, Any]) -> dict[str, Any]:
        query_id = example["query_id"]
        document_ids = example["document_ids"]
        scores = example["scores"]
        if not document_ids or not scores:
            raise ValueError("LightOn score rows must contain document_ids and scores")

        positive_id = document_ids[0]
        positive_score = float(scores[0])
        query_text = _lookup_text(self.queries, self.queries_index, query_id, "query")
        positive_text = _lookup_text(self.documents, self.documents_index, positive_id, "document")

        negative_ids: list[Any] = []
        negative_texts: list[str] = []
        negative_scores: list[float] = []
        for document_id, score in zip(document_ids[1:], scores[1:]):
            score = float(score)
            if score >= self.nv_threshold * positive_score:
                continue
            negative_ids.append(document_id)
            negative_texts.append(_lookup_text(self.documents, self.documents_index, document_id, "document"))
            negative_scores.append(score)
            if len(negative_ids) >= self.num_negatives:
                break

        row: dict[str, Any] = {
            "query": query_text,
            "pos": [positive_text],
            "neg": negative_texts,
            "pos_scores": [positive_score],
            "neg_scores": negative_scores,
            "pos_is_synthetic": [False],
            "prompt": self.prompt,
            "type": self.dataset_type,
            "source_dataset": self.source_dataset,
        }
        if self.include_ids:
            row.update(
                {
                    "query_id": query_id,
                    "pos_id": [positive_id],
                    "neg_id": negative_ids,
                }
            )
        return row


def _load_lightonai_component(
    dataset_name: str,
    component_name: str,
    split: str,
    hf_cache_dir: str | None,
    load_num_proc: int | None,
) -> datasets.Dataset:
    kwargs: dict[str, Any] = {
        "path": dataset_name,
        "name": component_name,
        "split": split,
        "cache_dir": hf_cache_dir,
    }
    if component_name in LIGHTONAI_COMPONENTS:
        kwargs["data_files"] = {split: f"{component_name}/{split}-*"}
        kwargs["verification_mode"] = "no_checks"
    if load_num_proc is not None:
        kwargs["num_proc"] = load_num_proc
    return datasets.load_dataset(**kwargs)


def build_lightonai_flag_embedding_dataset(
    dataset_name: str,
    split: str,
    num_negatives: int,
    nv_threshold: float,
    prompt: str = "",
    dataset_type: str = "retrieval",
    include_ids: bool = True,
    hf_cache_dir: str | None = None,
    load_num_proc: int | None = None,
    processing_num_proc: int | None = None,
) -> datasets.Dataset:
    scores = _load_lightonai_component(dataset_name, "scores", split, hf_cache_dir, load_num_proc)
    queries = _load_lightonai_component(dataset_name, "queries", split, hf_cache_dir, load_num_proc)
    documents = _load_lightonai_component(dataset_name, "documents", split, hf_cache_dir, load_num_proc)
    processor = LightOnKDToFlagEmbedding(
        queries,
        documents,
        num_negatives=num_negatives,
        nv_threshold=nv_threshold,
        prompt=prompt,
        dataset_type=dataset_type,
        source_dataset=dataset_name,
        include_ids=include_ids,
    )

    filter_kwargs: dict[str, Any] = {"desc": f"Filtering {split} rows with <{num_negatives} valid negatives"}
    map_kwargs: dict[str, Any] = {
        "remove_columns": scores.column_names,
        "desc": f"Creating FlagEmbedding rows for {split}",
    }
    if processing_num_proc is not None:
        filter_kwargs["num_proc"] = processing_num_proc
        map_kwargs["num_proc"] = processing_num_proc

    return scores.filter(processor.has_enough_negatives, **filter_kwargs).map(
        processor.map_to_flag_embedding,
        **map_kwargs,
    )


def _ensure_pos_is_synthetic(dataset: datasets.Dataset) -> datasets.Dataset:
    if "pos_is_synthetic" in dataset.column_names:
        return dataset
    return dataset.map(
        lambda row: {"pos_is_synthetic": [False] * len(row.get("pos") or [])},
        desc="Adding synthetic-positive metadata",
    )


def load_or_build_lightonai_flag_embedding_dataset(
    dataset_name: str,
    split: str,
    num_negatives: int,
    nv_threshold: float,
    prompt: str = "",
    dataset_type: str = "retrieval",
    include_ids: bool = True,
    processed_cache_dir: str | None = None,
    hf_cache_dir: str | None = None,
    load_num_proc: int | None = None,
    processing_num_proc: int | None = None,
) -> datasets.Dataset:
    cache_path = os.path.join(processed_cache_dir, split) if processed_cache_dir else None
    if cache_path and os.path.isdir(cache_path):
        print(f"Loaded processed {split} split from {cache_path}")
        return _ensure_pos_is_synthetic(datasets.load_from_disk(cache_path))

    dataset = build_lightonai_flag_embedding_dataset(
        dataset_name=dataset_name,
        split=split,
        num_negatives=num_negatives,
        nv_threshold=nv_threshold,
        prompt=prompt,
        dataset_type=dataset_type,
        include_ids=include_ids,
        hf_cache_dir=hf_cache_dir,
        load_num_proc=load_num_proc,
        processing_num_proc=processing_num_proc,
    )
    if cache_path:
        ensure_parent_dir(os.path.join(cache_path, "dataset_info.json"))
        dataset.save_to_disk(cache_path)
    return _ensure_pos_is_synthetic(dataset)


def export_lightonai_split_to_jsonl(
    output_path: str,
    dataset_name: str = DEFAULT_DATASET_NAME,
    split: str = "nq",
    num_negatives: int = 50,
    nv_threshold: float = 0.99,
    prompt: str = "",
    dataset_type: str = "retrieval",
    include_ids: bool = True,
    processed_cache_dir: str | None = None,
    hf_cache_dir: str | None = None,
    load_num_proc: int | None = None,
    processing_num_proc: int | None = None,
) -> None:
    dataset = load_or_build_lightonai_flag_embedding_dataset(
        dataset_name=dataset_name,
        split=split,
        num_negatives=num_negatives,
        nv_threshold=nv_threshold,
        prompt=prompt,
        dataset_type=dataset_type,
        include_ids=include_ids,
        processed_cache_dir=processed_cache_dir,
        hf_cache_dir=hf_cache_dir,
        load_num_proc=load_num_proc,
        processing_num_proc=processing_num_proc,
    )
    write_jsonl_rows(output_path, dataset)
    print(f"Wrote {len(dataset):,} FlagEmbedding rows to {output_path}")


def _parse_splits(raw_splits: str) -> list[str]:
    splits = [item.strip() for item in raw_splits.split(",") if item.strip()]
    if not splits:
        raise ValueError("At least one split is required")
    return splits


def _output_path_for_split(output_path: str | None, output_dir: str, split: str, total_splits: int) -> str:
    if output_path:
        if total_splits > 1 and "{split}" not in output_path:
            raise ValueError("--output_path must contain {split} when exporting more than one split")
        return output_path.format(split=split)
    return os.path.join(output_dir, f"{split}.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert LightOn mined negatives to FlagEmbedding JSONL.")
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET_NAME)
    parser.add_argument("--splits", type=str, default=",".join(DEFAULT_SPLITS))
    parser.add_argument("--output_dir", type=str, default="data/lightonai_flag_embedding")
    parser.add_argument("--output_path", type=str, default=None, help="Optional path or pattern with {split}.")
    parser.add_argument("--num_negatives", type=int, default=50)
    parser.add_argument("--nv_threshold", type=float, default=0.99)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--type", dest="dataset_type", type=str, default="retrieval")
    parser.add_argument("--processed_cache_dir", type=str, default="data/lightonai_flag_embedding_cache")
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--load_num_proc", type=int, default=None)
    parser.add_argument("--processing_num_proc", type=int, default=None)
    parser.add_argument("--include_ids", dest="include_ids", action="store_true")
    parser.add_argument("--no_include_ids", dest="include_ids", action="store_false")
    parser.set_defaults(include_ids=True)
    args = parser.parse_args()

    splits = _parse_splits(args.splits)
    for split in splits:
        output_path = _output_path_for_split(args.output_path, args.output_dir, split, len(splits))
        export_lightonai_split_to_jsonl(
            output_path=output_path,
            dataset_name=args.dataset_name,
            split=split,
            num_negatives=args.num_negatives,
            nv_threshold=args.nv_threshold,
            prompt=args.prompt,
            dataset_type=args.dataset_type,
            include_ids=args.include_ids,
            processed_cache_dir=args.processed_cache_dir,
            hf_cache_dir=args.hf_cache_dir,
            load_num_proc=args.load_num_proc,
            processing_num_proc=args.processing_num_proc,
        )


if __name__ == "__main__":
    main()
