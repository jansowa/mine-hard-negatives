import os
import argparse
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from models import get_reranker_model, rerank
from decouple import config
import uuid
import glob
import shutil
from typing import Iterable, Optional
import re

def _get_next_part_idx(parts_dir: str) -> int:
    parts = _list_parts(parts_dir)
    if not parts:
        return 0
    max_idx = 0
    for p in parts:
        m = re.search(r"part_(\d+)\.parquet$", os.path.basename(p))
        if m:
            idx = int(m.group(1))
            if idx > max_idx:
                max_idx = idx
    return max_idx

def _list_parts(parts_dir: str) -> list[str]:
    parts = sorted(glob.glob(os.path.join(parts_dir, "part_*.parquet")))
    return parts

def _finalize_single_file(parts_dir: str, output_path: str, schema: pa.schema, compression: str = "zstd") -> None:
    parts = _list_parts(parts_dir)
    if not parts:
        print(f"[Finalize] No parts found in: {parts_dir}")
        return

    print(f"[Finalize] Merging {len(parts)} parts into: {output_path}")

    temp_out = f"{output_path}.tmp-{uuid.uuid4().hex}"
    if os.path.exists(temp_out):
        os.remove(temp_out)

    writer = pq.ParquetWriter(temp_out, schema=schema, compression=compression, use_dictionary=True)

    try:
        with tqdm(total=len(parts), desc="Finalizing (merging parts)", unit="part") as pbar:
            for part_path in parts:
                table = pq.read_table(part_path)
                if table.schema != schema:
                    table = table.cast(schema)
                writer.write_table(table)
                pbar.update(1)
        writer.close()
        os.replace(temp_out, output_path)
        print(f"[Finalize] Final file saved at: {output_path}")

        print(f"[Finalize] Removed parts directory: {parts_dir}")
        shutil.rmtree(parts_dir, ignore_errors=True)

    except Exception as e:
        try:
            if writer and writer.is_open:
                writer.close()
        except Exception:
            pass
        if os.path.exists(temp_out):
            os.remove(temp_out)
        raise e

def _pack_pair(query_id: int, document_id: int) -> int:
    # pack two signed 32-bit ints into one 64-bit key
    return (int(query_id) & 0xFFFFFFFF) << 32 | (int(document_id) & 0xFFFFFFFF)

def _load_scored_pairs_from_files(paths: Iterable[str]) -> set[int]:
    scored: set[int] = set()
    for p in paths:
        try:
            pf = pq.ParquetFile(p)
        except Exception:
            continue
        for batch in pf.iter_batches(columns=["query_id", "document_id"], batch_size=200_000):
            tbl = pa.Table.from_batches([batch])
            q = tbl.column("query_id").to_pylist()
            d = tbl.column("document_id").to_pylist()
            for qi, di in zip(q, d):
                scored.add(_pack_pair(qi, di))
    return scored

def _load_scored_pairs(output_path: str) -> set[int]:
    parts_dir = f"{output_path}.parts"
    paths: list[str] = []
    if os.path.isdir(parts_dir):
        paths.extend(_list_parts(parts_dir))
    if os.path.isfile(output_path):
        paths.append(output_path)
    if not paths:
        return set()
    print(f"Scanning already-scored pairs from {len(paths)} file(s)...")
    return _load_scored_pairs_from_files(paths)

def _filter_relevant_to_missing(
    relevant_path: str,
    already_scored_pairs: set[int],
    skip: int,
    offset: Optional[int],
    tmp_dir: str,
    chunk_size: int = 200_000,
) -> tuple[str, int]:
    pf = pq.ParquetFile(relevant_path)
    total_rows = pf.metadata.num_rows

    if skip < 0: skip = 0
    if offset is not None and offset < 0: offset = 0

    window_start = min(skip, total_rows)
    window_end = total_rows if offset is None else min(total_rows, window_start + offset)
    window_size = max(0, window_end - window_start)

    print(f"Total pairs in relevant.parquet: {total_rows:,}")
    if window_size:
        print(f"Selected window (skip={skip}, offset={offset}): {window_size:,} rows "
              f"[{window_start:,} .. {window_end-1:,}]")
    else:
        print(f"Selected window (skip={skip}, offset={offset}): 0 rows")

    if window_size == 0:
        return "", 0

    filtered_schema = pa.schema([('query_id', pa.int32()), ('document_id', pa.int32())])
    filtered_path = os.path.join(tmp_dir, f"filtered_{uuid.uuid4().hex}.parquet")
    writer = pq.ParquetWriter(filtered_path, filtered_schema, compression="zstd", use_dictionary=True)

    seen_rows = 0
    kept = 0

    try:
        with tqdm(total=window_size, desc="Pre-filtering relevant (one-time)", unit="row") as pbar:
            for batch in pf.iter_batches(batch_size=chunk_size, columns=['query_id', 'document_id']):
                batch_len = batch.num_rows
                batch_start = seen_rows
                batch_end = seen_rows + batch_len
                seen_rows = batch_end

                if batch_end <= window_start or batch_start >= window_end:
                    continue

                local_start = max(0, window_start - batch_start)
                local_end = min(batch_len, window_end - batch_start)
                if local_start >= local_end:
                    continue

                sub = batch.slice(local_start, local_end - local_start)
                df = sub.to_pandas()

                if already_scored_pairs:
                    packed = (_pack_pair(q, d) for q, d in zip(df['query_id'].values, df['document_id'].values))
                    mask = [k not in already_scored_pairs for k in packed]
                    df = df.loc[mask]

                if not df.empty:
                    df = df.astype({'query_id': 'int32', 'document_id': 'int32'}, copy=False)
                    tbl = pa.Table.from_pandas(df, schema=filtered_schema, preserve_index=False)
                    writer.write_table(tbl)
                    kept += len(df)

                pbar.update(local_end - local_start)
    finally:
        try:
            writer.close()
        except Exception:
            pass

    if kept == 0:
        try:
            if os.path.exists(filtered_path):
                os.remove(filtered_path)
        except Exception:
            pass
        print("After pre-filtering: 0 rows remain to score.")
        return "", 0

    print(f"After pre-filtering: {kept:,} rows remain to score.")
    return filtered_path, kept

def process_relevant(
    queries_path: str,
    corpus_path: str,
    relevant_path: str,
    output_path: str,
    chunk_size: int,
    reranker_batch_size: int,
    reranker_model_name: str,
    skip: int = 0,
    offset: Optional[int] = None,
) -> None:
    output_dir: str = os.path.dirname(output_path) if output_path else "."
    parts_dir: str = f"{output_path}.parts"
    tmp_dir: str = os.path.join(output_dir or ".", ".tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    print("Starting creation of relevant_with_score dataset (crash-safe, multi-file)...")

    try:
        print(f"Loading data from:\n - {queries_path} (columns: id, text)\n - {corpus_path} (columns: id, text)")
        queries_df: pd.DataFrame = pd.read_parquet(queries_path, columns=['id', 'text'])
        queries_df = queries_df.rename(columns={'text': 'query_text'}).set_index('id')

        corpus_df: pd.DataFrame = pd.read_parquet(corpus_path, columns=['id', 'text'])
        corpus_df = corpus_df.rename(columns={'text': 'document_text'}).set_index('id')
        print("Files queries.parquet and corpus.parquet loaded and prepared.")
        print(f"Queries DataFrame shape: {queries_df.shape}, Corpus DataFrame shape: {corpus_df.shape}")

        already_scored_pairs: set[int] = _load_scored_pairs(output_path)
        already_scored_count = len(already_scored_pairs)
        if already_scored_count:
            print(f"Found {already_scored_count:,} previously scored unique pairs.")
        else:
            print("Found 0 previously scored pairs.")

        filtered_relevant_path, rows_to_score = _filter_relevant_to_missing(
            relevant_path=relevant_path,
            already_scored_pairs=already_scored_pairs,
            skip=skip,
            offset=offset,
            tmp_dir=tmp_dir,
            chunk_size=chunk_size
        )

        if rows_to_score == 0:
            print("Nothing to score after pre-filter. Exiting.")
            return

        tokenizer, reranker = get_reranker_model(reranker_model_name)
        print("Reranker loaded.")

        relevant_extended_schema: pa.Schema = pa.schema([
            ('query_id', pa.int32()),
            ('document_id', pa.int32()),
            ('positive_ranking', pa.float32())
        ])
        print("Schema for output parts defined.")

        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")
        os.makedirs(parts_dir, exist_ok=True)
        print(f"Parts will be written to: {parts_dir}")

        processed_pairs_count: int = 0
        part_idx: int = _get_next_part_idx(parts_dir)

        pf_filtered = pq.ParquetFile(filtered_relevant_path)

        with tqdm(total=rows_to_score, unit="row", desc="Scoring & writing (filtered)") as pbar:
            for batch in pf_filtered.iter_batches(batch_size=chunk_size, columns=['query_id', 'document_id']):
                relevant_chunk_df: pd.DataFrame = batch.to_pandas()
                if relevant_chunk_df.empty:
                    continue

                merged_chunk_df: pd.DataFrame = pd.merge(
                    relevant_chunk_df, queries_df,
                    left_on='query_id', right_index=True, how='inner'
                )
                merged_chunk_df = pd.merge(
                    merged_chunk_df, corpus_df,
                    left_on='document_id', right_index=True, how='inner'
                )
                if merged_chunk_df.empty:
                    pbar.update(len(relevant_chunk_df))
                    continue

                scores_chunk = rerank(
                    tokenizer, reranker,
                    merged_chunk_df['query_text'].values.tolist(),
                    merged_chunk_df['document_text'].values.tolist(),
                    batch_size=reranker_batch_size
                )

                result_chunk_df: pd.DataFrame = pd.DataFrame({
                    'query_id': merged_chunk_df['query_id'].astype('int32', copy=False),
                    'document_id': merged_chunk_df['document_id'].astype('int32', copy=False),
                    'positive_ranking': pd.Series(scores_chunk, dtype='float32')
                })

                result_table_chunk: pa.Table = pa.Table.from_pandas(
                    result_chunk_df,
                    schema=relevant_extended_schema,
                    preserve_index=False
                )

                part_idx += 1
                part_filename = f"part_{part_idx:06d}.parquet"
                temp_path = os.path.join(parts_dir, f".{part_filename}.tmp-{uuid.uuid4().hex}")
                final_path = os.path.join(parts_dir, part_filename)

                pq.write_table(
                    result_table_chunk,
                    temp_path,
                    compression="zstd",
                    use_dictionary=True
                )
                os.replace(temp_path, final_path)
                processed_pairs_count += len(result_chunk_df)

                pbar.update(len(relevant_chunk_df))

                del relevant_chunk_df, merged_chunk_df, scores_chunk, result_chunk_df, result_table_chunk

        print(f"\nDone. Wrote {processed_pairs_count:,} newly scored pairs into parts directory:")
        print(f"  {parts_dir}")

        print("\n[Finalize] Starting finalization into a single file...")
        _finalize_single_file(
            parts_dir=parts_dir,
            output_path=output_path,
            schema=relevant_extended_schema,
            compression="zstd"
        )

    except Exception as e:
        import traceback
        print(f"ERROR: {e}")
        traceback.print_exc()
    finally:
        try:
            if 'filtered_relevant_path' in locals() and filtered_relevant_path and os.path.exists(filtered_relevant_path):
                os.remove(filtered_relevant_path)
        except Exception:
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scoring relevant pairs and saving to Parquet (crash-safe, with finalization).")
    parser.add_argument("--queries_path", type=str, default=config("QUERIES_PATH"), help="Path to parquet file with queries")
    parser.add_argument("--corpus_path", type=str, default=config("CORPUS_PATH"), help="Path to parquet file with corpus")
    parser.add_argument("--relevant_path", type=str, default=config("RELEVANT_PATH"), help="Path to parquet file with relevant connections")
    parser.add_argument("--output_path", type=str, default=config("RELEVANT_WITH_SCORE_PATH"), help="Base path for output (final file will be at output_path)")
    parser.add_argument("--chunk_size", type=int, default=config("PROCESSING_CHUNK_SIZE", cast=int), help="Chunk size for processing")
    parser.add_argument("--reranker_batch_size", type=int, default=config("RERANKER_BATCH_SIZE", cast=int), help="Batch size for reranker")
    parser.add_argument("--reranker_model_name", type=str, default=config("RERANKER_NAME"), help="Name of the reranker model")
    parser.add_argument("--skip", type=int, default=0, help="How many initial items to skip in relevant_path (by position in the file).")
    parser.add_argument("--offset", type=int, default=None, help="How many items in total to process from relevant_path window (after skip). If omitted, process to the end.")
    args = parser.parse_args()

    process_relevant(
        queries_path=args.queries_path,
        corpus_path=args.corpus_path,
        relevant_path=args.relevant_path,
        output_path=args.output_path,
        chunk_size=args.chunk_size,
        reranker_batch_size=args.reranker_batch_size,
        reranker_model_name=args.reranker_model_name,
        skip=args.skip,
        offset=args.offset,
    )
