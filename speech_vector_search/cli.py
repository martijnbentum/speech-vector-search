import argparse
import json

import numpy as np

from speech_vector_search.evaluate import evaluate_same_word_retrieval
from speech_vector_search.io import load_prototypes, load_token_data, save_prototypes
from speech_vector_search.prototypes import build_subset_mean_prototypes
from speech_vector_search.search import PrototypeIndex
from speech_vector_search.utils import ensure_directory


def build_prototypes_command(args):
    '''build and save prototypes.
    args                     argparse namespace
    '''
    embeddings, metadata = load_token_data(args.embeddings, args.metadata)
    vectors, rows, config = build_subset_mean_prototypes(
        embeddings,
        metadata,
        subset_size=args.subset_size,
        n_subsets=args.n_subsets,
        min_count=args.min_count,
        seed=args.seed,
        strict_non_overlapping=not args.allow_partial,
    )
    ensure_directory(args.output_dir)
    paths = save_prototypes(vectors, rows, args.output_dir, config=config)
    print(json.dumps(paths, indent=2))


def build_index_command(args):
    '''copy normalized prototype files.
    args                     argparse namespace
    '''
    vectors, metadata = load_prototypes(args.vectors, args.metadata)
    ensure_directory(args.output_dir)
    np.save(args.output_dir + "/prototypes.npy", vectors)
    from speech_vector_search.io import save_metadata_jsonl
    save_metadata_jsonl(metadata, args.output_dir + "/metadata.jsonl")
    print(json.dumps({"vectors": args.output_dir + "/prototypes.npy", "metadata": args.output_dir + "/metadata.jsonl"}, indent=2))


def query_command(args):
    '''query nearest neighbours.
    args                     argparse namespace
    '''
    vectors, metadata = load_prototypes(args.vectors, args.metadata)
    index = PrototypeIndex(vectors, metadata, backend=args.backend)
    if args.query_index is not None:
        result = index.query_by_index(args.query_index, top_k=args.top_k)
    else:
        vector = np.asarray(np.load(args.query_vector), dtype=float)
        result = index.query(vector, top_k=args.top_k)
    result["scores"] = result["scores"].tolist()
    result["indices"] = result["indices"].tolist()
    print(json.dumps(result, indent=2))


def evaluate_command(args):
    '''evaluate same-word retrieval.
    args                     argparse namespace
    '''
    vectors, metadata = load_prototypes(args.vectors, args.metadata)
    result = evaluate_same_word_retrieval(vectors, metadata, top_k=args.top_k)
    print(json.dumps(result, indent=2))


def build_parser():
    '''create command line parser.
    '''
    parser = argparse.ArgumentParser(prog="speech-vector-search")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_prototypes = subparsers.add_parser("build-prototypes")
    build_prototypes.add_argument("--embeddings", required=True)
    build_prototypes.add_argument("--metadata", required=True)
    build_prototypes.add_argument("--output-dir", required=True)
    build_prototypes.add_argument("--subset-size", type=int, required=True)
    build_prototypes.add_argument("--n-subsets", type=int, required=True)
    build_prototypes.add_argument("--min-count", type=int, default=None)
    build_prototypes.add_argument("--seed", type=int, default=0)
    build_prototypes.add_argument("--allow-partial", action="store_true")
    build_prototypes.set_defaults(func=build_prototypes_command)

    build_index = subparsers.add_parser("build-index")
    build_index.add_argument("--vectors", required=True)
    build_index.add_argument("--metadata", required=True)
    build_index.add_argument("--output-dir", required=True)
    build_index.set_defaults(func=build_index_command)

    query = subparsers.add_parser("query")
    query.add_argument("--vectors", required=True)
    query.add_argument("--metadata", required=True)
    query.add_argument("--query-index", type=int)
    query.add_argument("--query-vector")
    query.add_argument("--top-k", type=int, default=5)
    query.add_argument("--backend", default="auto")
    query.set_defaults(func=query_command)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--vectors", required=True)
    evaluate.add_argument("--metadata", required=True)
    evaluate.add_argument("--top-k", type=int, default=5)
    evaluate.set_defaults(func=evaluate_command)

    return parser


def main():
    '''run command line interface.
    '''
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "query" and args.query_index is None and args.query_vector is None:
        parser.error("query requires --query-index or --query-vector")
    args.func(args)


if __name__ == "__main__":
    main()
