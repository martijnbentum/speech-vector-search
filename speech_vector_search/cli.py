import argparse
import json

import numpy as np

from speech_vector_search import evaluate
from speech_vector_search import io
from speech_vector_search import prototypes
from speech_vector_search import search
from speech_vector_search import utils


def build_prototypes_command(args):
    '''build and save prototypes.
    args                     argparse namespace
    '''
    embeddings, metadata = io.load_token_data(args.embeddings, args.metadata)
    vectors, rows, config = prototypes.build_subset_mean_prototypes(embeddings,
        metadata, subset_size=args.subset_size, n_subsets=args.n_subsets,
        min_count=args.min_count, seed=args.seed,
        strict_non_overlapping=not args.allow_partial)
    utils.ensure_directory(args.output_dir)
    paths = io.save_prototypes(vectors, rows, args.output_dir, config=config)
    print(json.dumps(paths, indent=2))


def build_index_command(args):
    '''copy normalized prototype files.
    args                     argparse namespace
    '''
    vectors, metadata = io.load_prototypes(args.vectors, args.metadata)
    utils.ensure_directory(args.output_dir)
    vectors_path = args.output_dir + "/prototypes.npy"
    metadata_path = args.output_dir + "/metadata.jsonl"
    np.save(vectors_path, vectors)
    io.save_metadata_jsonl(metadata, metadata_path)
    paths = {"vectors": vectors_path, "metadata": metadata_path}
    print(json.dumps(paths, indent=2))


def query_command(args):
    '''query nearest neighbours.
    args                     argparse namespace
    '''
    vectors, metadata = io.load_prototypes(args.vectors, args.metadata)
    index = search.PrototypeIndex(vectors, metadata, backend=args.backend)
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
    vectors, metadata = io.load_prototypes(args.vectors, args.metadata)
    result = evaluate.evaluate_same_word_retrieval(vectors, metadata,
        top_k=args.top_k)
    print(json.dumps(result, indent=2))


def main():
    '''run command line interface.
    no parameters            reads command line arguments directly
    '''
    parser = build_parser()
    args = parser.parse_args()
    missing_query = args.command == "query" and args.query_index is None
    missing_query = missing_query and args.query_vector is None
    if missing_query:
        parser.error("query requires --query-index or --query-vector")
    args.func(args)


def build_parser():
    '''create command line parser.
    no parameters            returns configured argument parser
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

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--vectors", required=True)
    evaluate_parser.add_argument("--metadata", required=True)
    evaluate_parser.add_argument("--top-k", type=int, default=5)
    evaluate_parser.set_defaults(func=evaluate_command)
    return parser


if __name__ == "__main__":
    main()
