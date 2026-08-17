import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

import ctranslate2
from tokenizers import Tokenizer
from transformers import WhisperFeatureExtractor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--audio",
        type=Path,
        default=Path("tests/data/audio/jfk.npy"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="bfloat16")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.max_length <= 0:
        parser.error("--max-length must be positive")

    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        args.model,
        local_files_only=True,
    )
    audio = np.load(args.audio)
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="np")
    features = ctranslate2.StorageView.from_array(inputs.input_features)

    load_start = time.perf_counter()
    model = ctranslate2.models.Whisper(
        str(args.model),
        device=args.device,
        compute_type=args.compute_type,
    )
    load_seconds = time.perf_counter() - load_start

    language_result = model.detect_language(features)
    language, language_probability = language_result[0][0]
    prompt = [
        "<|startoftranscript|>",
        language,
        "<|transcribe|>",
        "<|notimestamps|>",
    ]

    for _ in range(args.warmup):
        model.generate(features, [prompt], beam_size=1, max_length=args.max_length)

    timings = []
    result = None
    for _ in range(args.iterations):
        start = time.perf_counter()
        result = model.generate(
            features,
            [prompt],
            beam_size=1,
            max_length=args.max_length,
        )[0]
        timings.append(time.perf_counter() - start)

    token_ids = result.sequences_ids[0]
    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    transcription = tokenizer.decode(token_ids, skip_special_tokens=True).strip()

    output = {
        "ctranslate2_version": ctranslate2.__version__,
        "model": str(args.model.resolve()),
        "audio": str(args.audio.resolve()),
        "device": model.device,
        "compute_type": model.compute_type,
        "n_mels": model.n_mels,
        "load_seconds": load_seconds,
        "language": language,
        "language_probability": language_probability,
        "iterations": args.iterations,
        "mean_seconds": statistics.mean(timings),
        "median_seconds": statistics.median(timings),
        "min_seconds": min(timings),
        "max_seconds": max(timings),
        "generated_tokens": len(token_ids),
        "transcription": transcription,
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
