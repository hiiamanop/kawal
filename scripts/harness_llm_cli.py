from __future__ import annotations

import argparse

from services.harness.cassette import CassetteChatClient, CassetteError
from services.harness.llm import OpenRouterChatClient


def add_llm_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=None, help="OpenRouter model id")
    parser.add_argument("--input-price-per-million", type=float, default=0.0)
    parser.add_argument("--output-price-per-million", type=float, default=0.0)
    parser.add_argument(
        "--no-reasoning",
        action="store_true",
        help="Disable model reasoning (useful for reasoning models that emit no content)",
    )
    parser.add_argument(
        "--cassette",
        default=None,
        help="Path to a cassette JSON file for recording or replay",
    )
    parser.add_argument(
        "--replay-cassette",
        action="store_true",
        help="Replay --cassette without calling the live model",
    )
    parser.add_argument(
        "--cassette-overwrite",
        action="store_true",
        help="Replace an existing cassette when recording",
    )


def build_chat_client(
    args: argparse.Namespace,
) -> CassetteChatClient | OpenRouterChatClient:
    if args.replay_cassette and not args.cassette:
        raise CassetteError(
            "CASSETTE_REQUIRED", "--replay-cassette requires --cassette"
        )
    live = OpenRouterChatClient(
        model_id=args.model,
        input_price_per_million=args.input_price_per_million,
        output_price_per_million=args.output_price_per_million,
        reasoning_enabled=False if args.no_reasoning else None,
    )
    if not args.cassette:
        return live
    if args.replay_cassette:
        return CassetteChatClient(
            args.cassette,
            mode="replay",
            input_price_per_million=args.input_price_per_million,
            output_price_per_million=args.output_price_per_million,
        )
    return CassetteChatClient(
        args.cassette,
        mode="record",
        inner=live,
        input_price_per_million=args.input_price_per_million,
        output_price_per_million=args.output_price_per_million,
        overwrite=args.cassette_overwrite,
    )
