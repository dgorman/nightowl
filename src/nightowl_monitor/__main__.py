"""Entry point for `python -m nightowl_monitor`."""

from .poller import main


def run() -> None:  # pragma: no cover - thin wrapper
    raise SystemExit(main())


if __name__ == "__main__":  # pragma: no cover
    run()
