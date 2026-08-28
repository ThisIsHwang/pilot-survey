from __future__ import annotations

from stackpilot.query_credit_qwen35_runtime import install_no_think_completion


def main() -> None:
    # Install before the ordinary collector starts any worker threads. This
    # keeps the shared causal-query code unchanged for older experiments.
    install_no_think_completion()
    from stackpilot.query_credit_weekend_collect import main as collect_main

    collect_main()


if __name__ == "__main__":
    main()
