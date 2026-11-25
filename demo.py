"""兼容入口：运行 Day5 多 Worker 调度演示。"""
import asyncio

from artex.day05 import main


if __name__ == "__main__":
    asyncio.run(main())
