"""兼容入口：运行 Day9 模型预算/重试/熔断演示。"""
import asyncio

from artex.day09 import main


if __name__ == "__main__":
    asyncio.run(main())
