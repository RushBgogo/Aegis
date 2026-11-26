"""Day 11 评估与事件流演示：固定 20 场景，数据库游标可恢复。"""
import argparse
import asyncio
import json

from artex.db import Repository
from artex.events import EventService, merge_events
from artex.evaluation import evaluate


def build_parser():
    parser = argparse.ArgumentParser(description="Aegis Day11 评估/SSE 演示")
    parser.add_argument(
        "--events",
        type=int,
        default=300,
        help="慢消费者读取的事件数",
    )
    return parser


async def main():
    args = build_parser().parse_args()
    if args.events < 1:
        raise SystemExit("--events must be >= 1")

    report = await evaluate()
    summary = {
        key: value for key, value in report.items() if key != "cases"
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for item in report["cases"]:
        print(item["case"], "PASS" if item["passed"] else "FAIL")

    repo = Repository("sqlite:///:memory:")
    repo.setup()
    try:
        task = repo.create_task("慢消费者", "事件不丢")["id"]
        other = repo.create_task("另一任务", "制造合法 ID 空隙")["id"]
        expected = []
        for index in range(args.events):
            expected.append(
                repo.append_event(task, "step", {"index": index})
            )
            repo.append_event(other, "step", {"index": index})

        iterator = EventService(repo).tail(task)
        received = []
        try:
            for index in range(len(expected)):
                received.append(await anext(iterator))
                if index % 100 == 0:
                    await asyncio.sleep(0.01)
        finally:
            await iterator.aclose()

        merged = merge_events(received[:20], received)
        print(
            "慢消费者：",
            len(merged),
            "条；ID 与数据库一致：",
            [event["id"] for event in merged]
            == [event["id"] for event in expected],
        )
        print("SSE 样例：", EventService.format_sse(received[0]))
    finally:
        repo.close()

    if report["passed"] != report["count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
