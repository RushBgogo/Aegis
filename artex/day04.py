"""Day 4 安全评估演示：Planner 生成意图，Worker 提交证据，Planner 依据事实收口。

默认使用本地 fixture，不访问网络；--real-planner / --real-worker 会调用 DeepSeek。
observe 仍读取 fixture，真实 HTTP 适配器在后续课程接入。
"""
import argparse
import asyncio
import json

from artex.db import Repository
from artex.planner import Planner
from artex.worker import DEFAULT_FIXTURES, Worker


def build_parser():
    parser = argparse.ArgumentParser(description="Aegis Day4 安全评估演示")
    parser.add_argument(
        "--target",
        action="append",
        dest="targets",
        help="可重复；默认使用本地 fixture 目标",
    )
    parser.add_argument(
        "--goal",
        default="评估目标接口行为并给出可核验证据",
    )
    parser.add_argument(
        "--real-planner",
        action="store_true",
        help="使用 ModelPlanner（会调用 DeepSeek）",
    )
    parser.add_argument(
        "--real-worker",
        action="store_true",
        help="使用 WorkerDeepSeekModel（会调用 DeepSeek；observe 仍读 fixture）",
    )
    return parser


async def main():
    args = build_parser().parse_args()
    targets = args.targets or list(DEFAULT_FIXTURES)
    repo = Repository("sqlite:///:memory:")
    repo.setup()  # 仅演示内存库；生产数据库使用 alembic upgrade head
    try:
        task = repo.create_task("本地样例", args.goal, targets)
        task_id = task["id"]
        repo.transition_task(task_id, "created", "running")

        if args.real_planner:
            from artex.llm_planner import ModelPlanner
            planner = ModelPlanner(repo)
        else:
            planner = Planner(repo)

        before = len(repo.list_nodes(task_id, "intent"))
        await planner.plan(task_id)
        print("首轮新意图:", len(repo.list_nodes(task_id, "intent")) - before)

        before = len(repo.list_nodes(task_id, "intent"))
        await planner.plan(task_id)
        print("重复规划新意图:", len(repo.list_nodes(task_id, "intent")) - before)

        if args.real_worker:
            from artex.worker import WorkerDeepSeekModel
            worker_factory = WorkerDeepSeekModel
        else:
            worker_factory = None

        for intent in repo.list_nodes(task_id, "intent", "open"):
            worker = Worker(repo, model_factory=worker_factory) if worker_factory else Worker(repo)
            await worker.execute(task_id, intent["id"])

        await planner.plan(task_id)
        goals = repo.list_nodes(task_id, "goal")
        facts = repo.list_nodes(task_id, "fact")
        findings = repo.list_nodes(task_id, "finding")
        evidence = repo.list_records("evidence", task_id)
        events = repo.list_events(task_id)
        blocked = repo.list_nodes(task_id, "intent", "blocked")

        status = repo.get_task(task_id)["status"]
        print("任务状态:", status)
        print("目标:", [(goal["text"], goal["state"]) for goal in goals])
        print("事实数:", len(facts))
        print("发现数:", len(findings))
        print("证据数:", len(evidence))
        print("事件数:", len(events))
        print("目标范围:", json.dumps(targets, ensure_ascii=False))
        if blocked:
            print("阻塞意图:", [intent["text"] for intent in blocked])
        if status != "done":
            raise SystemExit(1)
    finally:
        repo.close()


if __name__ == "__main__":
    asyncio.run(main())
