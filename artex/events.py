"""SSE 从数据库按游标读取；进程通知不承担记录投递。"""
import asyncio
import json
import time


class EventService:
    def __init__(self, repo, poll_seconds=0.02):
        if poll_seconds <= 0:
            raise ValueError("poll_seconds 必须大于 0")
        self.repo, self.poll_seconds = repo, poll_seconds

    async def tail(self, task_id, since=0, stop=None):
        """有界读取接口：测试和 HTTP 层都用它，避免对无限流做完整 read。"""
        if since < 0:
            raise ValueError("事件游标不能小于 0")
        self.repo.get_task(task_id)
        while stop is None or not stop.is_set():
            rows = self.repo.list_events(
                task_id, since=since, limit=1000
            )
            if not rows:
                await asyncio.sleep(self.poll_seconds)
                continue
            for row in rows:
                yield row
                since = row["id"]

    @staticmethod
    def format_sse(event):
        """kind/data 全部放进 JSON，避免换行注入 SSE 控制字段。"""
        body = json.dumps(
            event, ensure_ascii=False, separators=(",", ":")
        )
        return f"id: {event['id']}\nevent: activity\ndata: {body}\n\n"

    async def sse(self, task_id, since=0, heartbeat=15):
        if since < 0:
            raise ValueError("事件游标不能小于 0")
        self.repo.get_task(task_id)
        last_sent = time.monotonic()
        while True:
            rows = self.repo.list_events(
                task_id, since=since, limit=1000
            )
            for row in rows:
                yield self.format_sse(row)
                since = row["id"]
                last_sent = time.monotonic()
            if not rows:
                if time.monotonic() - last_sent >= heartbeat:
                    yield ": ping\n\n"
                    last_sent = time.monotonic()
                await asyncio.sleep(self.poll_seconds)


def merge_events(previous, incoming):
    """重连允许重传；客户端以全局事件 id 去重并排序。"""
    by_id = {
        event["id"]: event for event in [*previous, *incoming]
    }
    return [by_id[key] for key in sorted(by_id)]
