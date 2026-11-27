"""HTTP 请求录制：精确范围、正文限额、内容哈希与展示脱敏。"""
import hashlib
import time
from uuid import uuid4

from artex.assets import normalize


def _scope_urls(repo, task_id):
    return {
        normalize("endpoint", value)
        for value in repo.get_task(task_id)["scope"]
        if value.startswith(("http://", "https://"))
    }


def check_intent_target(repo, task_id, intent_id, target):
    if not intent_id:
        return
    intent = repo.get_node(task_id, intent_id)
    if intent["kind"] != "intent":
        raise ValueError("intent_id 必须是本任务的意图节点")
    if normalize(
        "endpoint", intent["payload"].get("target", "")
    ) != target:
        raise ValueError("流量目标与来源意图不一致")


def record_exchange(
    repo,
    task_id,
    url,
    status,
    body,
    *,
    method="GET",
    intent_id="",
    call_id="",
    request_headers=None,
):
    """将一次完整交换写入证据记录；截断正文不会进入这里。"""
    if not isinstance(body, str):
        raise ValueError("正文必须是文本")
    target = normalize("endpoint", url)
    if target not in _scope_urls(repo, task_id):
        raise ValueError(
            "拒绝记录范围外 URL；重定向需要重新校验范围"
        )
    check_intent_target(repo, task_id, intent_id, target)

    evidence_id = "http-" + uuid4().hex
    # 仅白名单展示头，真实 Cookie/Authorization 不进入共享展示索引。
    shown_headers = {
        key.lower(): value
        for key, value in (request_headers or {}).items()
        if key.lower()
        in ("accept", "content-type", "user-agent")
    }
    value = {
        "evidence_id": evidence_id,
        "task_id": task_id,
        "intent_id": intent_id,
        "call_id": call_id,
        "target": target,
        "method": method,
        "status": int(status),
        "body": body,
        "body_sha256": hashlib.sha256(
            body.encode()
        ).hexdigest(),
        "request_headers": shown_headers,
        "captured_at": time.time(),
        "issue": None,
    }
    if not repo.put_record(
        "evidence", evidence_id, value, task_id
    ):
        raise ValueError("证据 ID 冲突")
    return value


class RecordingClient:
    """先校验范围，再发请求；正文超限时不提交不完整证据。"""

    def __init__(self, repo, task_id, client, max_body=1_000_000):
        if max_body < 1:
            raise ValueError("max_body 必须大于 0")
        self.repo = repo
        self.task_id = task_id
        self.client = client
        self.max_body = max_body

    async def get(self, url, *, intent_id="", call_id=""):
        canonical = normalize("endpoint", url)
        if canonical not in _scope_urls(self.repo, self.task_id):
            raise ValueError("HTTP 请求目标不在任务范围内")
        check_intent_target(
            self.repo, self.task_id, intent_id, canonical
        )

        async with self.client.stream(
            "GET", canonical, follow_redirects=False
        ) as response:
            body = bytearray()
            async for chunk in response.aiter_bytes():
                body.extend(chunk)
                if len(body) > self.max_body:
                    raise ValueError(
                        "正文超过本课采集上限，未提交不完整证据"
                    )
            text = bytes(body).decode(
                response.encoding or "utf-8",
                errors="replace",
            )
            return record_exchange(
                self.repo,
                self.task_id,
                canonical,
                response.status_code,
                text,
                intent_id=intent_id,
                call_id=call_id,
                request_headers=dict(response.request.headers),
            )
