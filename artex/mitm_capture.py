"""可选适配器；用 mitmdump -s 加载。只记录显式范围内的流量。"""
import os

from artex.db import Repository
from artex.traffic import record_exchange


class Capture:
    def __init__(self):
        self.repo = Repository()
        self.task_id = os.environ["ARTEX_CAPTURE_TASK"]
        self.repo.get_task(self.task_id)

    def response(self, flow):
        if flow.response is None:
            return
        if len(flow.response.raw_content or b"") > 1_000_000:
            return
        try:
            record_exchange(
                self.repo,
                self.task_id,
                flow.request.pretty_url,
                flow.response.status_code,
                flow.response.get_text(strict=False) or "",
                method=flow.request.method,
                request_headers=dict(flow.request.headers),
            )
        except ValueError:
            # 非范围流量不入库；代理转发与安全边界由运行环境配置。
            return

    def done(self):
        self.repo.close()


addons = [Capture()]
