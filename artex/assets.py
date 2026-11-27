"""精确范围内的资产归一化、同步、补全和覆盖度。"""
import asyncio
import hashlib
import ipaddress
import re
import time
from urllib.parse import urlsplit, urlunsplit


def _validate_domain(host):
    """拒绝空标签、超长标签和首尾连字符等非法域名。"""
    if (
        not host
        or len(host) > 253
        or ".." in host
        or host.startswith(".")
        or host.endswith(".")
    ):
        raise ValueError("域名格式错误")
    for label in host.split("."):
        if (
            not label
            or len(label) > 63
            or label.startswith("-")
            or label.endswith("-")
            or not re.fullmatch(r"[a-z0-9-]+", label)
        ):
            raise ValueError("域名格式错误")


def normalize(kind, value):
    """把资产值转成稳定规范形式；kind 是资产身份的一部分。"""
    if not isinstance(value, str):
        raise ValueError("资产值必须是字符串")
    value = value.strip()

    if kind == "ip":
        return str(ipaddress.ip_address(value))

    if kind == "domain":
        try:
            host = value.rstrip(".").lower().encode("idna").decode("ascii")
        except UnicodeError as exc:
            raise ValueError("域名格式错误") from exc
        _validate_domain(host)
        return host

    if kind not in ("service", "endpoint"):
        raise ValueError(
            "资产类型必须是 domain/ip/service/endpoint"
        )

    parts = urlsplit(value)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("需要完整的 HTTP(S) URL")
    if parts.username or parts.password:
        raise ValueError("资产 URL 不能包含用户名和密码")

    try:
        raw_host = parts.hostname.rstrip(".").lower()
        try:
            host = str(ipaddress.ip_address(raw_host))
        except ValueError:
            host = raw_host.encode("idna").decode("ascii")
            _validate_domain(host)
        port = parts.port
    except (UnicodeError, ValueError) as exc:
        raise ValueError("URL 主机或端口格式错误") from exc

    if ":" in host:
        host = "[" + host + "]"
    if port and port != (443 if parts.scheme == "https" else 80):
        host += ":" + str(port)
    return urlunsplit(
        (parts.scheme, host, parts.path or "/", parts.query, "")
    )


class AssetService:
    def __init__(self, repo):
        self.repo = repo

    def _allowed(self, task_id, kind, value):
        """只允许任务显式范围内的精确 URL/主机/IP。"""
        task = self.repo.get_task(task_id)
        allowed_urls, allowed_hosts, allowed_ips = set(), set(), set()
        for raw in task["scope"]:
            try:
                url = normalize("endpoint", raw)
                allowed_urls.add(url)
                host = urlsplit(url).hostname
                allowed_hosts.add(host)
                try:
                    allowed_ips.add(
                        str(ipaddress.ip_address(host))
                    )
                except ValueError:
                    continue
            except ValueError:
                try:
                    allowed_ips.add(normalize("ip", raw))
                except ValueError:
                    allowed_hosts.add(normalize("domain", raw))

        if (
            kind in ("service", "endpoint")
            and value not in allowed_urls
        ):
            raise ValueError(
                "URL 不在任务精确范围内；不会自动扩展到其他路径"
            )
        if kind == "domain" and value not in allowed_hosts:
            raise ValueError(
                "域名不在范围内；不会缩成根域并放大范围"
            )
        if kind == "ip" and value not in allowed_ips:
            raise ValueError(
                "IP 不在显式范围内；DNS 结果不自动获得测试权限"
            )

    def upsert(self, task_id, kind, value, source="manual"):
        """同 kind+规范值的资产合并来源；原来源不会被覆盖。"""
        canonical = normalize(kind, value)
        self._allowed(task_id, kind, canonical)
        key = hashlib.sha256(
            (kind + "\0" + canonical).encode()
        ).hexdigest()

        for _ in range(8):
            old = self.repo.get_record("asset", key, task_id)
            data = dict(old["value"]) if old else {
                "asset_id": key,
                "kind": kind,
                "value": canonical,
                "sources": [],
                "created_at": time.time(),
            }
            data["sources"] = sorted(
                set(data["sources"]) | {source}
            )
            data["last_seen"] = time.time()
            if self.repo.put_record(
                "asset",
                key,
                data,
                task_id,
                expected_version=(
                    old["version"] if old else 0
                ),
            ):
                return data
        raise ValueError("资产被频繁修改，请重试")

    def import_rows(self, task_id, rows, source="scopesentry"):
        """同步导入；单行坏数据不会中止整批。"""
        imported, rejected = [], []
        for index, row in enumerate(rows):
            try:
                if not isinstance(row, dict) or set(row) != {
                    "kind",
                    "value",
                }:
                    raise ValueError(
                        "每条资产必须只有 kind/value 字段"
                    )
                imported.append(
                    self.upsert(
                        task_id, row["kind"], row["value"], source
                    )
                )
            except (ValueError, TypeError) as exc:
                rejected.append(
                    {"index": index, "reason": str(exc)}
                )
        return {"imported": imported, "rejected": rejected}

    async def sync(self, task_id, client, endpoint, extract):
        """外部数据先适配成明确内部格式，再经过范围校验。"""
        response = await client.get(
            endpoint, follow_redirects=False
        )
        response.raise_for_status()
        rows = extract(response.json())
        if not isinstance(rows, list):
            raise ValueError("数据源适配器必须返回资产列表")
        return self.import_rows(task_id, rows)

    def mark_tested(self, task_id, target, fact_id):
        """只有匹配 Fact + Evidence 的结果才计入覆盖度。"""
        canonical = normalize("endpoint", target)
        self._allowed(task_id, "endpoint", canonical)
        fact = self.repo.get_node(task_id, fact_id)
        if fact["kind"] != "fact":
            raise ValueError("覆盖度必须由真实 Fact 支持")

        observation = fact["payload"].get("observation", {})
        evidence_id = fact["payload"].get("evidence_id")
        evidence = self.repo.get_record(
            "evidence", evidence_id or "", task_id
        )
        if evidence is None:
            raise ValueError(
                "Fact 与证据没有指向当前测试对象"
            )
        try:
            observed = normalize(
                "endpoint", observation.get("target", "")
            )
            evidenced = normalize(
                "endpoint",
                evidence["value"].get("target", ""),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Fact 与证据没有指向当前测试对象"
            ) from exc
        if observed != canonical or evidenced != canonical:
            raise ValueError(
                "Fact 与证据没有指向当前测试对象"
            )
        self.repo.put_record(
            "coverage",
            canonical,
            {"target": canonical, "fact_id": fact_id},
            task_id,
        )

    def coverage(self, task_id):
        task = self.repo.get_task(task_id)
        urls = {
            normalize("endpoint", item)
            for item in task["scope"]
            if item.startswith(("http://", "https://"))
        }
        tested = {
            row["value"]["target"]
            for row in self.repo.list_records(
                "coverage", task_id
            )
        }
        matched = tested & urls
        return {
            "total": len(urls),
            "tested": len(matched),
            "ratio": (
                len(matched) / len(urls) if urls else None
            ),
        }

    async def enrich(self, task_id, client, concurrency=3):
        """有界并发补全；HTTP 标题不等于漏洞或覆盖度。"""
        if concurrency < 1:
            raise ValueError("concurrency 必须为正数")
        semaphore = asyncio.Semaphore(concurrency)

        async def probe(row):
            asset = row["value"]
            if asset["kind"] not in ("service", "endpoint"):
                return {
                    "asset_id": asset["asset_id"],
                    "skipped": True,
                }
            async with semaphore:
                self._allowed(
                    task_id, asset["kind"], asset["value"]
                )
                response = await client.get(
                    asset["value"], follow_redirects=False
                )
                title = re.search(
                    r"<title>(.*?)</title>",
                    response.text[:100000],
                    re.I | re.S,
                )
                result = {
                    "asset_id": asset["asset_id"],
                    "status": response.status_code,
                    "title": (
                        title.group(1).strip() if title else ""
                    ),
                }
                old = self.repo.get_record(
                    "asset_probe", asset["asset_id"], task_id
                )
                if not self.repo.put_record(
                    "asset_probe",
                    asset["asset_id"],
                    result,
                    task_id,
                    expected_version=(
                        old["version"] if old else 0
                    ),
                ):
                    raise ValueError("补全结果提交冲突")
                return result

        return await asyncio.gather(
            *[
                probe(row)
                for row in self.repo.list_records(
                    "asset", task_id
                )
            ]
        )
