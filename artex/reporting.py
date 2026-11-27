"""把发现与已核对证据绑定，输出可识别陈旧状态的报告和复测历史。"""
import hashlib
import time
from uuid import uuid4

from artex.assets import normalize
from artex.traffic import record_exchange


class ReportService:
    def __init__(self, repo):
        self.repo = repo

    def _evidence(self, task_id, evidence_id):
        record = self.repo.get_record(
            "evidence", evidence_id, task_id
        )
        if (
            not record
            or record["value"].get("task_id") != task_id
        ):
            raise ValueError("证据不属于本任务或不存在")

        value = record["value"]
        scope = {
            normalize("endpoint", target)
            for target in self.repo.get_task(task_id)["scope"]
            if target.startswith(("http://", "https://"))
        }
        if (
            not value.get("body")
            or normalize(
                "endpoint", value.get("target", "")
            )
            not in scope
        ):
            raise ValueError("证据正文为空或目标不在任务范围")
        if (
            value.get("body_sha256")
            and hashlib.sha256(
                value["body"].encode()
            ).hexdigest()
            != value["body_sha256"]
        ):
            raise ValueError("证据正文与内容哈希不一致")
        return record

    def bind(self, task_id, finding_node_id, evidence_id, note):
        finding = self.repo.get_node(task_id, finding_node_id)
        if finding["kind"] != "finding":
            raise ValueError("finding_node_id 必须是漏洞图节点 ID")
        if not note.strip():
            raise ValueError(
                "必须写明该证据证明了什么；相近域名或时间不能替代归属核实"
            )
        confirmed = self._evidence(task_id, evidence_id)

        target = finding["payload"].get("target")
        if not target and finding["payload"].get("intent_id"):
            intent = self.repo.get_node(
                task_id, finding["payload"]["intent_id"]
            )
            target = intent["payload"].get("target")
        if target and normalize(
            "endpoint", target
        ) != normalize(
            "endpoint", confirmed["value"]["target"]
        ):
            raise ValueError("证据目标与漏洞目标不一致")

        for _ in range(8):
            old = self.repo.get_record(
                "finding_binding", finding_node_id, task_id
            )
            value = dict(old["value"]) if old else {"refs": []}
            value["refs"] = [
                ref
                for ref in value["refs"]
                if ref["evidence_id"] != evidence_id
            ]
            value["refs"].append(
                {
                    "evidence_id": evidence_id,
                    "note": note,
                    "confirmed_version": confirmed["version"],
                }
            )
            if self.repo.put_record(
                "finding_binding",
                finding_node_id,
                value,
                task_id,
                expected_version=(
                    old["version"] if old else 0
                ),
            ):
                return value
        raise ValueError("证据绑定被频繁修改，请重试")

    def build(self, task_id):
        task = self.repo.get_task(task_id)
        lines = [
            "# 任务报告",
            "",
            "目标：" + task["goal"],
            "状态：" + task["status"],
            "",
        ]
        versions, finding_versions, binding_versions = {}, {}, {}
        findings = self.repo.list_nodes(
            task_id, kind="finding"
        )
        if not findings:
            lines += [
                "当前没有已确认漏洞；这不证明所有目标都安全。",
                "",
            ]

        for finding in findings:
            finding_versions[finding["id"]] = finding["version"]
            binding = self.repo.get_record(
                "finding_binding", finding["id"], task_id
            )
            refs = binding["value"]["refs"] if binding else []
            binding_versions[finding["id"]] = (
                binding["version"] if binding else 0
            )
            # 兼容 Worker 的结构化 evidence_id，不靠时间/域名猜关联。
            if (
                not refs
                and finding["payload"].get("evidence_id")
            ):
                refs = [
                    {
                        "evidence_id": finding["payload"][
                            "evidence_id"
                        ],
                        "note": "Worker 保留的同一意图证据引用",
                    }
                ]
            if not refs:
                raise ValueError(
                    "已确认漏洞缺少证据引用，拒绝生成看似完整的报告"
                )

            lines += [
                "## " + finding["text"],
                "漏洞记录 ID："
                + str(
                    finding["payload"].get(
                        "finding_record_id", "未分配"
                    )
                ),
                "图节点 ID：" + finding["id"],
            ]
            for ref in refs:
                record = self._evidence(
                    task_id, ref["evidence_id"]
                )
                value = record["value"]
                if (
                    binding
                    and ref.get("confirmed_version")
                    != record["version"]
                ):
                    raise ValueError(
                        "绑定后证据已变化，必须重新核对并绑定，不能沿用旧用途说明"
                    )
                if (
                    not binding
                    and record["version"] != 1
                ):
                    raise ValueError(
                        "Worker 初始证据已变化，必须人工核对绑定当前版本"
                    )
                if (
                    not binding
                    and value.get("intent_id")
                    != finding["payload"].get("intent_id")
                ):
                    raise ValueError(
                        "Worker 引用的证据与漏洞意图不一致，需人工核对绑定"
                    )
                versions[ref["evidence_id"]] = record["version"]
                lines += [
                    "- 证据："
                    + ref["evidence_id"]
                    + "，版本 "
                    + str(record["version"]),
                    "- 用途：" + ref["note"],
                    "- 请求：" + value["target"],
                    "- 状态：" + str(value["status"]),
                    "- 正文摘要："
                    + value["body"][:300].replace(
                        "\n", " "
                    ),
                ]
            lines.append("")

        report = {
            "markdown": "\n".join(lines),
            "evidence_versions": versions,
            "finding_versions": finding_versions,
            "binding_versions": binding_versions,
            "created_at": time.time(),
        }
        if self.is_stale(task_id, report):
            raise ValueError(
                "报告生成时证据发生变化，请重新生成"
            )
        old = self.repo.get_record(
            "report", "latest", task_id
        )
        if not self.repo.put_record(
            "report",
            "latest",
            report,
            task_id,
            expected_version=old["version"] if old else 0,
        ):
            raise ValueError("其他报告已先提交，请重新读取")
        return report

    def is_stale(self, task_id, report=None):
        if report is None:
            record = self.repo.get_record(
                "report", "latest", task_id
            )
            if not record:
                return True
            report = record["value"]

        current_findings = {
            node["id"]: node["version"]
            for node in self.repo.list_nodes(
                task_id, kind="finding"
            )
        }
        if current_findings != report["finding_versions"]:
            return True
        for evidence_id, version in report[
            "evidence_versions"
        ].items():
            record = self.repo.get_record(
                "evidence", evidence_id, task_id
            )
            if (
                record is None
                or record["version"] != version
            ):
                return True
        for finding_id, version in report[
            "binding_versions"
        ].items():
            record = self.repo.get_record(
                "finding_binding", finding_id, task_id
            )
            if (
                record["version"] if record else 0
            ) != version:
                return True
        return False

    async def retest(self, task_id, finding_node_id, probe):
        finding = self.repo.get_node(task_id, finding_node_id)
        if finding["kind"] != "finding":
            raise ValueError("复测对象必须是漏洞图节点")

        target = finding["payload"].get("target")
        if not target and finding["payload"].get("intent_id"):
            intent = self.repo.get_node(
                task_id, finding["payload"]["intent_id"]
            )
            target = intent["payload"].get("target")
        allowed = {
            normalize("endpoint", item)
            for item in self.repo.get_task(task_id)["scope"]
            if item.startswith(("http://", "https://"))
        }
        if not target or normalize(
            "endpoint", target
        ) not in allowed:
            raise ValueError("漏洞未保留范围内的明确目标")

        observation = await probe(target)
        if not isinstance(observation.get("issue"), bool):
            raise ValueError(
                "复测探针必须返回明确的 issue 布尔结论"
            )
        evidence = record_exchange(
            self.repo,
            task_id,
            target,
            observation["status"],
            observation["body"],
            intent_id=finding["payload"].get("intent_id", ""),
        )
        result = {
            "id": "retest-" + uuid4().hex,
            "finding_node_id": finding_node_id,
            "status": (
                "reproduced"
                if observation["issue"]
                else "not_reproduced"
            ),
            "evidence_id": evidence["evidence_id"],
            "created_at": time.time(),
        }
        self.repo.put_record(
            "retest", result["id"], result, task_id
        )
        return result
