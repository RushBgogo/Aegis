"""规划只生产意图，完成判断只依赖持久化事实。"""
import hashlib


class Planner:
    def __init__(self, repo):
        self.repo = repo

    async def plan(self, task_id):
        task = self.repo.get_task(task_id)
        if task["status"] != "running":
            return 0
        scope = task["scope"]
        if not scope:
            self.repo.append_event(
                task_id, "planner_stalled", {"reason": "empty explicit scope"}
            )
            self.repo.transition_task(task_id, "running", "failed")
            return 0

        goals = self.repo.list_nodes(task_id, "goal")
        if not goals:
            goals = [
                self.repo.create_node(
                    task_id, "goal", task["goal"], key="goal:assessment"
                )
            ]

        fact_by_target = {
            fact["payload"]["observation"]["target"]: fact
            for fact in self.repo.list_nodes(task_id, "fact")
            if "observation" in fact["payload"]
        }
        hinted_targets = {
            token
            for hint in self.repo.list_nodes(task_id, "hint")
            for token in hint["text"].split()
            if token in scope
        }
        existing_keys = {
            node["dedup_key"] for node in self.repo.list_nodes(task_id, "intent")
        }

        created = 0
        for target in scope:
            if target in fact_by_target:
                for goal in goals:
                    if target in goal["payload"].get("targets", scope):
                        self.repo.add_edge(
                            task_id, fact_by_target[target]["id"], goal["id"], "proves"
                        )
                continue

            key = "intent:" + hashlib.sha256(target.encode()).hexdigest()
            intent = self.repo.create_node(
                task_id,
                "intent",
                "Observe " + target,
                key=key,
                payload={"target": target, "goal_id": goals[0]["id"]},
                priority=5 if target in hinted_targets else 0,
            )
            if (
                target in hinted_targets
                and intent["state"] == "open"
                and intent["priority"] < 5
            ):
                self.repo.set_node_priority(task_id, intent["id"], 5)
            for goal in goals:
                if target in goal["payload"].get("targets", scope):
                    self.repo.add_edge(task_id, goal["id"], intent["id"], "spawns")
            created += key not in existing_keys
            existing_keys.add(key)

        active = [
            node
            for node in self.repo.list_nodes(task_id, "intent")
            if node["state"] in {"open", "running"}
        ]
        for goal in goals:
            targets = goal["payload"].get("targets", scope)
            if targets and all(target in fact_by_target for target in targets):
                self.repo.transition_node(task_id, goal["id"], "open", "met")

        if all(target in fact_by_target for target in scope) and not active:
            self.repo.append_event(
                task_id,
                "goal_resolved",
                {
                    "goal_ids": [goal["id"] for goal in goals],
                    "conclusion": (
                        "assessment completed; see evidence, "
                        "including negative observations"
                    ),
                },
            )
            self.repo.transition_task(task_id, "running", "done")
        elif not active:
            self.repo.append_event(
                task_id,
                "planner_stalled",
                {"reason": "no runnable intent and missing evidence"},
            )
            self.repo.transition_task(task_id, "running", "failed")
        return created


def sample_task(repo, scope=None):
    if scope is None:
        from artex.worker import DEFAULT_FIXTURES
        scope = DEFAULT_FIXTURES
    task = repo.create_task(
        "固定实验数据", "给出接口行为评估和来源", list(scope)
    )
    repo.transition_task(task["id"], "created", "running")
    return task
