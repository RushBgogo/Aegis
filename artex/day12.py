"""离线验证 HTTP 层，无需启动浏览器或调用模型 API。"""
import argparse
import asyncio

import httpx

from artex.db import Repository
from artex.web import create_app


def build_parser():
    parser = argparse.ArgumentParser(description="Aegis Day12 HTTP 演示")
    parser.add_argument(
        "--scope",
        default="https://lab.example/public",
        help="练习目标 URL",
    )
    return parser


async def main():
    args = build_parser().parse_args()
    repo = Repository("sqlite:///:memory:")
    repo.setup()
    app = create_app(repo, api_token="offline-demo-token")
    try:
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://study",
            ) as client:
                client.headers["Authorization"] = (
                    "Bearer offline-demo-token"
                )
                response = await client.post(
                    "/api/tasks",
                    json={
                        "description": "HTTP 闭环",
                        "goal": "可复核结论",
                        "scope": [args.scope],
                    },
                )
                task = response.json()
                await client.post(
                    f"/api/tasks/{task['id']}/run", json={}
                )
                await asyncio.gather(*app.state.jobs.values())
                result = await client.get(
                    f"/api/tasks/{task['id']}"
                )
                print(
                    "HTTP",
                    response.status_code,
                    "任务状态",
                    result.json()["status"],
                )
    finally:
        repo.close()


if __name__ == "__main__":
    asyncio.run(main())
