"""Day 14 资产同步/补全演示：精确范围，标题不等于覆盖度。"""
import argparse
import asyncio

import httpx

from artex.assets import AssetService
from artex.db import Repository


def build_parser():
    parser = argparse.ArgumentParser(description="Aegis Day14 资产演示")
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
    try:
        task = repo.create_task(
            "资产同步", "评估样例", scope=[args.scope]
        )
        service = AssetService(repo)

        def handler(request):
            if request.url.path == "/sample-export":
                return httpx.Response(
                    200,
                    json={
                        "items": [
                            {
                                "kind": "endpoint",
                                "value": "https://LAB.example:443/public",
                            },
                            {
                                "kind": "endpoint",
                                "value": "https://lab.example/outside",
                            },
                        ]
                    },
                )
            return httpx.Response(
                200, text="<title>Offline Lab</title>"
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="https://source.example",
        ) as client:
            result = await service.sync(
                task["id"],
                client,
                "/sample-export",
                lambda body: body["items"],
            )
            print(
                "导入",
                len(result["imported"]),
                "拒绝",
                len(result["rejected"]),
            )
            print(await service.enrich(task["id"], client))

        print(service.coverage(task["id"]))
    finally:
        repo.close()


if __name__ == "__main__":
    asyncio.run(main())
