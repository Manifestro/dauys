"""Private-only experiment logging for remote training runs."""

import base64
import json
import os
import urllib.request
from typing import Any


def private_entity(project: str) -> str:
    import wandb

    token = os.environ.get("WANDB_API_KEY")
    if not token:
        raise RuntimeError("W&B authentication is unavailable")
    entity = wandb.Api(timeout=30).default_entity
    if not entity:
        raise RuntimeError("W&B entity is unavailable")
    query = {
        "query": f'query {{ project(name: "{project}", entityName: "{entity}") {{ access }} }}'
    }
    authorization = base64.b64encode(("api:" + token).encode()).decode()
    endpoint = (
        os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai").rstrip("/")
        + "/graphql"
    )
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(query).encode(),
        headers={
            "Authorization": "Basic " + authorization,
            "Content-Type": "application/json",
        },
    )
    result = json.load(urllib.request.urlopen(request, timeout=30))
    if result.get("data", {}).get("project", {}).get("access") != "PRIVATE":
        raise RuntimeError("W&B project must be private before uploading")
    return entity


def start_private_run(project: str, name: str, config: dict[str, Any]):
    import wandb

    entity = private_entity(project)
    return wandb.init(
        entity=entity,
        project=project,
        name=name,
        group="hybrid-v0.2",
        config=config,
    )
