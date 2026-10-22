import json
from pathlib import Path

import httpx

from modelgraph.compiler import compile_graph
from modelgraph.contracts.loader import load_document
from modelgraph.contracts.manifest import ModelManifest
from modelgraph.providers.ollama import OllamaProvider
from modelgraph.runtime.engine import run_graph

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"


def test_engine_with_ollama_adapter_and_stub_server(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        payload = json.loads(request.content)
        prompt = "\n".join(message["content"] for message in payload["messages"])
        if "<routes>" in prompt:
            content = json.dumps(
                {
                    "decision": "route",
                    "to": "coding",
                    "confidence": 0.92,
                    "reason": "coding request",
                }
            )
        else:
            content = "const answer = 42;\n"
        return httpx.Response(
            200,
            json={
                "message": {"content": content},
                "prompt_eval_count": 10,
                "eval_count": 5,
                "done_reason": "stop",
            },
        )

    compiled = compile_graph(EXAMPLES / "hello-triage.graph.yaml", strict_models=True)
    assert compiled.graph
    client = httpx.Client(transport=httpx.MockTransport(handler))

    def factory(manifest: ModelManifest) -> OllamaProvider:
        return OllamaProvider(manifest, client=client)

    outcome = run_graph(
        compiled.graph,
        "Write a TypeScript function.",
        factory,
        runs_dir=tmp_path,
    )
    assert outcome.ok, outcome.error
    assert outcome.artifacts[0].content == "const answer = 42;\n"
    assert outcome.usage["input_tokens"] == 20
    assert outcome.usage["output_tokens"] == 10
    client.close()


def test_manifest_endpoint_is_used() -> None:
    manifest = ModelManifest.model_validate(load_document(EXAMPLES / "models" / "story-7b.yaml"))
    provider = OllamaProvider(manifest)
    assert provider.endpoint == "http://127.0.0.1:11434"
    provider.close()
