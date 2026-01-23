from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import requests


Json = Dict[str, Any]


class ComfyUIError(RuntimeError):
    pass


@dataclass(frozen=True)
class ComfyFileRef:
    """Reference to a file that ComfyUI can serve via /view."""
    filename: str
    subfolder: str = ""
    type: str = "output"  # "output" | "input" | "temp"

    def view_params(self) -> Dict[str, str]:
        p = {"filename": self.filename, "type": self.type}
        if self.subfolder:
            p["subfolder"] = self.subfolder
        return p


@dataclass(frozen=True)
class ComfyInvokeResult:
    prompt_id: str
    history: Json
    outputs: List[ComfyFileRef]


class ComfyUIInvoker:
    """
    Minimal ComfyUI client.

    Requires:
      - ComfyUI reachable at base_url (e.g. http://node2.lan:8188)
      - ComfyUI API enabled (default in ComfyUI)

    Endpoints used:
      - POST /prompt
      - GET  /history/{prompt_id}
      - POST /upload/image
      - GET  /view
    """

    def __init__(
        self,
        base_url: str,
        timeout_s: float = 60.0,
        session: Optional[requests.Session] = None,
        verify_tls: Union[bool, str] = True,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.verify_tls = verify_tls
        self.session = session or requests.Session()
        self.headers = headers or {}

    # ---------------------------
    # Public API
    # ---------------------------

    def invoke(
        self,
        workflow: Json,
        overrides: Optional[Dict[str, Any]] = None,
        client_id: Optional[str] = None,
        poll_interval_s: float = 0.5,
        max_wait_s: float = 600.0,
    ) -> ComfyInvokeResult:
        """
        Queue a workflow and wait for results.

        `workflow` is the ComfyUI prompt graph JSON (what you export as workflow/api JSON).
        `overrides` can be applied via apply_overrides() (see below).
        """
        wf = workflow
        if overrides:
            wf = self.apply_overrides(workflow, overrides)

        prompt_id = self.queue_prompt(wf, client_id=client_id)
        history = self.wait_for_history(prompt_id, poll_interval_s=poll_interval_s, max_wait_s=max_wait_s)
        outputs = self.extract_outputs(history, prompt_id)
        return ComfyInvokeResult(prompt_id=prompt_id, history=history, outputs=outputs)

    def queue_prompt(self, workflow: Json, client_id: Optional[str] = None) -> str:
        """
        POST /prompt with a prompt graph.
        Returns ComfyUI prompt_id.
        """
        payload = {
            "prompt": workflow,
            "client_id": client_id or self._default_client_id(),
        }
        r = self._post("/prompt", json_body=payload)
        data = self._json_or_raise(r, "queue_prompt")
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            raise ComfyUIError(f"ComfyUI /prompt did not return prompt_id: {data}")
        return str(prompt_id)

    def get_history(self, prompt_id: str) -> Json:
        """
        GET /history/{prompt_id}
        """
        r = self._get(f"/history/{prompt_id}")
        return self._json_or_raise(r, "get_history")

    def wait_for_history(
        self,
        prompt_id: str,
        poll_interval_s: float = 0.5,
        max_wait_s: float = 600.0,
    ) -> Json:
        """
        Poll history until outputs appear or timeout.
        """
        deadline = time.time() + max_wait_s
        last = None

        while time.time() < deadline:
            hist = self.get_history(prompt_id)
            last = hist

            # ComfyUI returns { "<prompt_id>": { ... } } when completed/known
            if isinstance(hist, dict) and prompt_id in hist:
                # If outputs exist, we're basically done
                node_graph = hist[prompt_id]
                if self._history_has_outputs(node_graph):
                    return hist

                # If it contains an "status" error, fail early
                self._raise_if_history_error(node_graph)

            time.sleep(poll_interval_s)

        raise ComfyUIError(f"Timed out waiting for prompt_id={prompt_id}. Last history={last}")

    def upload_image(
        self,
        image_bytes: bytes,
        filename: str,
        subfolder: str = "",
        overwrite: bool = True,
        image_type: str = "input",
    ) -> Json:
        """
        POST /upload/image
        Returns ComfyUI response JSON (contains saved name/subfolder/type).
        """
        files = {"image": (filename, image_bytes, "application/octet-stream")}
        data = {
            "overwrite": "true" if overwrite else "false",
            "type": image_type,
        }
        if subfolder:
            data["subfolder"] = subfolder

        r = self._post("/upload/image", data_body=data, files=files)
        return self._json_or_raise(r, "upload_image")

    def download_file(self, ref: ComfyFileRef) -> bytes:
        """
        GET /view?filename=...&type=...&subfolder=...
        Returns raw bytes.
        """
        r = self._get("/view", params=ref.view_params(), stream=True)
        if r.status_code != 200:
            raise ComfyUIError(f"download_file failed HTTP {r.status_code}: {r.text[:500]}")
        return r.content

    # ---------------------------
    # Workflow override helpers
    # ---------------------------

    def apply_overrides(self, workflow: Json, overrides: Dict[str, Any]) -> Json:
        """
        Apply overrides to a workflow graph.

        This expects `workflow` to be the *API prompt graph* format (node-id keyed dict),
        which looks like:
            {
              "3": {"class_type":"KSampler","inputs":{"seed":..., "steps":...}},
              "7": {"class_type":"CLIPTextEncode","inputs":{"text":"..."}}
            }

        Overrides format (simple, explicit):
            {
              "<node_id>.inputs.<field>": value,
              "<node_id>.inputs.<nested>.<field>": value,
            }

        Example:
            overrides = {
              "3.inputs.seed": 123,
              "3.inputs.steps": 20,
              "7.inputs.text": "a castle at night",
            }
        """
        # Deep copy via json roundtrip (safe for pure JSON objects)
        wf = json.loads(json.dumps(workflow))

        for path, value in overrides.items():
            parts = path.split(".")
            if not parts:
                continue
            node_id = parts[0]
            if node_id not in wf:
                raise ComfyUIError(f"Override node_id={node_id} not found in workflow keys: {list(wf.keys())[:10]}...")

            cur: Any = wf[node_id]
            for key in parts[1:-1]:
                if isinstance(cur, dict) and key in cur:
                    cur = cur[key]
                else:
                    # Create missing dicts if needed (common for inputs.*)
                    if isinstance(cur, dict):
                        cur[key] = {}
                        cur = cur[key]
                    else:
                        raise ComfyUIError(f"Cannot create path segment '{key}' in override '{path}'")

            last_key = parts[-1]
            if not isinstance(cur, dict):
                raise ComfyUIError(f"Override target is not a dict at '{'.'.join(parts[:-1])}' for override '{path}'")
            cur[last_key] = value

        return wf

    # ---------------------------
    # Output extraction
    # ---------------------------

    def extract_outputs(self, history: Json, prompt_id: str) -> List[ComfyFileRef]:
        """
        Attempt to extract output file references from ComfyUI history response.
        Works for typical image/video/audio outputs.
        """
        if prompt_id not in history:
            return []

        node_graph = history[prompt_id]
        self._raise_if_history_error(node_graph)

        outputs: List[ComfyFileRef] = []
        # history[prompt_id]["outputs"] is a mapping: node_id -> {"images":[{...}], "gifs":[...], ...}
        out_map = node_graph.get("outputs", {})
        if not isinstance(out_map, dict):
            return outputs

        for _node_id, node_out in out_map.items():
            if not isinstance(node_out, dict):
                continue
            # Images
            outputs.extend(self._extract_file_refs_from_list(node_out.get("images"), default_type="output"))
            # Videos/gifs sometimes under "gifs"
            outputs.extend(self._extract_file_refs_from_list(node_out.get("gifs"), default_type="output"))
            # Some nodes return "audio"
            outputs.extend(self._extract_file_refs_from_list(node_out.get("audio"), default_type="output"))
            # Other payloads might include "files"
            outputs.extend(self._extract_file_refs_from_list(node_out.get("files"), default_type="output"))

        # De-dup (filename+subfolder+type)
        uniq: Dict[Tuple[str, str, str], ComfyFileRef] = {}
        for ref in outputs:
            uniq[(ref.filename, ref.subfolder, ref.type)] = ref
        return list(uniq.values())

    def _extract_file_refs_from_list(self, maybe_list: Any, default_type: str) -> List[ComfyFileRef]:
        refs: List[ComfyFileRef] = []
        if not isinstance(maybe_list, list):
            return refs
        for item in maybe_list:
            if not isinstance(item, dict):
                continue
            fn = item.get("filename")
            if not fn:
                continue
            refs.append(
                ComfyFileRef(
                    filename=str(fn),
                    subfolder=str(item.get("subfolder") or ""),
                    type=str(item.get("type") or default_type),
                )
            )
        return refs

    # ---------------------------
    # Internal HTTP helpers
    # ---------------------------

    def _default_client_id(self) -> str:
        return f"invokers-{uuid.uuid4()}"

    def _get(self, path: str, params: Optional[Dict[str, str]] = None, stream: bool = False) -> requests.Response:
        url = self.base_url + path
        r = self.session.get(
            url,
            params=params,
            timeout=self.timeout_s,
            headers=self.headers,
            verify=self.verify_tls,
            stream=stream,
        )
        return r

    def _post(
        self,
        path: str,
        json_body: Optional[Json] = None,
        data_body: Optional[Dict[str, str]] = None,
        files: Optional[Dict[str, Tuple[str, bytes, str]]] = None,
    ) -> requests.Response:
        url = self.base_url + path
        r = self.session.post(
            url,
            json=json_body,
            data=data_body,
            files=files,
            timeout=self.timeout_s,
            headers=self.headers,
            verify=self.verify_tls,
        )
        return r

    def _json_or_raise(self, r: requests.Response, where: str) -> Json:
        if r.status_code < 200 or r.status_code >= 300:
            raise ComfyUIError(f"{where} failed HTTP {r.status_code}: {r.text[:1000]}")
        try:
            return r.json()
        except Exception as e:
            raise ComfyUIError(f"{where} could not decode JSON: {e}; body={r.text[:1000]}")

    def _history_has_outputs(self, node_graph: Any) -> bool:
        if not isinstance(node_graph, dict):
            return False
        outs = node_graph.get("outputs")
        if not isinstance(outs, dict) or not outs:
            return False
        # Any node with any images/gifs/audio/files
        for v in outs.values():
            if not isinstance(v, dict):
                continue
            for k in ("images", "gifs", "audio", "files"):
                if isinstance(v.get(k), list) and len(v.get(k)) > 0:
                    return True
        return False

    def _raise_if_history_error(self, node_graph: Any) -> None:
        """
        ComfyUI sometimes includes error info in history.
        We try to fail early if present.
        """
        if not isinstance(node_graph, dict):
            return
        status = node_graph.get("status")
        # status can be dict with "status_str" or contain errors
        if isinstance(status, dict):
            if status.get("status_str") == "error" or status.get("error") or status.get("exception_message"):
                raise ComfyUIError(f"ComfyUI execution error: {status}")

        # Some builds include "error" top-level
        if node_graph.get("error"):
            raise ComfyUIError(f"ComfyUI execution error: {node_graph.get('error')}")


# ---------------------------
# Optional: tiny convenience for loading workflow JSON from disk
# ---------------------------

def load_workflow_json(path: str) -> Json:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)