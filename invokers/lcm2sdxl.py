from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import requests

Json = Dict[str, Any]


class ComfyUIError(RuntimeError):
    pass


@dataclass(frozen=True)
class ComfyFileRef:
    filename: str
    subfolder: str = ""
    type: str = "output"  # output|input|temp

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
    Thin client around ComfyUI's HTTP API.
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

    def invoke(
        self,
        prompt_graph: Json,
        client_id: Optional[str] = None,
        poll_interval_s: float = 0.5,
        max_wait_s: float = 900.0,
    ) -> ComfyInvokeResult:
        prompt_id = self.queue_prompt(prompt_graph, client_id=client_id)
        history = self.wait_for_history(prompt_id, poll_interval_s=poll_interval_s, max_wait_s=max_wait_s)
        outputs = self.extract_outputs(history, prompt_id)
        return ComfyInvokeResult(prompt_id=prompt_id, history=history, outputs=outputs)

    def queue_prompt(self, prompt_graph: Json, client_id: Optional[str] = None) -> str:
        payload = {"prompt": prompt_graph, "client_id": client_id or f"invokers-{uuid.uuid4()}"}
        r = self._post("/prompt", json_body=payload)
        data = self._json_or_raise(r, "queue_prompt")
        pid = data.get("prompt_id")
        if not pid:
            raise ComfyUIError(f"/prompt missing prompt_id: {data}")
        return str(pid)

    def get_history(self, prompt_id: str) -> Json:
        r = self._get(f"/history/{prompt_id}")
        return self._json_or_raise(r, "get_history")

    def wait_for_history(self, prompt_id: str, poll_interval_s: float, max_wait_s: float) -> Json:
        deadline = time.time() + max_wait_s
        last = None
        while time.time() < deadline:
            hist = self.get_history(prompt_id)
            last = hist
            if isinstance(hist, dict) and prompt_id in hist:
                node_graph = hist[prompt_id]
                self._raise_if_history_error(node_graph)
                if self._history_has_outputs(node_graph):
                    return hist
            time.sleep(poll_interval_s)
        raise ComfyUIError(f"Timed out waiting for {prompt_id}. Last={last}")

    def download_file(self, ref: ComfyFileRef) -> bytes:
        r = self._get("/view", params=ref.view_params(), stream=True)
        if r.status_code != 200:
            raise ComfyUIError(f"download_file HTTP {r.status_code}: {r.text[:500]}")
        return r.content

    def extract_outputs(self, history: Json, prompt_id: str) -> List[ComfyFileRef]:
        if prompt_id not in history:
            return []
        node_graph = history[prompt_id]
        self._raise_if_history_error(node_graph)

        out_map = node_graph.get("outputs", {})
        if not isinstance(out_map, dict):
            return []

        refs: List[ComfyFileRef] = []
        for node_out in out_map.values():
            if not isinstance(node_out, dict):
                continue
            for key in ("images", "gifs", "audio", "files"):
                vals = node_out.get(key)
                if not isinstance(vals, list):
                    continue
                for item in vals:
                    if not isinstance(item, dict):
                        continue
                    fn = item.get("filename")
                    if not fn:
                        continue
                    refs.append(
                        ComfyFileRef(
                            filename=str(fn),
                            subfolder=str(item.get("subfolder") or ""),
                            type=str(item.get("type") or "output"),
                        )
                    )

        # de-dupe
        uniq: Dict[Tuple[str, str, str], ComfyFileRef] = {}
        for r in refs:
            uniq[(r.filename, r.subfolder, r.type)] = r
        return list(uniq.values())

    # ----- HTTP helpers -----

    def _get(self, path: str, params: Optional[Dict[str, str]] = None, stream: bool = False) -> requests.Response:
        return self.session.get(
            self.base_url + path,
            params=params,
            timeout=self.timeout_s,
            headers=self.headers,
            verify=self.verify_tls,
            stream=stream,
        )

    def _post(
        self,
        path: str,
        json_body: Optional[Json] = None,
        data_body: Optional[Dict[str, str]] = None,
        files: Optional[Dict[str, Any]] = None,
    ) -> requests.Response:
        return self.session.post(
            self.base_url + path,
            json=json_body,
            data=data_body,
            files=files,
            timeout=self.timeout_s,
            headers=self.headers,
            verify=self.verify_tls,
        )

    def _json_or_raise(self, r: requests.Response, where: str) -> Json:
        if not (200 <= r.status_code < 300):
            raise ComfyUIError(f"{where} HTTP {r.status_code}: {r.text[:1000]}")
        try:
            return r.json()
        except Exception as e:
            raise ComfyUIError(f"{where} JSON decode error: {e}; body={r.text[:1000]}")

    def _history_has_outputs(self, node_graph: Any) -> bool:
        if not isinstance(node_graph, dict):
            return False
        outs = node_graph.get("outputs")
        if not isinstance(outs, dict) or not outs:
            return False
        for v in outs.values():
            if not isinstance(v, dict):
                continue
            for k in ("images", "gifs", "audio", "files"):
                if isinstance(v.get(k), list) and len(v.get(k)) > 0:
                    return True
        return False

    def _raise_if_history_error(self, node_graph: Any) -> None:
        if not isinstance(node_graph, dict):
            return
        status = node_graph.get("status")
        if isinstance(status, dict):
            if status.get("status_str") == "error" or status.get("error") or status.get("exception_message"):
                raise ComfyUIError(f"ComfyUI error: {status}")
        if node_graph.get("error"):
            raise ComfyUIError(f"ComfyUI error: {node_graph.get('error')}")


# -----------------------------------------------------------------------------
# LCM -> SDXL workflow adapter for YOUR graph (IMG2IMG-5.json)
# -----------------------------------------------------------------------------

class LCMtoSDXLWorkflow:
    """
    Convenience wrapper around your specific workflow JSON.

    Based on your uploaded graph:
      - KSampler: node 50
      - Seed generator: node 29
      - Input resize: node 48
      - Output resize: node 45
      - VAE encode: node 47
      - VAE decode: node 52
      - LoRAs: nodes 14, 15, 55
      - CLIP last layer: node 13
    (IDs from the uploaded file.)
    """
    def __init__(self, workflow_ui_json: Json) -> None:
        self.ui = json.loads(json.dumps(workflow_ui_json))  # deep copy

    @staticmethod
    def load(path: str) -> "LCMtoSDXLWorkflow":
        with open(path, "r", encoding="utf-8") as f:
            return LCMtoSDXLWorkflow(json.load(f))

    # --- Convert UI workflow (nodes/links) into API prompt graph dict ---
    # ComfyUI /prompt expects the "prompt graph" format: {node_id: {class_type, inputs}}
    # Your file is UI format: {"nodes":[...], "links":[...]}
    def to_prompt_graph(self) -> Json:
        nodes = self.ui.get("nodes", [])
        links = self.ui.get("links", [])

        # Build link_id -> (from_node_id, from_slot, to_node_id, to_slot, type)
        link_map: Dict[int, Tuple[int, int, int, int, str]] = {}
        for l in links:
            # format: [link_id, from_node, from_slot, to_node, to_slot, "TYPE"]
            if isinstance(l, list) and len(l) >= 6:
                link_map[int(l[0])] = (int(l[1]), int(l[2]), int(l[3]), int(l[4]), str(l[5]))

        # For each node, build inputs dict where linked inputs become [from_node_id, output_index]
        prompt: Json = {}
        for n in nodes:
            node_id = str(n["id"])
            class_type = n["type"]
            inputs: Dict[str, Any] = {}

            # Start with widget_values if present (these are positional; not reliable without schema)
            # We primarily wire links; widget tuning is done via setters below that update widgets_values.
            # For API prompt graphs, many nodes accept direct scalar inputs (seed, steps, cfg, denoise, etc).
            # We'll fill those explicitly for known nodes via setters that also keep UI widgets consistent.

            # Wire linked inputs
            for inp in n.get("inputs", []) or []:
                name = inp.get("name")
                link_id = inp.get("link")
                if name and link_id is not None:
                    lm = link_map.get(int(link_id))
                    if lm:
                        from_node, from_slot, _to_node, _to_slot, _ty = lm
                        inputs[name] = [str(from_node), int(from_slot)]

            prompt[node_id] = {"class_type": class_type, "inputs": inputs}

        # Now, for nodes where scalar widget values matter (KSampler), populate them from widgets_values.
        # This keeps the prompt runnable even without additional overrides.
        self._apply_known_widget_scalars(prompt)

        return prompt

    def _apply_known_widget_scalars(self, prompt: Json) -> None:
        # KSampler node 50 widgets_values:
        # [seed, "fixed", steps, cfg, sampler_name, scheduler, denoise]
        n50 = self._get_node(50)
        if n50:
            w = n50.get("widgets_values") or []
            if len(w) >= 7:
                prompt["50"]["inputs"]["seed"] = int(w[0]) if isinstance(w[0], (int, float)) else w[0]
                prompt["50"]["inputs"]["steps"] = int(w[2])
                prompt["50"]["inputs"]["cfg"] = float(w[3])
                prompt["50"]["inputs"]["sampler_name"] = str(w[4])
                prompt["50"]["inputs"]["scheduler"] = str(w[5])
                prompt["50"]["inputs"]["denoise"] = float(w[6])

        # CLIPSetLastLayer node 13 widgets_values: [-2]
        n13 = self._get_node(13)
        if n13:
            w = n13.get("widgets_values") or []
            if len(w) >= 1:
                prompt["13"]["inputs"]["stop_at_clip_layer"] = int(w[0])

        # VAELoader node 54 widgets_values: ["ae.safetensors"]
        n54 = self._get_node(54)
        if n54:
            w = n54.get("widgets_values") or []
            if len(w) >= 1:
                prompt["54"]["inputs"]["vae_name"] = str(w[0])

        # CheckpointLoaderSimple node 4 widgets_values: ["icbinpXL_v6.safetensors"]
        n4 = self._get_node(4)
        if n4:
            w = n4.get("widgets_values") or []
            if len(w) >= 1:
                prompt["4"]["inputs"]["ckpt_name"] = str(w[0])

    # --- High-level knobs (no prompt text required) ---

    def set_steps(self, steps: int) -> "LCMtoSDXLWorkflow":
        n = self._get_node_or_raise(50)
        w = n.setdefault("widgets_values", [])
        self._ensure_len(w, 7)
        w[2] = int(steps)
        return self

    def set_cfg(self, cfg: float) -> "LCMtoSDXLWorkflow":
        n = self._get_node_or_raise(50)
        w = n.setdefault("widgets_values", [])
        self._ensure_len(w, 7)
        w[3] = float(cfg)
        return self

    def set_sampler(self, sampler_name: str) -> "LCMtoSDXLWorkflow":
        n = self._get_node_or_raise(50)
        w = n.setdefault("widgets_values", [])
        self._ensure_len(w, 7)
        w[4] = str(sampler_name)
        return self

    def set_scheduler(self, scheduler: str) -> "LCMtoSDXLWorkflow":
        n = self._get_node_or_raise(50)
        w = n.setdefault("widgets_values", [])
        self._ensure_len(w, 7)
        w[5] = str(scheduler)
        return self

    def set_denoise(self, denoise: float) -> "LCMtoSDXLWorkflow":
        n = self._get_node_or_raise(50)
        w = n.setdefault("widgets_values", [])
        self._ensure_len(w, 7)
        w[6] = float(denoise)
        return self

    def set_seed(self, seed: int) -> "LCMtoSDXLWorkflow":
        # Your KSampler seed input is wired from Seed Generator node 29 via link,
        # but node 50 still has a seed widget value too; we set BOTH.
        n = self._get_node_or_raise(50)
        w = n.setdefault("widgets_values", [])
        self._ensure_len(w, 7)
        w[0] = int(seed)

        n29 = self._get_node(29)
        if n29:
            w29 = n29.setdefault("widgets_values", [])
            self._ensure_len(w29, 3)
            w29[0] = int(seed)
        return self

    def set_input_resize(self, width: int, height: int, mode: str = "crop") -> "LCMtoSDXLWorkflow":
        # Node 48 ImageResizeKJv2 widgets_values:
        # [w, h, interp, mode, color, align, ???, device, htmlinfo]
        n = self._get_node_or_raise(48)
        w = n.setdefault("widgets_values", [])
        self._ensure_len(w, 9)
        w[0] = int(width)
        w[1] = int(height)
        w[3] = str(mode)
        return self

    def set_output_resize(self, width: int, height: int, mode: str = "pad") -> "LCMtoSDXLWorkflow":
        n = self._get_node_or_raise(45)
        w = n.setdefault("widgets_values", [])
        self._ensure_len(w, 9)
        w[0] = int(width)
        w[1] = int(height)
        w[3] = str(mode)
        return self

    def set_clip_last_layer(self, layer: int) -> "LCMtoSDXLWorkflow":
        n = self._get_node_or_raise(13)
        w = n.setdefault("widgets_values", [])
        self._ensure_len(w, 1)
        w[0] = int(layer)
        return self

    def set_lora_strength(self, node_id: int, strength_model: float, strength_clip: Optional[float] = None) -> "LCMtoSDXLWorkflow":
        """
        Works for your LoRA nodes 14/15 (pysssss) and 55 (core LoraLoader).
        For pysssss nodes, widgets look like: [name, sm, sc, "", "[none]"]
        For core LoraLoader, widgets: [name, sm, sc]
        """
        n = self._get_node_or_raise(node_id)
        w = n.setdefault("widgets_values", [])
        if len(w) >= 3:
            w[1] = float(strength_model)
            w[2] = float(strength_clip if strength_clip is not None else strength_model)
        else:
            raise ComfyUIError(f"LoRA node {node_id} unexpected widgets_values: {w}")
        return self

    # --- internals ---

    def _get_node(self, node_id: int) -> Optional[Json]:
        for n in self.ui.get("nodes", []) or []:
            if int(n.get("id")) == int(node_id):
                return n
        return None

    def _get_node_or_raise(self, node_id: int) -> Json:
        n = self._get_node(node_id)
        if not n:
            raise ComfyUIError(f"Node {node_id} not found in workflow")
        return n

    def _ensure_len(self, arr: List[Any], n: int) -> None:
        while len(arr) < n:
            arr.append(None)


def load_ui_workflow(path: str) -> Json:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)