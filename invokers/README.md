# Python ComfyUI invoker 

This override scheme assumes you’re using API prompt JSON (node-id dict).

## Example Usage

```python
from invokers.comfyui import ComfyUIInvoker, load_workflow_json

inv = ComfyUIInvoker("http://node2.lan:8188", timeout_s=120)

workflow = load_workflow_json("my_workflow_api.json")

overrides = {
    "7.inputs.text": "a neon-lit alleyway, rain, cinematic",
    "3.inputs.seed": 123456,
    "3.inputs.steps": 20,
    "3.inputs.cfg": 6.5,
}

result = inv.invoke(workflow, overrides=overrides, max_wait_s=900)

print("prompt_id:", result.prompt_id)
print("outputs:", result.outputs)

# Download first output:
if result.outputs:
    img_bytes = inv.download_file(result.outputs[0])
    open("out.png", "wb").write(img_bytes)
```