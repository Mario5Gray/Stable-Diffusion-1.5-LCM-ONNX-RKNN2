// src/lib/comfyInvokerApi.js
export function createComfyInvokerApi(baseUrl) {
  const u = (p) => `${baseUrl.replace(/\/$/, "")}${p}`;

  return {
    async startJob({ workflowId, params, inputImageFile }, { signal } = {}) {
      // If you upload image to *your* backend: use multipart
      const form = new FormData();
      form.append("workflowId", workflowId);
      form.append("params", JSON.stringify(params ?? {}));
      if (inputImageFile) form.append("image", inputImageFile);

      const res = await fetch(u("/v1/comfy/jobs"), {
        method: "POST",
        body: form,
        signal,
      });
      if (!res.ok) throw new Error(await safeText(res));
      return res.json(); // => { jobId, promptId? }
    },

    async getJob(jobId, { signal } = {}) {
      const res = await fetch(u(`/v1/comfy/jobs/${encodeURIComponent(jobId)}`), {
        method: "GET",
        signal,
      });
      if (!res.ok) throw new Error(await safeText(res));
      return res.json(); // => { status, progress?, outputs?, error? }
    },

    async cancelJob(jobId, { signal } = {}) {
      const res = await fetch(u(`/v1/comfy/jobs/${encodeURIComponent(jobId)}`), {
        method: "DELETE",
        signal,
      });
      if (!res.ok) throw new Error(await safeText(res));
      return res.json();
    },
  };
}

async function safeText(res) {
  try { return await res.text(); } catch { return `HTTP ${res.status}`; }
}