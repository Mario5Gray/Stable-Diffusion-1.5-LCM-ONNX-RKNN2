// src/components/options/ComfyOptions.jsx
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Slider } from '@/components/ui/slider';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Sparkles, Heart, Pause, Play } from 'lucide-react';
import { Input } from '@/components/ui/input';
import {CSS_CLASSES} from '../../utils/constants';
import { createComfyInvokerApi } from "@/lib/comfyInvokerApi";
import { useComfyJob } from "@/hooks/useComfyJob";
import React, { useMemo, useEffect, useRef, useState, useCallback } from "react";
import { urlToFile } from "@/utils/imageToFile";

export function ComfyOptions({
  inputImage,
  apiBase = "https://node2:4205",
  workflowId = "TRACKING-LCM-DIFFS",
  defaultCfg = 1.0,
  defaultSteps = 16,
  defaultDenoise = 0.03,
  onStart,
  onDone,
  onError,
}) {

  const api = useMemo(() => createComfyInvokerApi(apiBase), [apiBase]);
  const comfy = useComfyJob({ api });

  const [cfg, setCfg] = useState(defaultCfg);
  const [steps, setSteps] = useState(defaultSteps);
  const [denoise, setDenoise] = useState(defaultDenoise);
  const [file, setFile] = useState(null);

  const run = useCallback(async () => {
    onStart?.();

    console.log("Lets getComfy!! inputImage=", inputImage);
console.log("kind=", inputImage?.kind, "key=", inputImage?.key, "url=", inputImage?.url);

    let inputImageFile = null;    

    if (inputImage?.kind === "file") {
      inputImageFile = inputImage.file;
    } else if (inputImage?.kind === "url") {
      const res = await fetch(inputImage.url);
      const blob = await res.blob();
      inputImageFile = new File(
        [blob],
        inputImage.filename || "input.png",
        { type: blob.type || "image/png" }
      );
    } else {
      inputImageFile = file; // fallback to manual upload input
    }

    return comfy.start({
      workflowId,
      params: { cfg, steps, denoise },
      inputImageFile,
    });
  }, [onStart, comfy, workflowId, cfg, steps, denoise, inputImage, file]);

  // If you want a "done" callback when outputs arrive:
  // (cheap + safe: just watch comfy.state)
  React.useEffect(() => {
    if (comfy.state === "done") {
      onDone?.(comfy.job);
    }
  }, [comfy.state, comfy.job, onDone]);

  const lastKeyRef = useRef(null);

  useEffect(() => {
    let cancelled = false;

    async function sync() {
      const key = inputImage?.key ?? null;
  
      // if selection changed, clear stale file immediately
      if (key !== lastKeyRef.current) {
        lastKeyRef.current = key;
        setFile(null);
      }

      if (!inputImage) return;

      if (inputImage.kind === "file") {
        setFile(inputImage.file);
        return;
      }

      if (inputImage.kind === "url") {
        const f = await urlToFile(inputImage.url, inputImage.filename || "input.png");
        if (!cancelled) setFile(f);
      }
    }

    sync();
    return () => { cancelled = true; };
  }, [inputImage]);

  console.log("[OptionsPanel] inputImage=", inputImage);
  return (
    <div className="space-y-3 rounded-2xl border p-4 bg-gradient-to-br from-purple-50/50 to-pink-50/50 dark:from-purple-950/20 dark:to-pink-950/20">
      <Label className="text-base font-semibold">Send to ComfyUI Workflow</Label>
      <div className="flex items-center gap-2">
        <Badge variant={comfy.state === "running" ? "default" : "secondary"}>
          {comfy.state}
        </Badge>
      </div>

      <div className="space-y-1">
      <label>CFG</label>
        <input
          type="number"
          value={cfg}
          step="0.05"
          onChange={(e) => setCfg(parseFloat(e.target.value))}
        />
      </div>

      <div className="space-y-1">
      <label>Steps</label>
        <input
          type="number"
          value={steps}
          step="1"
          onChange={(e) => setSteps(parseInt(e.target.value, 10))}
        />
      </div>

      <label>
        Denoise
        <input
          type="number"
          value={denoise}
          step="0.05"
          onChange={(e) => setDenoise(parseFloat(e.target.value))}
        />
      </label>

      <div className="space-y-1">
          Use selected chat image
              <Input
                type="file"
                accept="image/*"
                className={CSS_CLASSES.INPUT}
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              />
      {comfy.error ? (
        <pre style={{ whiteSpace: "pre-wrap" }}>
          {String(comfy.error.message || comfy.error)}
        </pre>
       ) : null}
      </div>

      <div style={{ display: "flex", gap: 8 }}>
        <Button
          onClick={run}
          disabled={comfy.isBusy}
          className="
            relative overflow-hidden
            border border-purple-400/40
            bg-gradient-to-br from-purple-500/90 to-pink-500/90
            text-white
            shadow-md
            hover:from-purple-500 hover:to-pink-500
            active:scale-[0.98]
            transition-all
          "
        >
          <Sparkles className="mr-2 h-4 w-4" />
          Run
        </Button>

        <Button
          variant="outline"
          onClick={comfy.cancel}
          disabled={!comfy.isBusy}
          className="
            border-red-400/40
            text-red-500
            hover:bg-red-500/10
          "
        >
          <Pause className="mr-2 h-4 w-4" />
          Stop
        </Button>        
      </div>


 {comfy.job?.outputs?.length ? (
        <div style={{ display: "grid", gap: 8 }}>
          {comfy.job.outputs.map((o) => (
            <img
              key={o.url ?? `${o.filename}-${o.subfolder ?? ""}`}
              src={o.url}
              alt=""
              style={{ maxWidth: "100%" }}
            />
          ))}
        </div>
      ) : null}
    </div>
  );
}