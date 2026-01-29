// src/hooks/useComfyJob.js
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

export function useComfyJob({ api, pollMs = 750, autoPoll = true } = {}) {
  const abortRef = useRef(null);
  const pollTimerRef = useRef(null);

  const [jobId, setJobId] = useState(null);
  const [state, setState] = useState("idle"); // idle|starting|running|done|error|canceled
  const [job, setJob] = useState(null);       // latest job payload
  const [error, setError] = useState(null);
  const lastFingerprintRef = useRef(null);
  const lastChangeAtRef = useRef(0);
  const startedAtRef = useRef(0);

  const hardTimeoutMs = 6 * 60 * 1000;   // e.g. 6 minutes
  const staleTimeoutMs = 120 * 1000;      // e.g. 30s with no changes

  const clearPoll = useCallback(() => {
    if (pollTimerRef.current) {
      clearTimeout(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  }, []);

  const cancel = useCallback(async () => {
    clearPoll();
    if (abortRef.current) abortRef.current.abort();
    abortRef.current = null;

    // Optional: tell backend to cancel server-side too
    if (jobId && api?.cancelJob) {
      try {
        await api.cancelJob(jobId);
      } catch {
        // ignore cancel errors
      }
    }
    setState("canceled");
  }, [api, clearPoll, jobId]);

  const start = useCallback(
    async (payload) => {
      // cancel any in-flight run
      await cancel();
      setError(null);
      setJob(null);

      const ac = new AbortController();
      abortRef.current = ac;

      try {
        const started = await api.startJob(payload, { signal: ac.signal });                
        const newJobId = started.jobId ?? started.job_id ?? started.id;
        if (!newJobId) throw new Error(`startJob: missing job id in response: ${JSON.stringify(started)}`);
        setJobId(newJobId);        
        setState("running");
        setJob(null);
        
        // Timeout
        startedAtRef.current = Date.now();
        lastChangeAtRef.current = Date.now();
        lastFingerprintRef.current = null;

        return started;
      } catch (e) {
        if (e?.name === "AbortError") return;
        setError(e);
        setState("error");
        throw e;
      }
    },
    [api, cancel]
  );

  const refresh = useCallback(async () => {
    if (!jobId || !api) return;
    const ac = abortRef.current ?? new AbortController();
    abortRef.current = ac;

    try {
      const latest = await api.getJob(jobId, { signal: ac.signal });
      setJob(latest);

      // Timeout 
      const fp = JSON.stringify({
        status: latest?.status,
        // progress fields you care about
        frac: latest?.progress?.fraction,
        node: latest?.progress?.current_node,
        seen: latest?.progress?.nodes_seen,
        out: latest?.outputs?.length,
      });      

      const now = Date.now();
      if (fp !== lastFingerprintRef.current) {
        lastFingerprintRef.current = fp;
        lastChangeAtRef.current = now;
      }

      const startedAt = startedAtRef.current || now;
      const lastChangeAt = lastChangeAtRef.current || now;

      if (now - startedAt > hardTimeoutMs) {
        clearPoll();
        setState("error");
        setError(new Error("Generation timed out (hard timeout)."));
        // optional: cancel server-side
        if (api?.cancelJob && jobId) api.cancelJob(jobId).catch(() => {});
        return;
      }

      if (now - lastChangeAt > staleTimeoutMs && latest?.status === "running") {
        clearPoll();
        setState("error");
        setError(new Error("Generation stalled (no progress updates)."));
        if (api?.cancelJob && jobId) api.cancelJob(jobId).catch(() => {});
        return;
      }

      // //timeout
      if (latest?.status === "done") {
        clearPoll();
        setState("done");
      } else if (latest?.status === "error") {
        clearPoll();
        setState("error");
        setError(new Error(latest?.error || "Job failed"));
      } else if (latest?.status === "canceled") {
        clearPoll();
        setState("canceled");
      } else {
        setState("running");
      }
    } catch (e) {
      if (e?.name === "AbortError") return;
      clearPoll();
      setState("error");
      setError(e);
    }
  }, [api, clearPoll, jobId]);

  // Poll loop
  useEffect(() => {
    if (!autoPoll) return;
    if (!jobId) return;
    if (state !== "running") return;

    let alive = true;
    const tick = async () => {
      await refresh();
      if (!alive) return;
      // keep polling only if still running
      pollTimerRef.current = setTimeout(tick, pollMs);
    };

    pollTimerRef.current = setTimeout(tick, pollMs);

    return () => {
      alive = false;
      clearPoll();
    };
  }, [autoPoll, clearPoll, jobId, pollMs, refresh, state]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      clearPoll();
      if (abortRef.current) abortRef.current.abort();
    };
  }, [clearPoll]);

  const isBusy = state === "starting" || state === "running";

  return useMemo(
    () => ({
      jobId,
      state,
      isBusy,
      job,
      error,
      start,
      refresh,
      cancel,
      setJobId, // in case you want to resume an existing job
    }),
    [cancel, error, isBusy, job, jobId, refresh, start, state]
  );
}