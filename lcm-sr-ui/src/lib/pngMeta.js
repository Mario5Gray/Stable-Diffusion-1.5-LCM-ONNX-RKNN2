// src/lib/pngMeta.js
export function extractPngTextChunks(arrayBuffer) {
  const dv = new DataView(arrayBuffer);
  const u8 = new Uint8Array(arrayBuffer);

  // PNG signature
  const sig = [137, 80, 78, 71, 13, 10, 26, 10];
  for (let i = 0; i < sig.length; i++) if (u8[i] !== sig[i]) return {};

  const readU32 = (o) => dv.getUint32(o, false);
  const readType = (o) => String.fromCharCode(u8[o], u8[o + 1], u8[o + 2], u8[o + 3]);

  let offset = 8;
  const out = {};

  while (offset + 12 <= u8.length) {
    const len = readU32(offset);
    const type = readType(offset + 4);
    const dataStart = offset + 8;
    const dataEnd = dataStart + len;
    if (dataEnd + 4 > u8.length) break;

    if (type === "tEXt") {
      const chunk = u8.slice(dataStart, dataEnd);
      const nul = chunk.indexOf(0);
      if (nul > 0) {
        const key = new TextDecoder("latin1").decode(chunk.slice(0, nul));
        const val = new TextDecoder("utf-8", { fatal: false }).decode(chunk.slice(nul + 1));
        out[key] = val;
      }
    } else if (type === "iTXt") {
      const chunk = u8.slice(dataStart, dataEnd);
      let p = 0;

      const readNullTerm = () => {
        const start = p;
        while (p < chunk.length && chunk[p] !== 0) p++;
        const s = chunk.slice(start, p);
        p++; // skip null
        return s;
      };

      const key = new TextDecoder("latin1").decode(readNullTerm());
      const compressionFlag = chunk[p++]; // 0 or 1
      p++; // compressionMethod
      readNullTerm(); // langTag
      readNullTerm(); // translatedKeyword

      const textBytes = chunk.slice(p);

      // NOTE: if compressedFlag===1, you'd need zlib inflate; most pipelines use 0.
      if (compressionFlag === 0) {
        const val = new TextDecoder("utf-8", { fatal: false }).decode(textBytes);
        out[key] = val;
      }
    }

    if (type === "IEND") break;
    offset = dataEnd + 4; // CRC
  }

  return out;
}

export function normalizeGenerationParamsFromTextChunks(textMap) {
  // Prefer JSON blobs if present
  const candidates = [
    textMap.lcm,
    textMap.LCM,
    textMap.LCM_META,
    textMap.meta,
    textMap.Metadata,
    textMap.parameters_json,
  ].filter(Boolean);

  for (const c of candidates) {
    try {
      const obj = JSON.parse(c);
      if (obj && typeof obj === "object") return obj;
    } catch {}
  }

  // A1111-ish "parameters" fallback
  if (textMap.parameters) {
    const s = String(textMap.parameters);

    const out = {};
    const mSteps = s.match(/Steps:\s*(\d+)/i);
    const mCfg = s.match(/CFG(?:\s*scale)?:\s*([0-9.]+)/i);
    const mSeed = s.match(/Seed:\s*(\d+)/i);
    const mSize = s.match(/Size:\s*(\d+x\d+)/i);
    const mDen = s.match(/Denois(?:ing)?(?:\s*strength)?:\s*([0-9.]+)/i);

    if (mSteps) out.steps = Number(mSteps[1]);
    if (mCfg) out.cfg = Number(mCfg[1]);
    if (mSeed) out.seed = Number(mSeed[1]);
    if (mSize) out.size = mSize[1];
    if (mDen) out.denoise = Number(mDen[1]);

    // prompt line is often before "Negative prompt:" etc; keep simple:
    out.prompt = s.split("\n")[0]?.trim();

    return out;
  }

  return null;
}