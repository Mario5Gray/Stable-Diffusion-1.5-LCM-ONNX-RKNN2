// utils/imageToFile.js
export async function urlToFile(url, filename = "chat.png") {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch image failed: ${res.status}`);
  const blob = await res.blob();

  // try to preserve type
  const ext = blob.type === "image/jpeg" ? "jpg"
           : blob.type === "image/webp" ? "webp"
           : "png";

  const safeName = filename.includes(".") ? filename : `${filename}.${ext}`;
  return new File([blob], safeName, { type: blob.type || "image/png" });
}