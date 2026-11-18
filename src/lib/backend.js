// Shared backend URL / session-ID helpers (previously copy-pasted per page).

export function resolveBackendUrl() {
  if (process.env.REACT_APP_BACKEND_URL) return process.env.REACT_APP_BACKEND_URL.replace(/\/+$/, "");
  if (typeof window !== "undefined" && window.location.hostname !== "localhost" && window.location.hostname !== "127.0.0.1") {
    return window.location.origin;
  }
  return "http://localhost:8010";
}

export function toWsUrl(httpUrl) {
  return httpUrl.replace(/^http(s)?:\/\//i, (_m, isHttps) => (isHttps ? "wss://" : "ws://"));
}

/** 128-bit random session ID, e.g. session_3f9c…; matches the server whitelist. */
export function newSessionId() {
  const c = typeof window !== "undefined" ? window.crypto : undefined;
  if (c && typeof c.randomUUID === "function") return `session_${c.randomUUID().replace(/-/g, "")}`;
  const bytes = new Uint8Array(16);
  if (c && c.getRandomValues) c.getRandomValues(bytes);
  else for (let i = 0; i < 16; i++) bytes[i] = Math.floor(Math.random() * 256);
  return `session_${Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("")}`;
}

export function formatClock(seconds) {
  const s = Math.max(0, Math.round(Number(seconds) || 0));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}
