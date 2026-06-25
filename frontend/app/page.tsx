"use client";

import { FormEvent, useRef, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8001";

type Status = "idle" | "loading" | "streaming" | "done" | "error";

function parseSseBlock(block: string): { event: string; data: string } | null {
  const lines = block.split("\n");
  let event = "message";
  let data = "";

  for (const line of lines) {
    if (line.startsWith("event: ")) {
      event = line.slice(7).trim();
    } else if (line.startsWith("data: ")) {
      data += line.slice(6);
    }
  }

  if (!data && event === "message") {
    return null;
  }

  return { event, data };
}

type DebugTrace = {
  stages_ms?: Record<string, number>;
  cache_stats?: Record<string, unknown>;
  sub_queries?: string[];
  chunk_count?: number;
  total_latency_ms?: number;
  top_chunks?: Array<Record<string, unknown>>;
};

export default function Home() {
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<string[]>([]);
  const [statusText, setStatusText] = useState("");
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState("");
  const [debugTrace, setDebugTrace] = useState<DebugTrace | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  async function handleSearch(event: FormEvent) {
    event.preventDefault();
    const trimmed = query.trim();
    if (!trimmed) return;

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setAnswer("");
    setSources([]);
    setError("");
    setDebugTrace(null);
    setStatusText("Starting...");
    setStatus("loading");

    try {
      const response = await fetch(`${API_URL}/api/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: trimmed }),
        signal: controller.signal,
      });

      if (!response.ok) {
        const detail = await response.text();
        throw new Error(detail || `Request failed (${response.status})`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error("No response stream");
      }

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const blocks = buffer.split("\n\n");
        buffer = blocks.pop() ?? "";

        for (const block of blocks) {
          const parsed = parseSseBlock(block);
          if (!parsed) continue;

          if (parsed.event === "status") {
            setStatusText(parsed.data);
          } else if (parsed.event === "sources") {
            setSources(JSON.parse(parsed.data));
          } else if (parsed.event === "token") {
            setStatus("streaming");
            setAnswer((prev) => prev + parsed.data);
          } else if (parsed.event === "error") {
            throw new Error(parsed.data);
          } else if (parsed.event === "done") {
            setStatus("done");
            setStatusText("");
          } else if (parsed.event === "debug") {
            setDebugTrace(JSON.parse(parsed.data));
          }
        }
      }

      setStatus((current) =>
        current === "streaming" || current === "loading" ? "done" : current,
      );
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        return;
      }
      setStatus("error");
      setError(err instanceof Error ? err.message : "Something went wrong");
    }
  }

  const isBusy = status === "loading" || status === "streaming";

  return (
    <main className="mx-auto flex min-h-screen max-w-3xl flex-col px-6 py-16">
      <header className="mb-10">
        <h1 className="text-3xl font-semibold tracking-tight">synthAI</h1>
        <p className="mt-2 text-sm text-[var(--muted)]">
          Search the web, retrieve relevant passages, stream an answer.
        </p>
      </header>

      <form onSubmit={handleSearch} className="mb-8 flex gap-3">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask anything..."
          className="flex-1 rounded-xl border border-[var(--border)] bg-[#111] px-4 py-3 text-sm outline-none focus:border-[var(--accent)]"
          disabled={isBusy}
        />
        <button
          type="submit"
          disabled={isBusy || !query.trim()}
          className="rounded-xl bg-[var(--accent)] px-5 py-3 text-sm font-medium text-white disabled:opacity-50"
        >
          {isBusy ? "Searching..." : "Search"}
        </button>
      </form>

      {statusText && (
        <p className="mb-4 text-sm text-[var(--muted)]">{statusText}</p>
      )}

      {error && (
        <div className="mb-6 rounded-xl border border-red-900 bg-red-950/40 px-4 py-3 text-sm text-red-300">
          {error}
        </div>
      )}

      {sources.length > 0 && (
        <section className="mb-8">
          <h2 className="mb-3 text-sm font-medium text-[var(--muted)]">Sources</h2>
          <ul className="space-y-2">
            {sources.map((url) => (
              <li key={url}>
                <a
                  href={url}
                  target="_blank"
                  rel="noreferrer"
                  className="block truncate text-sm text-[var(--accent)] hover:underline"
                >
                  {url}
                </a>
              </li>
            ))}
          </ul>
        </section>
      )}

      {debugTrace && (
        <section className="mb-8 rounded-xl border border-dashed border-[var(--border)] bg-[#0a0a0a] p-4 text-xs text-[var(--muted)]">
          <h2 className="mb-2 font-medium text-[var(--muted)]">Debug trace</h2>
          {debugTrace.sub_queries && debugTrace.sub_queries.length > 0 && (
            <p className="mb-2">
              Sub-queries: {debugTrace.sub_queries.join(" | ")}
            </p>
          )}
          {debugTrace.cache_stats && (
            <p className="mb-2">
              Cache: {JSON.stringify(debugTrace.cache_stats)}
            </p>
          )}
          {debugTrace.stages_ms && (
            <p className="mb-2">
              Stages (ms): {JSON.stringify(debugTrace.stages_ms)}
            </p>
          )}
          <p>Total latency: {debugTrace.total_latency_ms ?? "—"} ms</p>
        </section>
      )}

      {answer && (
        <section className="rounded-2xl border border-[var(--border)] bg-[#111] p-6">
          <h2 className="mb-4 text-sm font-medium text-[var(--muted)]">Answer</h2>
          <div className="whitespace-pre-wrap text-sm leading-7">{answer}</div>
        </section>
      )}
    </main>
  );
}
