"use client";

import { FormEvent, useRef, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8001";

type Status = "idle" | "loading" | "streaming" | "done" | "error";
type StageState = "pending" | "active" | "done";

type PipelineStage = {
  id: string;
  label: string;
};

type ChatTurn = {
  role: "user" | "assistant";
  content: string;
};

type ConversationTurn = {
  query: string;
  answer: string;
  sources: string[];
};

const MAX_HISTORY_TURNS = 6;

type StageUpdate = {
  id: string;
  state: StageState;
  label: string;
};

function buildApiHistory(turns: ConversationTurn[]): ChatTurn[] {
  const history = turns.flatMap((turn) => [
    { role: "user" as const, content: turn.query },
    { role: "assistant" as const, content: turn.answer },
  ]);
  return history.slice(-MAX_HISTORY_TURNS);
}

function ConversationThread({ turns }: { turns: ConversationTurn[] }) {
  if (turns.length === 0) {
    return null;
  }

  return (
    <section className="mb-8 space-y-8">
      {turns.map((turn, index) => (
        <article key={`${turn.query}-${index}`} className="space-y-4">
          <p className="text-lg font-medium leading-snug">{turn.query}</p>
          {turn.sources.length > 0 && (
            <div>
              <h3 className="mb-2 text-xs font-medium uppercase tracking-wide text-[var(--muted)]">
                Sources
              </h3>
              <ul className="space-y-1">
                {turn.sources.map((url) => (
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
            </div>
          )}
          <div className="rounded-2xl border border-[var(--border)] bg-[#111] p-6">
            <div className="whitespace-pre-wrap text-sm leading-7">{turn.answer}</div>
          </div>
        </article>
      ))}
    </section>
  );
}

const PIPELINE_STAGES: PipelineStage[] = [
  { id: "rewrite", label: "Rewriting query" },
  { id: "decompose", label: "Planning sub-searches" },
  { id: "search", label: "Searching the web" },
  { id: "fetch", label: "Reading sources" },
  { id: "ingest", label: "Indexing content" },
  { id: "retrieve", label: "Ranking passages" },
  { id: "generate", label: "Generating answer" },
];

function initialStageStates(): Record<string, StageState> {
  return Object.fromEntries(
    PIPELINE_STAGES.map((stage) => [stage.id, "pending"]),
  );
}

function parseSseBlock(block: string): { event: string; data: string } | null {
  const lines = block.split("\n");
  let event = "message";
  let data = "";

  for (const line of lines) {
    if (!line || line.startsWith(":")) {
      continue;
    }
    if (line.startsWith("event: ")) {
      event = line.slice(7).trim();
    } else if (line.startsWith("data: ")) {
      data += (data ? "\n" : "") + line.slice(6);
    }
  }

  if (!data && event === "message") {
    return null;
  }

  return { event, data };
}

function extractSseBlocks(buffer: string): { blocks: string[]; rest: string } {
  const normalized = buffer.replace(/\r\n/g, "\n");
  const parts = normalized.split("\n\n");
  const rest = parts.pop() ?? "";
  return { blocks: parts.filter(Boolean), rest };
}

function applyStageUpdate(
  prev: Record<string, StageState>,
  update: StageUpdate,
): Record<string, StageState> {
  const next = { ...prev, [update.id]: update.state };
  if (update.state === "active") {
    const activeIndex = PIPELINE_STAGES.findIndex((s) => s.id === update.id);
    for (let i = 0; i < activeIndex; i += 1) {
      const stageId = PIPELINE_STAGES[i].id;
      if (next[stageId] !== "done") {
        next[stageId] = "done";
      }
    }
  }
  return next;
}

function PipelineProgress({
  stageStates,
  stageLabels,
  statusText,
}: {
  stageStates: Record<string, StageState>;
  stageLabels: Record<string, string>;
  statusText: string;
}) {
  return (
    <section className="mb-6 rounded-xl border border-[var(--border)] bg-[#111] p-4">
      <ul className="space-y-2">
        {PIPELINE_STAGES.map((stage) => {
          const state = stageStates[stage.id] ?? "pending";
          const label = stageLabels[stage.id] || stage.label;
          const isActive = state === "active";
          const isDone = state === "done";

          return (
            <li
              key={stage.id}
              className={`flex items-center gap-3 text-sm transition-colors ${
                isActive
                  ? "text-white"
                  : isDone
                    ? "text-[var(--muted)]"
                    : "text-[var(--muted)]/50"
              }`}
            >
              <span
                className={`flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[10px] ${
                  isDone
                    ? "bg-emerald-900/60 text-emerald-400"
                    : isActive
                      ? "bg-[var(--accent)]/20 text-[var(--accent)]"
                      : "border border-[var(--border)]"
                }`}
              >
                {isDone ? "✓" : isActive ? "…" : ""}
              </span>
              <span className={isActive ? "font-medium" : ""}>{label}</span>
              {isActive && (
                <span className="ml-auto h-1.5 w-1.5 animate-pulse rounded-full bg-[var(--accent)]" />
              )}
            </li>
          );
        })}
      </ul>
      {statusText && (
        <p className="mt-3 border-t border-[var(--border)] pt-3 text-xs text-[var(--muted)]">
          {statusText}
        </p>
      )}
    </section>
  );
}

function SearchForm({
  query,
  setQuery,
  onSubmit,
  isBusy,
  sticky = false,
}: {
  query: string;
  setQuery: (value: string) => void;
  onSubmit: (event: FormEvent) => void;
  isBusy: boolean;
  sticky?: boolean;
}) {
  if (sticky) {
    return (
      <div className="fixed inset-x-0 bottom-0 z-50 border-t border-[var(--border)] bg-[#0a0a0a]/90 backdrop-blur-md">
        <form
          onSubmit={onSubmit}
          className="mx-auto flex max-w-3xl gap-3 px-6 py-4"
        >
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask a follow-up..."
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
      </div>
    );
  }

  return (
    <form onSubmit={onSubmit} className="mb-8 flex gap-3">
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
  );
}

export default function Home() {
  const [query, setQuery] = useState("");
  const [turns, setTurns] = useState<ConversationTurn[]>([]);
  const [activeQuery, setActiveQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<string[]>([]);
  const [statusText, setStatusText] = useState("");
  const [status, setStatus] = useState<Status>("idle");
  const [stageStates, setStageStates] = useState<Record<string, StageState>>({});
  const [stageLabels, setStageLabels] = useState<Record<string, string>>({});
  const [error, setError] = useState("");
  const abortRef = useRef<AbortController | null>(null);
  const answerRef = useRef("");
  const sourcesRef = useRef<string[]>([]);
  const submittedQueryRef = useRef("");

  function handleSseEvent(parsed: { event: string; data: string }) {
    if (parsed.event === "status") {
      setStatusText(parsed.data);
    } else if (parsed.event === "stage") {
      const update = JSON.parse(parsed.data) as StageUpdate;
      setStageStates((prev) => applyStageUpdate(prev, update));
      if (update.label) {
        setStageLabels((prev) => ({ ...prev, [update.id]: update.label }));
      }
    } else if (parsed.event === "sources") {
      const nextSources = JSON.parse(parsed.data) as string[];
      sourcesRef.current = nextSources;
      setSources(nextSources);
    } else if (parsed.event === "token") {
      setStatus("streaming");
      answerRef.current += parsed.data;
      setAnswer((prev) => prev + parsed.data);
    } else if (parsed.event === "error") {
      throw new Error(parsed.data);
    } else if (parsed.event === "done") {
      setStatus("done");
      setStatusText("");
      setStageStates((prev) =>
        Object.fromEntries(
          PIPELINE_STAGES.map((stage) => [stage.id, "done" as StageState]),
        ),
      );
    }
  }

  function processSseBuffer(buffer: string): string {
    const { blocks, rest } = extractSseBlocks(buffer);
    for (const block of blocks) {
      const parsed = parseSseBlock(block);
      if (parsed) {
        handleSseEvent(parsed);
      }
    }
    return rest;
  }

  async function handleSearch(event: FormEvent) {
    event.preventDefault();
    const trimmed = query.trim();
    if (!trimmed) return;

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    submittedQueryRef.current = trimmed;
    setActiveQuery(trimmed);
    setQuery("");
    window.scrollTo({ top: 0, behavior: "smooth" });
    answerRef.current = "";
    sourcesRef.current = [];
    setAnswer("");
    setSources([]);
    setError("");
    setStatusText("");
    setStageStates(initialStageStates());
    setStageLabels({});
    setStatus("loading");

    const history = buildApiHistory(turns);
    const priorUrls =
      turns.length > 0 ? turns[turns.length - 1].sources.slice(0, 8) : [];

    try {
      const response = await fetch(`${API_URL}/api/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: trimmed,
          history,
          prior_urls: priorUrls,
        }),
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
        if (value) {
          buffer += decoder.decode(value, { stream: true });
          buffer = processSseBuffer(buffer);
        }
        if (done) {
          buffer += decoder.decode();
          processSseBuffer(buffer);
          break;
        }
      }

      setStatus((current) =>
        current === "streaming" || current === "loading" ? "done" : current,
      );

      if (answerRef.current) {
        setTurns((prev) => [
          ...prev,
          {
            query: submittedQueryRef.current,
            answer: answerRef.current,
            sources: sourcesRef.current,
          },
        ]);
        setActiveQuery("");
        setAnswer("");
        setSources([]);
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        return;
      }
      setStatus("error");
      setError(err instanceof Error ? err.message : "Something went wrong");
    }
  }

  const isBusy = status === "loading" || status === "streaming";
  const showProgress = isBusy || (status === "done" && Object.keys(stageStates).length > 0);
  const hasSession =
    status !== "idle" ||
    turns.length > 0 ||
    Boolean(activeQuery) ||
    Boolean(answer) ||
    sources.length > 0;

  return (
    <main
      className={`mx-auto flex min-h-screen max-w-3xl flex-col px-6 py-16 ${
        hasSession ? "pb-28" : ""
      }`}
    >
      <header className="mb-10">
        <h1 className="text-3xl font-semibold tracking-tight">synthAI</h1>
        <p className="mt-2 text-sm text-[var(--muted)]">
          Search the web, retrieve relevant passages, stream an answer.
        </p>
      </header>

      {!hasSession && (
        <SearchForm
          query={query}
          setQuery={setQuery}
          onSubmit={handleSearch}
          isBusy={isBusy}
        />
      )}

      <ConversationThread turns={turns} />

      {activeQuery && (
        <p className="mb-6 text-lg font-medium leading-snug">{activeQuery}</p>
      )}

      {showProgress && (
        <PipelineProgress
          stageStates={stageStates}
          stageLabels={stageLabels}
          statusText={statusText}
        />
      )}

      {error && (
        <div className="mb-6 rounded-xl border border-red-900 bg-red-950/40 px-4 py-3 text-sm text-red-300">
          {error}
        </div>
      )}

      {sources.length > 0 && activeQuery && (
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

      {answer && activeQuery && (
        <section className="rounded-2xl border border-[var(--border)] bg-[#111] p-6">
          <h2 className="mb-4 text-sm font-medium text-[var(--muted)]">Answer</h2>
          <div className="whitespace-pre-wrap text-sm leading-7">{answer}</div>
        </section>
      )}

      {hasSession && (
        <SearchForm
          query={query}
          setQuery={setQuery}
          onSubmit={handleSearch}
          isBusy={isBusy}
          sticky
        />
      )}
    </main>
  );
}
