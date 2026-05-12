import { useState, useRef, useEffect, KeyboardEvent } from "react";
import { useChat } from "./useChat";
import { loadWebLlm, unloadWebLlm } from "./llm";
import type { Message } from "./store";
import "./App.css";

// ---------------------------------------------------------------------------
// Sidebar
// ---------------------------------------------------------------------------

function Sidebar({
  conversations,
  activeId,
  onNew,
  onSelect,
  onDelete,
  onRename,
}: {
  conversations: ReturnType<typeof useChat>["conversations"];
  activeId: string | null;
  onNew: () => void;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  onRename: (id: string, title: string) => void;
}) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editValue, setEditValue] = useState("");

  const startEdit = (id: string, current: string) => {
    setEditingId(id);
    setEditValue(current);
  };

  const commitEdit = (id: string) => {
    if (editValue.trim()) onRename(id, editValue.trim());
    setEditingId(null);
  };

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <span className="sidebar-title">Conversations</span>
        <button className="btn-icon" onClick={onNew} title="New conversation">+</button>
      </div>
      <ul className="conv-list">
        {conversations.length === 0 && (
          <li className="conv-empty">No conversations yet</li>
        )}
        {conversations.map((c) => (
          <li
            key={c.id}
            className={"conv-item" + (c.id === activeId ? " active" : "")}
            onClick={() => onSelect(c.id)}
          >
            {editingId === c.id ? (
              <input
                className="conv-rename-input"
                value={editValue}
                autoFocus
                onChange={(e) => setEditValue(e.target.value)}
                onBlur={() => commitEdit(c.id)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") commitEdit(c.id);
                  if (e.key === "Escape") setEditingId(null);
                }}
                onClick={(e) => e.stopPropagation()}
              />
            ) : (
              <>
                <span className="conv-title">{c.title}</span>
                <span className="conv-count">{c.messages.length}</span>
                <button className="btn-icon conv-action" title="Rename"
                  onClick={(e) => { e.stopPropagation(); startEdit(c.id, c.title); }}>✎</button>
                <button className="btn-icon conv-action danger" title="Delete"
                  onClick={(e) => { e.stopPropagation(); onDelete(c.id); }}>✕</button>
              </>
            )}
          </li>
        ))}
      </ul>
    </aside>
  );
}

// ---------------------------------------------------------------------------
// Message bubble
// ---------------------------------------------------------------------------

function MessageBubble({ msg }: { msg: Message }) {
  const hasVec = (msg.embedding?.length ?? 0) > 0;
  return (
    <div className={"bubble-wrap " + msg.role}>
      <div className={"bubble " + msg.role}>
        <div className="bubble-content">{msg.content}</div>
        <div className="bubble-meta">
          <span className="bubble-time">{new Date(msg.timestamp).toLocaleTimeString()}</span>
          <span
            className={"vec-badge " + (hasVec ? "ready" : "pending")}
            title={hasVec ? "Embedding dim: " + msg.embedding!.length : "Embedding pending…"}
          >
            {hasVec ? "⬡ vec" : "⬡ …"}
          </span>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// RAG debug panel
// ---------------------------------------------------------------------------

function RagPanel({ chunks }: { chunks: Array<{ content: string; score: number; role: string }> }) {
  if (chunks.length === 0) return null;
  return (
    <div className="rag-panel">
      <div className="rag-panel-title">RAG — retrieved context</div>
      {chunks.map((c, i) => (
        <div key={i} className="rag-chunk">
          <span className="rag-score">{c.score.toFixed(3)}</span>
          <span className="rag-role">{c.role}</span>
          <span className="rag-text">{c.content.slice(0, 120)}{c.content.length > 120 ? "…" : ""}</span>
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Toolbar
// ---------------------------------------------------------------------------

type ToolbarPanel = "none" | "llm" | "api" | "embed";

function Toolbar({
  ragEnabled, onToggleRag, embeddingStatus, embeddingBackend,
  onInitLiteRt, llmBackend, onConfigureApi, onLlmBackendChange,
}: {
  ragEnabled: boolean;
  onToggleRag: (v: boolean) => void;
  embeddingStatus: ReturnType<typeof useChat>["embeddingStatus"];
  embeddingBackend: string;
  llmBackend: ReturnType<typeof useChat>["llmBackend"];
  onInitLiteRt: (url: string) => void;
  onConfigureApi: (baseUrl: string, apiKey: string, model: string) => void;
  onLlmBackendChange: () => void;
}) {
  const [panel, setPanel] = useState<ToolbarPanel>("none");
  const [liteRtUrl, setLiteRtUrl] = useState("");
  const [llmUrl, setLlmUrl] = useState("");
  const [llmLoading, setLlmLoading] = useState(false);
  const [apiUrl, setApiUrl] = useState("https://api.groq.com/openai/v1");
  const [apiKey, setApiKey] = useState("");
  const [apiModel, setApiModel] = useState("llama-3.1-8b-instant");

  const embedLabel =
    embeddingStatus === null ? "initialising…"
    : embeddingStatus.backend === "litert" ? "embed: LiteRT"
    : embeddingStatus.backend === "use" ? "embed: USE"
    : "embed: BoW";

  const embedBadgeClass =
    embeddingStatus?.backend === "litert" ? "badge-litert"
    : embeddingStatus?.backend === "use" ? "badge-use"
    : "badge-bow";

  const llmLabel =
    llmBackend === "tauri" ? "LLM: LiteRT-LM"
    : llmBackend === "mediapipe" ? "LLM: on-device"
    : llmBackend === "api" ? "LLM: API"
    : "LLM: mock";

  const llmBadgeClass =
    llmBackend === "tauri" || llmBackend === "mediapipe" ? "badge-litert"
    : llmBackend === "api" ? "badge-use"
    : "badge-bow";

  const handleLoadLlm = async () => {
    if (!llmUrl.trim()) return;
    setLlmLoading(true);
    try {
      await loadWebLlm({ modelUrl: llmUrl.trim() });
      onLlmBackendChange();
      setPanel("none");
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      if (msg.toLowerCase().includes("webgpu") || msg.toLowerCase().includes("adapter")) {
        alert(
          "WebGPU is not available in this browser.\n\n" +
          "Use Chrome 113+ or Edge 113+ with WebGPU enabled.\n\n" +
          "The model will still load using the CPU/Wasm fallback — retrying…"
        );
        // Retry was already attempted with CPU fallback inside loadWebLlm.
        // If it still failed, surface the original error.
      } else {
        alert("Failed to load LLM: " + msg);
      }
    } finally {
      setLlmLoading(false);
    }
  };

  const handleUnloadLlm = () => {
    unloadWebLlm();
    onLlmBackendChange();
  };

  return (
    <div className="toolbar">
      <div className="toolbar-left">
        <label className="rag-toggle">
          <input type="checkbox" checked={ragEnabled} onChange={(e) => onToggleRag(e.target.checked)} />
          <span>RAG</span>
        </label>
        <span className={"badge " + embedBadgeClass} title={"Embedding: " + embeddingBackend}>{embedLabel}</span>
        <span className={"badge " + llmBadgeClass}>{llmLabel}</span>
      </div>
      <div className="toolbar-right">
        {panel === "llm" ? (
          <div className="litert-url-row">
            <input className="litert-url-input" style={{width: 340}}
              placeholder="https://…/gemma3-1b-it-int4-web.task"
              value={llmUrl} onChange={(e) => setLlmUrl(e.target.value)} />
            <button className="btn-sm" onClick={handleLoadLlm} disabled={llmLoading}>
              {llmLoading ? "Loading…" : "Load"}
            </button>
            <button className="btn-sm secondary" onClick={() => setPanel("none")}>✕</button>
          </div>
        ) : panel === "api" ? (
          <div className="litert-url-row">
            <input className="litert-url-input" placeholder="API base URL" style={{width:190}}
              value={apiUrl} onChange={(e) => setApiUrl(e.target.value)} />
            <input className="litert-url-input" placeholder="API key" style={{width:120}}
              value={apiKey} onChange={(e) => setApiKey(e.target.value)} />
            <input className="litert-url-input" placeholder="model" style={{width:150}}
              value={apiModel} onChange={(e) => setApiModel(e.target.value)} />
            <button className="btn-sm" onClick={() => {
              if (apiUrl.trim()) { onConfigureApi(apiUrl.trim(), apiKey.trim(), apiModel.trim()); setPanel("none"); }
            }}>Save</button>
            <button className="btn-sm secondary" onClick={() => setPanel("none")}>✕</button>
          </div>
        ) : panel === "embed" ? (
          <div className="litert-url-row">
            <input className="litert-url-input" placeholder="https://…/model.tflite"
              value={liteRtUrl} onChange={(e) => setLiteRtUrl(e.target.value)} />
            <button className="btn-sm" onClick={() => { if (liteRtUrl.trim()) { onInitLiteRt(liteRtUrl.trim()); setPanel("none"); } }}>Load</button>
            <button className="btn-sm secondary" onClick={() => setPanel("none")}>✕</button>
          </div>
        ) : (
          <>
            {llmBackend === "mediapipe"
              ? <button className="btn-sm danger" onClick={handleUnloadLlm}>Unload LLM</button>
              : <button className="btn-sm secondary" onClick={() => setPanel("llm")}>Load LLM</button>
            }
            <button className="btn-sm secondary" onClick={() => setPanel("api")}>API config</button>
            <button className="btn-sm secondary" onClick={() => setPanel("embed")}>Embed model</button>
          </>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Chat panel
// ---------------------------------------------------------------------------

function ChatPanel({
  conversation, isGenerating, ragEnabled, lastRagChunks, onSend,
}: {
  conversation: ReturnType<typeof useChat>["activeConversation"];
  isGenerating: boolean;
  ragEnabled: boolean;
  lastRagChunks: ReturnType<typeof useChat>["lastRagChunks"];
  onSend: (text: string) => void;
}) {
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [conversation?.messages.length, isGenerating]);

  const handleSend = () => {
    const text = input.trim();
    if (!text || isGenerating) return;
    setInput("");
    onSend(text);
  };

  const handleKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
  };

  if (!conversation) {
    return (
      <div className="chat-empty">
        <p>Select a conversation or create a new one.</p>
      </div>
    );
  }

  return (
    <div className="chat-panel">
      <div className="messages">
        {conversation.messages.length === 0 && (
          <div className="messages-empty">Send a message to start.</div>
        )}
        {conversation.messages.map((m) => <MessageBubble key={m.id} msg={m} />)}
        {isGenerating && (
          <div className="bubble-wrap assistant">
            <div className="bubble assistant typing">
              <span className="dot" /><span className="dot" /><span className="dot" />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {ragEnabled && <RagPanel chunks={lastRagChunks} />}

      <div className="input-row">
        <textarea
          className="chat-input"
          rows={2}
          placeholder={ragEnabled ? "Ask something… (RAG on)" : "Ask something…"}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKey}
          disabled={isGenerating}
        />
        <button className="btn-send" onClick={handleSend} disabled={isGenerating || !input.trim()}>
          Send
        </button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// App root
// ---------------------------------------------------------------------------

export default function App() {
  const chat = useChat();
  return (
    <div className="app">
      <Sidebar
        conversations={chat.conversations}
        activeId={chat.activeId}
        onNew={chat.newConversation}
        onSelect={chat.selectConversation}
        onDelete={chat.deleteConv}
        onRename={chat.renameConv}
      />
      <div className="main">
        <Toolbar
          ragEnabled={chat.ragEnabled}
          onToggleRag={chat.setRagEnabled}
          embeddingStatus={chat.embeddingStatus}
          embeddingBackend={chat.embeddingBackend}
          llmBackend={chat.llmBackend}
          onInitLiteRt={(url) => chat.initEmbeddingEngine(url)}
          onConfigureApi={(baseUrl, apiKey, model) =>
            chat.configureApi({ baseUrl, apiKey: apiKey || undefined, model })
          }
          onLlmBackendChange={() => chat.configureLmModel(null)}
        />
        <ChatPanel
          conversation={chat.activeConversation}
          isGenerating={chat.isGenerating}
          ragEnabled={chat.ragEnabled}
          lastRagChunks={chat.lastRagChunks}
          onSend={chat.sendMessage}
        />
      </div>
    </div>
  );
}
