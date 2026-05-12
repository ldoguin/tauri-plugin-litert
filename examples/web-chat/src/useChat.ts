import { useState, useCallback, useEffect, useRef } from "react";
import {
  createConversation,
  getConversations,
  getConversation,
  deleteConversation,
  renameConversation,
  addMessage,
  appendMessageContent,
  setMessageEmbedding,
  indexMessage,
  type Conversation,
  type Message,
} from "./store";
import { embed, initEmbeddings, getActiveBackend, type EmbeddingStatus } from "./embeddings";
import { retrieve, buildRagContext } from "./rag";
import {
  generateStream,
  setApiConfig,
  setActiveLmModel,
  getActiveLmModel,
  getActiveBackend as getLlmBackend,
  type ApiConfig,
  type LlmBackend,
} from "./llm";

export interface ChatState {
  conversations: Conversation[];
  activeId: string | null;
  activeConversation: Conversation | null;
  ragEnabled: boolean;
  isGenerating: boolean;
  embeddingStatus: EmbeddingStatus | null;
  embeddingBackend: string;
  llmBackend: LlmBackend;
  lastRagChunks: Array<{ content: string; score: number; role: string }>;
}

export interface ChatActions {
  newConversation: () => void;
  selectConversation: (id: string) => void;
  deleteConv: (id: string) => void;
  renameConv: (id: string, title: string) => void;
  sendMessage: (text: string) => Promise<void>;
  setRagEnabled: (v: boolean) => void;
  initEmbeddingEngine: (liteRtModelUrl?: string) => Promise<void>;
  configureApi: (config: ApiConfig) => void;
  configureLmModel: (modelId: string | null) => void;
}

export function useChat(): ChatState & ChatActions {
  const [conversations, setConversations] = useState<Conversation[]>(() => getConversations());
  const [activeId, setActiveId] = useState<string | null>(null);
  const [ragEnabled, setRagEnabled] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [embeddingStatus, setEmbeddingStatus] = useState<EmbeddingStatus | null>(null);
  const [llmBackend, setLlmBackend] = useState<LlmBackend>("mock");
  const [lastRagChunks, setLastRagChunks] = useState<
    Array<{ content: string; score: number; role: string }>
  >([]);

  const activeIdRef = useRef<string | null>(null);
  activeIdRef.current = activeId;

  // Streaming assistant message id — kept in a ref so the streaming callback
  // can update the store without stale closure issues.
  const streamingMsgIdRef = useRef<string | null>(null);
  const streamingConvIdRef = useRef<string | null>(null);

  const refresh = useCallback(() => setConversations([...getConversations()]), []);

  useEffect(() => {
    initEmbeddings().then((status) => setEmbeddingStatus(status));
  }, []);

  const initEmbeddingEngine = useCallback(async (liteRtModelUrl?: string) => {
    const status = await initEmbeddings(liteRtModelUrl);
    setEmbeddingStatus(status);
  }, []);

  const configureApi = useCallback((config: ApiConfig) => {
    setApiConfig(config);
    setLlmBackend("api");
  }, []);

  const configureLmModel = useCallback((modelId: string | null) => {
    setActiveLmModel(modelId);
    // Re-read backend — may have changed to "mediapipe" if loadWebLlm() was called.
    setLlmBackend(getLlmBackend());
  }, []);

  const newConversation = useCallback(() => {
    const conv = createConversation();
    refresh();
    setActiveId(conv.id);
  }, [refresh]);

  const selectConversation = useCallback((id: string) => {
    setActiveId(id);
    setLastRagChunks([]);
  }, []);

  const deleteConv = useCallback((id: string) => {
    deleteConversation(id);
    refresh();
    if (activeIdRef.current === id) setActiveId(null);
  }, [refresh]);

  const renameConv = useCallback((id: string, title: string) => {
    renameConversation(id, title);
    refresh();
  }, [refresh]);

  const embedAndIndex = useCallback(async (msg: Message, convId: string) => {
    try {
      const vec = await embed(msg.content);
      setMessageEmbedding(convId, msg.id, vec);
      indexMessage({
        conversationId: convId,
        messageId: msg.id,
        role: msg.role,
        content: msg.content,
        embedding: vec,
      });
      setConversations([...getConversations()]);
    } catch (e) {
      console.warn("[useChat] embed failed:", e);
    }
  }, []);

  const sendMessage = useCallback(async (text: string) => {
    const convId = activeIdRef.current;
    if (!convId || !text.trim()) return;

    // 1. Add user message.
    const userMsg = addMessage(convId, "user", text.trim());
    refresh();

    // 2. Embed user message in background.
    const userEmbedPromise = embedAndIndex(userMsg, convId);

    setIsGenerating(true);
    setLastRagChunks([]);

    try {
      // 3. Build history for the LLM.
      const conv = getConversation(convId)!;
      const history = conv.messages.map((m) => ({ role: m.role, content: m.content }));

      // 4. RAG retrieval.
      let ragContext = "";
      if (ragEnabled) {
        await userEmbedPromise;
        const userVec = getConversation(convId)?.messages.find(
          (m) => m.id === userMsg.id
        )?.embedding;
        if (userVec) {
          const chunks = retrieve(userVec, { topK: 3, threshold: 0.25, excludeConversationId: convId });
          ragContext = buildRagContext(chunks);
          setLastRagChunks(chunks.map(({ entry, score }) => ({
            content: entry.content, score, role: entry.role,
          })));
        }
      }

      // 5. Create a placeholder assistant message to stream into.
      const assistantMsg = addMessage(convId, "assistant", "");
      streamingMsgIdRef.current = assistantMsg.id;
      streamingConvIdRef.current = convId;
      refresh();

      // 6. Stream generation — each chunk appends to the assistant message.
      await new Promise<void>((resolve) => {
        generateStream(history, ragContext, { modelId: getActiveLmModel() ?? undefined }, {
          onChunk: (chunk) => {
            const cId = streamingConvIdRef.current;
            const mId = streamingMsgIdRef.current;
            if (!cId || !mId) return;
            appendMessageContent(cId, mId, chunk);
            setConversations([...getConversations()]);
          },
          onDone: (_latencyMs) => {
            // Embed the completed assistant message.
            const cId = streamingConvIdRef.current;
            const mId = streamingMsgIdRef.current;
            if (cId && mId) {
              const msg = getConversation(cId)?.messages.find((m) => m.id === mId);
              if (msg) embedAndIndex(msg, cId);
            }
            streamingMsgIdRef.current = null;
            streamingConvIdRef.current = null;
            resolve();
          },
          onError: (err) => {
            console.error("[useChat] generation error:", err);
            streamingMsgIdRef.current = null;
            streamingConvIdRef.current = null;
            resolve();
          },
        });
      });
    } finally {
      setIsGenerating(false);
      refresh();
    }
  }, [ragEnabled, refresh, embedAndIndex]);

  const activeConversation = activeId
    ? (conversations.find((c) => c.id === activeId) ?? null)
    : null;

  return {
    conversations,
    activeId,
    activeConversation,
    ragEnabled,
    isGenerating,
    embeddingStatus,
    embeddingBackend: getActiveBackend(),
    llmBackend,
    lastRagChunks,
    newConversation,
    selectConversation,
    deleteConv,
    renameConv,
    sendMessage,
    setRagEnabled,
    initEmbeddingEngine,
    configureApi,
    configureLmModel,
  };
}
