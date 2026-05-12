/**
 * In-memory store for conversations, messages, and embedding vectors.
 * All state lives in module-level variables — no persistence, no external deps.
 */

export interface Message {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: number;
  /** Embedding vector, set after vectorisation completes. */
  embedding?: number[];
}

export interface Conversation {
  id: string;
  title: string;
  createdAt: number;
  messages: Message[];
}

/** One entry in the flat vector index. */
export interface VectorEntry {
  conversationId: string;
  messageId: string;
  role: Message["role"];
  content: string;
  embedding: number[];
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let conversations: Conversation[] = [];
let vectorIndex: VectorEntry[] = [];
let nextConvId = 1;
let nextMsgId = 1;

// ---------------------------------------------------------------------------
// Conversation CRUD
// ---------------------------------------------------------------------------

export function createConversation(title?: string): Conversation {
  const conv: Conversation = {
    id: `conv-${nextConvId++}`,
    title: title ?? `Conversation ${nextConvId - 1}`,
    createdAt: Date.now(),
    messages: [],
  };
  conversations = [...conversations, conv];
  return conv;
}

export function getConversations(): Conversation[] {
  return conversations;
}

export function getConversation(id: string): Conversation | undefined {
  return conversations.find((c) => c.id === id);
}

export function deleteConversation(id: string): void {
  conversations = conversations.filter((c) => c.id !== id);
  vectorIndex = vectorIndex.filter((e) => e.conversationId !== id);
}

export function renameConversation(id: string, title: string): void {
  conversations = conversations.map((c) =>
    c.id === id ? { ...c, title } : c
  );
}

// ---------------------------------------------------------------------------
// Message CRUD
// ---------------------------------------------------------------------------

export function addMessage(
  conversationId: string,
  role: Message["role"],
  content: string
): Message {
  const msg: Message = {
    id: `msg-${nextMsgId++}`,
    role,
    content,
    timestamp: Date.now(),
  };
  conversations = conversations.map((c) =>
    c.id === conversationId
      ? { ...c, messages: [...c.messages, msg] }
      : c
  );
  return msg;
}

export function setMessageEmbedding(
  conversationId: string,
  messageId: string,
  embedding: number[]
): void {
  conversations = conversations.map((c) => {
    if (c.id !== conversationId) return c;
    return {
      ...c,
      messages: c.messages.map((m) =>
        m.id === messageId ? { ...m, embedding } : m
      ),
    };
  });
}

/** Append a text chunk to a message's content (used during streaming). */
export function appendMessageContent(
  conversationId: string,
  messageId: string,
  chunk: string
): void {
  conversations = conversations.map((c) => {
    if (c.id !== conversationId) return c;
    return {
      ...c,
      messages: c.messages.map((m) =>
        m.id === messageId ? { ...m, content: m.content + chunk } : m
      ),
    };
  });
}

// ---------------------------------------------------------------------------
// Vector index
// ---------------------------------------------------------------------------

export function indexMessage(entry: VectorEntry): void {
  // Replace if already indexed (re-embed scenario).
  const existing = vectorIndex.findIndex(
    (e) => e.messageId === entry.messageId
  );
  if (existing >= 0) {
    vectorIndex = [
      ...vectorIndex.slice(0, existing),
      entry,
      ...vectorIndex.slice(existing + 1),
    ];
  } else {
    vectorIndex = [...vectorIndex, entry];
  }
}

export function getVectorIndex(): VectorEntry[] {
  return vectorIndex;
}

export function clearVectorIndex(): void {
  vectorIndex = [];
}
