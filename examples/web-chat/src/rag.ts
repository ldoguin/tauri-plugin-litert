/**
 * RAG retrieval utilities.
 * Generation is handled by llm.ts.
 */

import { cosineSimilarity } from "./embeddings";
import { getVectorIndex, type VectorEntry } from "./store";

export interface RetrievedChunk {
  entry: VectorEntry;
  score: number;
}

export function retrieve(
  queryEmbedding: number[],
  opts: {
    topK?: number;
    threshold?: number;
    excludeConversationId?: string;
  } = {}
): RetrievedChunk[] {
  const { topK = 3, threshold = 0.3, excludeConversationId } = opts;

  return getVectorIndex()
    .filter((e) => {
      if (excludeConversationId && e.conversationId === excludeConversationId) return false;
      return e.embedding.length > 0;
    })
    .map((entry) => ({ entry, score: cosineSimilarity(queryEmbedding, entry.embedding) }))
    .filter(({ score }) => score >= threshold)
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}

export function buildRagContext(chunks: RetrievedChunk[]): string {
  if (chunks.length === 0) return "";
  const lines = chunks.map(
    ({ entry, score }) =>
      `[${entry.role} — similarity ${score.toFixed(2)}]: ${entry.content}`
  );
  return (
    "Relevant context from previous conversations:\n" +
    lines.join("\n") +
    "\n\nUse the above context to inform your reply if relevant."
  );
}
