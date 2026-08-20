import type { EdgeType } from "../graph/types";
import { tokenizeQuery } from "../text/tokenize";
import type { DistilledQuery } from "./distill";

export type RecallIntent = "causal" | "chronology" | "current" | "direct";
export type EmbeddingQueryMode = "distilled" | "semantic" | "hybrid";
export const DEFAULT_EMBEDDING_QUERY_MODE: EmbeddingQueryMode = "distilled";

export interface QueryProfile {
  semanticQuery: string;
  lexicalQuery: string;
  lexicalTokens: string[];
  intent: RecallIntent;
}

const WORD = (items: string[]) => new RegExp(`(?<![\\w-])(?:${items.join("|")})(?![\\w-])`, "i");
const CAUSAL = WORD(["why", "reason", "decide", "decided", "chose", "choice", "changed", "change", "switched"]);
const CHRONOLOGY = WORD(["before", "after", "then", "history", "evolution", "became"]);
const CURRENT = WORD(["current", "now", "still", "latest"]);

export function buildQueryProfile(semanticQuery: string, distilled: DistilledQuery): QueryProfile {
  const clean = semanticQuery.trim();
  const intent: RecallIntent = CAUSAL.test(clean)
    ? "causal"
    : CHRONOLOGY.test(clean)
      ? "chronology"
      : CURRENT.test(clean)
        ? "current"
        : "direct";
  return {
    semanticQuery: clean,
    lexicalQuery: distilled.query,
    lexicalTokens: tokenizeQuery(distilled.query),
    intent,
  };
}

export function embeddingInput(profile: QueryProfile, mode: EmbeddingQueryMode): string {
  if (mode === "semantic") return profile.semanticQuery;
  if (mode === "hybrid") return profile.semanticQuery === profile.lexicalQuery
    ? profile.semanticQuery
    : `${profile.semanticQuery} ${profile.lexicalQuery}`;
  return profile.lexicalQuery;
}

export function edgeIntentCompatibility(intent: RecallIntent, edgeType: EdgeType): number {
  if (intent === "causal" && ["decided", "caused_by", "supersedes"].includes(edgeType)) return 1;
  if (intent === "chronology" && ["follows", "supersedes"].includes(edgeType)) return 1;
  return 0.5;
}
