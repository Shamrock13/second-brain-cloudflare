import { VECTORIZE_TOP_K_MULTIPLIER } from "../constants";
import type { EdgeProvenance } from "../graph/types";
import type { DistilledQuery } from "./distill";

const SUBSTRING_WEIGHT = 0.25;

export interface LinkedEvidenceInput {
  parentScore: number;
  parentContent: string;
  content: string;
  queryTokens: string[];
  corpus: Pick<DistilledQuery, "df" | "total">;
  hop: number;
  edgeWeight: number;
  provenance: EdgeProvenance;
  hopDecay: number;
}

export interface LinkedEvidenceScore {
  eligible: boolean;
  score: number;
}

export function graphSeedLimit(topK: number, candidateCount: number): number {
  return Math.min(candidateCount, topK * VECTORIZE_TOP_K_MULTIPLIER, 50);
}

export function relatedSlotLimit(topK: number): number {
  if (topK < 3) return 0;
  return topK < 6 ? 1 : 2;
}

const escapeRegExp = (value: string) => value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

function weightedCoverage(
  content: string,
  tokens: string[],
  corpus: Pick<DistilledQuery, "df" | "total">,
): number {
  const normalizedTokens = [...new Set(tokens.map(t => t.toLowerCase()).filter(Boolean))];
  if (!normalizedTokens.length) return 0;

  const hasCorpusIdf = !!corpus.df && !!corpus.total && normalizedTokens.every(t => corpus.df!.has(t));
  const weightOf = (token: string) => hasCorpusIdf
    ? Math.log(1 + corpus.total! / ((corpus.df!.get(token) ?? 0) + 1))
    : 1;
  const lower = content.toLowerCase();
  let matched = 0;
  let total = 0;

  for (const token of normalizedTokens) {
    const weight = weightOf(token);
    total += weight;
    if (new RegExp(`(?<![\\w])${escapeRegExp(token)}(?![\\w])`).test(lower)) {
      matched += weight;
    } else if (lower.includes(token)) {
      matched += weight * SUBSTRING_WEIGHT;
    }
  }

  return total > 0 ? matched / total : 0;
}

export function scoreLinkedEvidence(input: LinkedEvidenceInput): LinkedEvidenceScore {
  const parentCoverage = weightedCoverage(input.parentContent, input.queryTokens, input.corpus);
  const linkedCoverage = weightedCoverage(input.content, input.queryTokens, input.corpus);
  const unionCoverage = weightedCoverage(
    `${input.parentContent}\n${input.content}`,
    input.queryTokens,
    input.corpus,
  );
  const explicitContinuation = input.provenance === "explicit" && parentCoverage > 0;
  const eligible = linkedCoverage > 0 || explicitContinuation;
  if (!eligible) return { eligible: false, score: 0 };

  const provenanceFactor = input.provenance === "explicit"
    ? 1
    : input.provenance === "system"
      ? 0.9
      : 0.8;
  const coverageGain = Math.max(0, unionCoverage - parentCoverage);
  const evidenceFactor = 0.5 + 0.3 * linkedCoverage + 0.2 * coverageGain;

  return {
    eligible: true,
    score: input.parentScore
      * Math.pow(input.hopDecay, input.hop)
      * input.edgeWeight
      * provenanceFactor
      * evidenceFactor,
  };
}
