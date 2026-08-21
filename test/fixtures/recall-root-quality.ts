import type { EdgeProvenance, EdgeType } from "../../src/graph/types";
import type { RecallIntent } from "../../src/recall/query-profile";

export type RootQualitySplit = "development" | "holdout";
export type RootQualityDomain = "personal" | "enterprise" | "product" | "architecture";
export type RootFailureShape =
  | "crowded-lexical-root"
  | "popular-broad-summary"
  | "long-parent-pollution"
  | "weak-generic-neighbor"
  | "absent-cluster-control";

export interface CandidateFixture {
  id: string;
  content: string;
  /** Query-local Vectorize chunk text. Absent on keyword-only and linked rows. */
  vectorContent?: string;
  /** Raw Vectorize similarity. Absent means this row is not in the dense pool. */
  denseScore?: number;
  /** Frozen post-fusion relevance used by the pre-plan baseline arm. */
  baselineScore?: number;
  /** Include this row in the controlled keyword arm. */
  keywordCandidate?: boolean;
  recallCount?: number;
  createdAt?: number;
  tags?: string[];
}

export interface EdgeFixture {
  sourceId: string;
  targetId: string;
  type: EdgeType;
  weight: number;
  provenance: EdgeProvenance;
}

export interface RootQualityCase {
  split: RootQualitySplit;
  domain: RootQualityDomain;
  failureShape: RootFailureShape;
  query: string;
  intent: RecallIntent;
  candidates: CandidateFixture[];
  edges: EdgeFixture[];
  authoritativeIds: string[];
  acceptableRootIds: string[];
  candidateAvailable: boolean;
}

const OLD = 1;

function directCandidates(prefix: string, authoritativeFifthId?: string): CandidateFixture[] {
  return Array.from({ length: 5 }, (_, index) => ({
    id: index === 4 && authoritativeFifthId ? authoritativeFifthId : `${prefix}-direct-${index}`,
    content: index === 4 && authoritativeFifthId
      ? "The controlled record contains the authoritative current answer."
      : `General direct record ${index} for the controlled corpus.`,
    vectorContent: `General direct record ${index}.`,
    denseScore: 0.99 - index * 0.01,
    baselineScore: 0.99 - index * 0.01,
    createdAt: OLD,
  }));
}

function semanticDistractors(prefix: string, count: number, start = 0.94): CandidateFixture[] {
  return Array.from({ length: count }, (_, index) => ({
    id: `${prefix}-semantic-${index}`,
    content: `Broad background summary ${index} without the decisive evidence.`,
    vectorContent: `Broad background summary ${index}.`,
    denseScore: start - index * 0.01,
    baselineScore: start - index * 0.01,
    createdAt: OLD,
  }));
}

function crowdedCandidates(
  prefix: string,
  rootId: string,
  answerId: string,
  rootChunk: string,
  answer: string,
  keywordText: string,
): CandidateFixture[] {
  return [
    ...directCandidates(prefix),
    ...semanticDistractors(prefix, 9),
    {
      id: rootId,
      content: "A compact decision marker whose parent text omits the query wording.",
      vectorContent: rootChunk,
      denseScore: 0.76,
      baselineScore: 0.5,
      createdAt: OLD,
    },
    ...Array.from({ length: 4 }, (_, index) => ({
      id: `${prefix}-keyword-${index}`,
      content: `${keywordText} background index ${index}.`,
      baselineScore: 0.7 - index * 0.01,
      keywordCandidate: true,
      createdAt: OLD,
    })),
    { id: answerId, content: answer, createdAt: OLD },
  ];
}

function popularityCandidates(
  prefix: string,
  rootId: string,
  popularId: string,
  answerId: string,
  answer: string,
  keywordText: string,
): CandidateFixture[] {
  return [
    ...directCandidates(prefix),
    ...semanticDistractors(prefix, 8),
    {
      id: rootId,
      content: "A specific decision marker with intentionally neutral lexical evidence.",
      vectorContent: "Specific decision marker.",
      denseScore: 0.81,
      baselineScore: 0.86,
      createdAt: OLD,
    },
    {
      id: popularId,
      content: "A popular broad summary with no authoritative answer.",
      vectorContent: "Popular broad summary.",
      denseScore: 0.8,
      baselineScore: 0.85,
      recallCount: 10_000,
      createdAt: OLD,
    },
    {
      id: `${prefix}-keyword-leader`,
      content: keywordText,
      baselineScore: 0.99,
      keywordCandidate: true,
      createdAt: OLD,
    },
    { id: answerId, content: answer, createdAt: OLD },
  ];
}

function longParentCandidates(
  prefix: string,
  rootId: string,
  answerId: string,
  pollutedSection: string,
  localChunk: string,
  answer: string,
): CandidateFixture[] {
  return [
    ...directCandidates(prefix),
    {
      id: rootId,
      content: `${pollutedSection} ${"Unrelated archival material. ".repeat(30)}`,
      vectorContent: localChunk,
      denseScore: 0.75,
      baselineScore: 0.7,
      createdAt: OLD,
    },
    { id: answerId, content: answer, createdAt: OLD },
  ];
}

function weakNeighborCandidates(
  prefix: string,
  authoritativeDirectId: string,
  rootId: string,
  weakId: string,
  weakContent: string,
): CandidateFixture[] {
  const direct = directCandidates(prefix, authoritativeDirectId);
  const root: CandidateFixture = {
    id: rootId,
    denseScore: 0.6,
    baselineScore: 0.6,
    content: "A weakly relevant root marker.",
    vectorContent: "A weakly relevant root marker.",
    createdAt: OLD,
  };
  return [...direct, root, { id: weakId, content: weakContent, createdAt: OLD }];
}

function absentCandidates(prefix: string, answerId: string): CandidateFixture[] {
  return [...directCandidates(prefix), { id: answerId, content: "Authoritative evidence outside the raw pool.", createdAt: OLD }];
}

const decided = (sourceId: string, targetId: string): EdgeFixture => ({
  sourceId,
  targetId,
  type: "decided",
  weight: 1,
  provenance: "explicit",
});

const weak = (sourceId: string, targetId: string): EdgeFixture => ({
  sourceId,
  targetId,
  type: "relates_to",
  weight: 0.1,
  provenance: "inferred",
});

/**
 * Frozen before benchmark assertions or constant tuning. Authoritative and root IDs are
 * intentionally literal in this manifest so the implementation cannot manufacture its
 * own expected answers. The (domain index + shape index) parity assignment yields ten
 * cases per split and places every failure shape in both splits.
 */
export const ROOT_QUALITY_CASES: readonly RootQualityCase[] = [
  {
    split: "development", domain: "personal", failureShape: "crowded-lexical-root",
    query: "why did the cedar itinerary change", intent: "causal",
    candidates: crowdedCandidates("personal-crowded", "personal-crowded-root", "personal-crowded-answer", "cedar itinerary change", "The cedar itinerary changed because the venue moved the reservation.", "cedar itinerary"),
    edges: [decided("personal-crowded-root", "personal-crowded-answer")],
    authoritativeIds: ["personal-crowded-answer"], acceptableRootIds: ["personal-crowded-root"], candidateAvailable: true,
  },
  {
    split: "holdout", domain: "personal", failureShape: "popular-broad-summary",
    query: "why did the harbor schedule change", intent: "causal",
    candidates: popularityCandidates("personal-popular", "personal-popular-root", "personal-popular-summary", "personal-popular-answer", "The harbor schedule changed because the permit window moved.", "harbor schedule change"),
    edges: [decided("personal-popular-root", "personal-popular-answer")],
    authoritativeIds: ["personal-popular-answer"], acceptableRootIds: ["personal-popular-root"], candidateAvailable: true,
  },
  {
    split: "development", domain: "personal", failureShape: "long-parent-pollution",
    query: "why did the maple booking change", intent: "causal",
    candidates: longParentCandidates("personal-long", "personal-long-root", "personal-long-answer", "Maple booking change venue", "maple planning context", "The booking changed venue after the maple room became unavailable."),
    edges: [decided("personal-long-root", "personal-long-answer")],
    authoritativeIds: ["personal-long-answer"], acceptableRootIds: ["personal-long-root"], candidateAvailable: true,
  },
  {
    split: "holdout", domain: "personal", failureShape: "weak-generic-neighbor",
    query: "neighborhood budget status review planning update summary", intent: "direct",
    candidates: weakNeighborCandidates("personal-weak", "personal-weak-direct-answer", "personal-weak-root", "personal-weak-neighbor", "Neighborhood budget overview."),
    edges: [weak("personal-weak-root", "personal-weak-neighbor")],
    authoritativeIds: ["personal-weak-direct-answer"], acceptableRootIds: ["personal-weak-root"], candidateAvailable: true,
  },
  {
    split: "development", domain: "personal", failureShape: "absent-cluster-control",
    query: "what happened before the studio move", intent: "chronology",
    candidates: absentCandidates("personal-absent", "personal-absent-answer"), edges: [],
    authoritativeIds: ["personal-absent-answer"], acceptableRootIds: [], candidateAvailable: false,
  },

  {
    split: "holdout", domain: "enterprise", failureShape: "crowded-lexical-root",
    query: "why did the cobalt runtime change", intent: "causal",
    candidates: crowdedCandidates("enterprise-crowded", "enterprise-crowded-root", "enterprise-crowded-answer", "cobalt runtime change", "The cobalt runtime changed because the support contract required it.", "cobalt runtime"),
    edges: [decided("enterprise-crowded-root", "enterprise-crowded-answer")],
    authoritativeIds: ["enterprise-crowded-answer"], acceptableRootIds: ["enterprise-crowded-root"], candidateAvailable: true,
  },
  {
    split: "development", domain: "enterprise", failureShape: "popular-broad-summary",
    query: "current atlas support model", intent: "current",
    candidates: popularityCandidates("enterprise-popular", "enterprise-popular-root", "enterprise-popular-summary", "enterprise-popular-answer", "The current atlas support model assigns a named rotation owner.", "atlas support model"),
    edges: [decided("enterprise-popular-root", "enterprise-popular-answer")],
    authoritativeIds: ["enterprise-popular-answer"], acceptableRootIds: ["enterprise-popular-root"], candidateAvailable: true,
  },
  {
    split: "holdout", domain: "enterprise", failureShape: "long-parent-pollution",
    query: "what happened before the ember migration", intent: "chronology",
    candidates: longParentCandidates("enterprise-long", "enterprise-long-root", "enterprise-long-answer", "Ember migration before sequence", "ember planning context", "Before the ember migration, the compatibility audit closed its final exception."),
    edges: [decided("enterprise-long-root", "enterprise-long-answer")],
    authoritativeIds: ["enterprise-long-answer"], acceptableRootIds: ["enterprise-long-root"], candidateAvailable: true,
  },
  {
    split: "development", domain: "enterprise", failureShape: "weak-generic-neighbor",
    query: "delivery control status review planning update summary", intent: "direct",
    candidates: weakNeighborCandidates("enterprise-weak", "enterprise-weak-direct-answer", "enterprise-weak-root", "enterprise-weak-neighbor", "Delivery control overview."),
    edges: [weak("enterprise-weak-root", "enterprise-weak-neighbor")],
    authoritativeIds: ["enterprise-weak-direct-answer"], acceptableRootIds: ["enterprise-weak-root"], candidateAvailable: true,
  },
  {
    split: "holdout", domain: "enterprise", failureShape: "absent-cluster-control",
    query: "why did the orchid queue change", intent: "causal",
    candidates: absentCandidates("enterprise-absent", "enterprise-absent-answer"), edges: [],
    authoritativeIds: ["enterprise-absent-answer"], acceptableRootIds: [], candidateAvailable: false,
  },

  {
    split: "development", domain: "product", failureShape: "crowded-lexical-root",
    query: "why did the beacon launch change", intent: "causal",
    candidates: crowdedCandidates("product-crowded", "product-crowded-root", "product-crowded-answer", "beacon launch change", "The beacon launch changed because the review window opened earlier.", "beacon launch"),
    edges: [decided("product-crowded-root", "product-crowded-answer")],
    authoritativeIds: ["product-crowded-answer"], acceptableRootIds: ["product-crowded-root"], candidateAvailable: true,
  },
  {
    split: "holdout", domain: "product", failureShape: "popular-broad-summary",
    query: "current compass onboarding direction", intent: "current",
    candidates: popularityCandidates("product-popular", "product-popular-root", "product-popular-summary", "product-popular-answer", "The current compass onboarding direction starts with a guided workspace.", "compass onboarding direction"),
    edges: [decided("product-popular-root", "product-popular-answer")],
    authoritativeIds: ["product-popular-answer"], acceptableRootIds: ["product-popular-root"], candidateAvailable: true,
  },
  {
    split: "development", domain: "product", failureShape: "long-parent-pollution",
    query: "what happened before the mosaic rollout", intent: "chronology",
    candidates: longParentCandidates("product-long", "product-long-root", "product-long-answer", "Mosaic rollout before sequence", "mosaic planning context", "Before the mosaic rollout, the activation checklist passed its final trial."),
    edges: [decided("product-long-root", "product-long-answer")],
    authoritativeIds: ["product-long-answer"], acceptableRootIds: ["product-long-root"], candidateAvailable: true,
  },
  {
    split: "holdout", domain: "product", failureShape: "weak-generic-neighbor",
    query: "product review status planning update summary roadmap", intent: "direct",
    candidates: weakNeighborCandidates("product-weak", "product-weak-direct-answer", "product-weak-root", "product-weak-neighbor", "Product review overview."),
    edges: [weak("product-weak-root", "product-weak-neighbor")],
    authoritativeIds: ["product-weak-direct-answer"], acceptableRootIds: ["product-weak-root"], candidateAvailable: true,
  },
  {
    split: "development", domain: "product", failureShape: "absent-cluster-control",
    query: "current pricing experiment outcome", intent: "current",
    candidates: absentCandidates("product-absent", "product-absent-answer"), edges: [],
    authoritativeIds: ["product-absent-answer"], acceptableRootIds: [], candidateAvailable: false,
  },

  {
    split: "holdout", domain: "architecture", failureShape: "crowded-lexical-root",
    query: "why did the quartz cache design change", intent: "causal",
    candidates: crowdedCandidates("architecture-crowded", "architecture-crowded-root", "architecture-crowded-answer", "quartz cache design", "The quartz cache design changed because invalidation needed one owner.", "quartz cache"),
    edges: [decided("architecture-crowded-root", "architecture-crowded-answer")],
    authoritativeIds: ["architecture-crowded-answer"], acceptableRootIds: ["architecture-crowded-root"], candidateAvailable: true,
  },
  {
    split: "development", domain: "architecture", failureShape: "popular-broad-summary",
    query: "current delta indexing approach", intent: "current",
    candidates: popularityCandidates("architecture-popular", "architecture-popular-root", "architecture-popular-summary", "architecture-popular-answer", "The current delta indexing approach writes one normalized projection.", "delta indexing approach"),
    edges: [decided("architecture-popular-root", "architecture-popular-answer")],
    authoritativeIds: ["architecture-popular-answer"], acceptableRootIds: ["architecture-popular-root"], candidateAvailable: true,
  },
  {
    split: "holdout", domain: "architecture", failureShape: "long-parent-pollution",
    query: "what happened before the lattice storage migration", intent: "chronology",
    candidates: longParentCandidates("architecture-long", "architecture-long-root", "architecture-long-answer", "Lattice storage migration before", "lattice planning context", "Before the lattice storage migration, the restore rehearsal verified the archive."),
    edges: [decided("architecture-long-root", "architecture-long-answer")],
    authoritativeIds: ["architecture-long-answer"], acceptableRootIds: ["architecture-long-root"], candidateAvailable: true,
  },
  {
    split: "development", domain: "architecture", failureShape: "weak-generic-neighbor",
    query: "platform architecture status planning update summary review", intent: "direct",
    candidates: weakNeighborCandidates("architecture-weak", "architecture-weak-direct-answer", "architecture-weak-root", "architecture-weak-neighbor", "Platform architecture overview."),
    edges: [weak("architecture-weak-root", "architecture-weak-neighbor")],
    authoritativeIds: ["architecture-weak-direct-answer"], acceptableRootIds: ["architecture-weak-root"], candidateAvailable: true,
  },
  {
    split: "holdout", domain: "architecture", failureShape: "absent-cluster-control",
    query: "why did the horizon protocol change", intent: "causal",
    candidates: absentCandidates("architecture-absent", "architecture-absent-answer"), edges: [],
    authoritativeIds: ["architecture-absent-answer"], acceptableRootIds: [], candidateAvailable: false,
  },
];

function deepFreeze<T>(value: T): T {
  if (value && typeof value === "object" && !Object.isFrozen(value)) {
    Object.freeze(value);
    for (const child of Object.values(value as Record<string, unknown>)) deepFreeze(child);
  }
  return value;
}

deepFreeze(ROOT_QUALITY_CASES);
