import { describe, expect, it } from "vitest";
import {
  graphSeedLimit,
  queryCoverage,
  relatedSlotLimit,
  scoreLinkedEvidence,
} from "../../src/recall/neighborhood";

describe("graph-aware recall neighborhood policy", () => {
  it("bounds graph seeds at the existing overfetch scale and Vectorize ceiling", () => {
    expect(graphSeedLimit(5, 40)).toBe(15);
    expect(graphSeedLimit(20, 90)).toBe(50);
    expect(graphSeedLimit(5, 8)).toBe(8);
  });

  it("reserves no related slot for tiny result sets and at most two otherwise", () => {
    expect(relatedSlotLimit(1)).toBe(0);
    expect(relatedSlotLimit(2)).toBe(0);
    expect(relatedSlotLimit(3)).toBe(1);
    expect(relatedSlotLimit(5)).toBe(1);
    expect(relatedSlotLimit(6)).toBe(2);
    expect(relatedSlotLimit(20)).toBe(2);
  });
});

describe("deterministic linked-evidence scoring", () => {
  const base = {
    parentScore: 0.8,
    queryTokens: ["anniversary", "childcare"],
    corpus: { df: null, total: null },
    hop: 1,
    edgeWeight: 1,
    provenance: "explicit" as const,
    hopDecay: 0.6,
    replacementCoverage: 0,
    intent: "direct" as const,
    edgeType: "relates_to" as const,
  };

  it("rejects an explicit continuation when only the root carries query context", () => {
    const result = scoreLinkedEvidence({
      ...base,
      parentContent: "We changed the anniversary plan because the trip was impractical",
      content: "Chateau Elan solved the sitter constraint",
    });

    expect(result.eligible).toBe(false);
    expect(result.score).toBe(0);
    expect(result.rejection).toBe("no-linked-evidence");
  });

  it("rejects an inferred neighbor that contributes no query evidence", () => {
    const result = scoreLinkedEvidence({
      ...base,
      parentContent: "Enterprise architecture review",
      content: "Unrelated grocery list",
      queryTokens: ["architecture", "review"],
      provenance: "inferred",
    });

    expect(result.eligible).toBe(false);
    expect(result.score).toBe(0);
    expect(result.rejection).toBe("no-linked-evidence");
  });

  it("ranks a rare linked term above an equally connected common-term match", () => {
    const corpus = {
      df: new Map([
        ["platform", 90],
        ["dotnet", 2],
      ]),
      total: 100,
    };
    const common = scoreLinkedEvidence({
      ...base,
      parentContent: "The platform backend changed",
      content: "The platform supports teams",
      queryTokens: ["platform", "dotnet"],
      corpus,
    });
    const rare = scoreLinkedEvidence({
      ...base,
      parentContent: "The platform backend changed",
      content: "Dotnet aligned with enterprise support",
      queryTokens: ["platform", "dotnet"],
      corpus,
    });

    expect(rare.eligible).toBe(true);
    expect(rare.score).toBeGreaterThan(common.score);
  });

  it("applies hop decay so otherwise equal one-hop evidence outranks two-hop evidence", () => {
    const oneHop = scoreLinkedEvidence({
      ...base,
      parentContent: "Anniversary plan",
      content: "Anniversary childcare constraint",
      hop: 1,
    });
    const twoHop = scoreLinkedEvidence({
      ...base,
      parentContent: "Anniversary plan",
      content: "Anniversary childcare constraint",
      hop: 2,
    });

    expect(oneHop.score).toBeGreaterThan(twoHop.score);
  });

  it("rejects a linked memory that matches only a common term", () => {
    const score = scoreLinkedEvidence({
      ...base,
      parentContent: "Enterprise platform review",
      content: "The platform home page was redesigned",
      queryTokens: ["platform", "ledger"],
      corpus: { df: new Map([["platform", 90], ["ledger", 2]]), total: 100 },
      replacementCoverage: 0.2,
      intent: "causal",
      edgeType: "relates_to",
    });

    expect(score.eligible).toBe(false);
    expect(score.rejection).toBe("weak-neighborhood");
  });

  it("rejects linked evidence that only contains a query token as a substring", () => {
    const score = scoreLinkedEvidence({
      ...base,
      parentContent: "Ledger status changed",
      content: "The ledgering job completed",
      queryTokens: ["ledger", "status"],
      corpus: { df: new Map([["ledger", 90], ["status", 80]]), total: 100 },
      replacementCoverage: 0,
    });

    expect(score.eligible).toBe(false);
    expect(score.rejection).toBe("weak-neighborhood");
  });

  it("rejects a favorable one-token substring-only neighborhood", () => {
    const score = scoreLinkedEvidence({
      ...base,
      parentScore: 1,
      parentContent: "",
      content: "platforms",
      queryTokens: ["platform"],
      corpus: { df: new Map([["platform", 90]]), total: 100 },
      hop: 0,
      hopDecay: 1,
      edgeWeight: 1,
      intent: "causal",
      edgeType: "decided",
    });

    expect(score.eligible).toBe(false);
    expect(score.rejection).toBe("weak-neighborhood");
  });

  const qualifying = {
    ...base,
    parentContent: "The backend direction changed",
    content: "Dotnet matched enterprise support skills",
    queryTokens: ["backend", "dotnet"],
    corpus: { df: new Map([["backend", 30], ["dotnet", 2]]), total: 100 },
    intent: "causal" as const,
    edgeType: "decided" as const,
  };

  it("accepts complementary rare evidence through a compatible decision edge", () => {
    const score = scoreLinkedEvidence({ ...qualifying, replacementCoverage: 0.1 });

    expect(score.eligible).toBe(true);
    expect(score.coverageGain).toBeGreaterThan(0.1);
  });

  it("abstains when the neighborhood does not improve on the replaced direct evidence", () => {
    const score = scoreLinkedEvidence({ ...qualifying, replacementCoverage: 1 });

    expect(score.eligible).toBe(false);
    expect(score.rejection).toBe("no-evidence-gain");
  });
});

describe("query coverage details", () => {
  it("labels only exact rare token matches as high-IDF", () => {
    const corpus = { df: new Map([["dotnet", 10]]), total: 100 };

    expect(queryCoverage("Dotnet supports the backend", ["dotnet"], corpus)).toEqual({
      score: 1,
      exactHighIdf: true,
    });
    expect(queryCoverage("adotnetservice", ["dotnet"], corpus)).toEqual({
      score: 0.25,
      exactHighIdf: false,
    });
  });
});
