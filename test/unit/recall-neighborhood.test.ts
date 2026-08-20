import { describe, expect, it } from "vitest";
import {
  graphSeedLimit,
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
  };

  it("rejects an explicit continuation when only the root carries query context", () => {
    const result = scoreLinkedEvidence({
      ...base,
      parentContent: "We changed the anniversary plan because the trip was impractical",
      content: "Chateau Elan solved the sitter constraint",
    });

    expect(result).toEqual({ eligible: false, score: 0 });
  });

  it("rejects an inferred neighbor that contributes no query evidence", () => {
    const result = scoreLinkedEvidence({
      ...base,
      parentContent: "Enterprise architecture review",
      content: "Unrelated grocery list",
      queryTokens: ["architecture", "review"],
      provenance: "inferred",
    });

    expect(result).toEqual({ eligible: false, score: 0 });
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
      content: "Childcare constraint",
      hop: 1,
    });
    const twoHop = scoreLinkedEvidence({
      ...base,
      parentContent: "Anniversary plan",
      content: "Childcare constraint",
      hop: 2,
    });

    expect(oneHop.score).toBeGreaterThan(twoHop.score);
  });
});
