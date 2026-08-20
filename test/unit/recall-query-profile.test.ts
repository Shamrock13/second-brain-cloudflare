import { describe, expect, it } from "vitest";
import {
  buildQueryProfile,
  edgeIntentCompatibility,
  embeddingInput,
} from "../../src/recall/query-profile";

describe("recall query profile", () => {
  it.each([
    ["why did the backend change", "causal"],
    ["what happened before the launch", "chronology"],
    ["what is the current platform direction", "current"],
    ["enterprise platform architecture", "direct"],
  ] as const)("classifies %s", (query, intent) => {
    expect(buildQueryProfile(query, { query: "backend platform", df: null, total: null }).intent).toBe(intent);
  });

  it("keeps semantic and lexical representations separate", () => {
    const profile = buildQueryProfile(
      "why did we change the enterprise backend direction",
      { query: "enterprise backend", df: null, total: null },
    );
    expect(profile.semanticQuery).toBe("why did we change the enterprise backend direction");
    expect(profile.lexicalQuery).toBe("enterprise backend");
    expect(embeddingInput(profile, "distilled")).toBe("enterprise backend");
    expect(embeddingInput(profile, "semantic")).toBe(profile.semanticQuery);
    expect(embeddingInput(profile, "hybrid")).toBe(
      "why did we change the enterprise backend direction enterprise backend",
    );
  });

  it("uses edge types as soft intent compatibility", () => {
    expect(edgeIntentCompatibility("causal", "caused_by")).toBeGreaterThan(
      edgeIntentCompatibility("causal", "relates_to"),
    );
    expect(edgeIntentCompatibility("chronology", "follows")).toBeGreaterThan(
      edgeIntentCompatibility("chronology", "relates_to"),
    );
    expect(edgeIntentCompatibility("direct", "decided")).toBe(0.5);
  });
});
