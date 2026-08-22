export type EvidenceSlotSource = "omitted-root" | "related";

export interface EvidenceSlotCandidate {
  id: string;
  coverage: number;
  exactHighIdf: boolean;
  exactMatchCount: number;
  metadataAlignment: number;
  score: number;
  source: EvidenceSlotSource;
}

const compareEvidence = (a: EvidenceSlotCandidate, b: EvidenceSlotCandidate) =>
  b.coverage - a.coverage
  || Number(b.exactHighIdf) - Number(a.exactHighIdf)
  || b.exactMatchCount - a.exactMatchCount
  || b.metadataAlignment - a.metadataAlignment
  || b.score - a.score
  || a.id.localeCompare(b.id);

/**
 * Chooses one final-slot candidate using only evidence already computed during
 * the current recall. The gate is deliberately relative to the result it would
 * displace, so it is independent of any brain's score distribution.
 */
export function chooseEvidenceSlot(
  replacementCoverage: number,
  candidates: readonly EvidenceSlotCandidate[],
): EvidenceSlotCandidate | undefined {
  return candidates
    .filter(candidate => candidate.coverage > replacementCoverage)
    .filter(candidate => candidate.exactHighIdf || candidate.exactMatchCount >= 2)
    .slice()
    .sort(compareEvidence)[0];
}
