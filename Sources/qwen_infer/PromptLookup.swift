// PromptLookup.swift — n-gram prompt-lookup candidate generator (experimental, off by default)
// =============================================================================
// Prompt lookup proposes continuation tokens by suffix-matching the context. Why it suits the ANE:
//   • ZERO extra model weights, ZERO conv-budget cost (vs Medusa heads)
//   • fixed K-candidate batch → fits the existing batch4 `verify` function cleanly
//     (vs Jacobi's variable-shape window, which fights the ANE static graph)
//   • candidates verified in ONE batched ANE dispatch (rides the seq-amortization
//     proven in model.mil: infer(seq1) & prefill(seq4) share 212 identical convs)
//   • published speedups are task-dependent; none is measured or claimed for this build
//
// This is SELF-CONTAINED — no dependency on the pruned DraftEngine/NeuralDraft/
// SpeculativeEngine scaffolding. It only produces candidate tokens; CortexEngine's
// existing _verifyCandidatesUnlocked + acceptMedusaDraftsUnlocked (generic, takes any
// `drafts`) + the replay-commit block (generate loop ~L631-668) do the rest.
//
// Why this ISN'T the "dead" n-gram spec result: that 15%-accept number was n-gram as a
// general draft on OPEN-ENDED PROSE (worst case), with a slow batch64 verify. Prompt-
// lookup targets input-grounded generation where the output echoes the context — where
// suffix-matching the prior tokens predicts the continuation well.
// =============================================================================

import Foundation

/// Generates speculative continuation tokens by matching the longest suffix of recent
/// history against earlier occurrences in the same sequence (prompt + generated tokens),
/// then returning whatever followed the matched window. ~microseconds, pure CPU.
public struct PromptLookup {
    /// Longest n-gram suffix to attempt to match. Larger = more selective (higher accept
    /// when it hits, fewer hits). 3 is the prompt-lookup-decoding default.
    public let maxN: Int

    public init(maxN: Int = 3) { self.maxN = max(1, maxN) }

    /// Predict the next `k` tokens following `history` by greedy longest-suffix lookahead.
    /// Returns up to `k` candidates (fewer, or empty, when no suffix matches — caller then
    /// simply gets a shorter/empty speculative batch and the verify still produces the
    /// primary token). These are the `drafts` fed to acceptMedusaDraftsUnlocked.
    public func candidates(history: [Int32], k: Int) -> [Int32] {
        guard k > 0, history.count >= 2 else { return [] }
        var drafts = [Int32]()
        drafts.reserveCapacity(k)
        // Greedily extend a coherent continuation: match suffix → take follower → append →
        // repeat. Repetitive/echoing context (the case prompt-lookup wins on) extends well.
        var work = history
        for _ in 0..<k {
            guard let next = predictNext(work) else { break }
            drafts.append(next)
            work.append(next)
        }
        return drafts
    }

    /// Find the most-recent earlier occurrence of the longest possible suffix of `h`,
    /// and return the token that followed it. Tries longest suffix first (most selective).
    private func predictNext(_ h: [Int32]) -> Int32? {
        let upper = h.count
        guard upper >= 1 else { return nil }
        let maxLen = min(maxN, upper)
        var n = maxLen
        while n >= 1 {
            let qStart = upper - n
            // Walk backwards so the MOST RECENT match wins (best locality).
            var i = qStart - 1
            while i >= 0 {
                if matches(h, at: i, length: n, query: qStart) {
                    let follow = i + n
                    if follow < upper { return h[follow] }
                }
                i -= 1
            }
            n -= 1
        }
        return nil
    }

    @inline(__always)
    private func matches(_ h: [Int32], at start: Int, length: Int, query: Int) -> Bool {
        var k = 0
        while k < length {
            if h[start + k] != h[query + k] { return false }
            k += 1
        }
        return true
    }
}
