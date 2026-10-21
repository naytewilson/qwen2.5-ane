import Foundation
import NaturalLanguage

// ─────────────────────────────────────────────────────────────────────────────
// PromptInjector — Dynamic Semantic Prompt Injection
//
// Opt-in: loads a lessons knowledge base from the path in QWEN_LESSONS_PATH
// (sections separated by "\n---\n"). On each request, ranks lessons by semantic
// similarity to the incoming prompt using NLEmbedding.sentenceEmbedding, then
// injects the top-k relevant lessons into the system prompt before tokenization.
// When QWEN_LESSONS_PATH is unset the injector is inert.
//
// Designed to be zero-latency: lessons are mtime-cached, and NLEmbedding is
// a native Apple framework (no model load overhead).
// ─────────────────────────────────────────────────────────────────────────────

struct PromptInjector {

    // MARK: - Config
    // Opt-in only. Point QWEN_LESSONS_PATH at a lessons file to enable injection.
    static var lessonsPath: String? {
        guard let path = ProcessInfo.processInfo.environment["QWEN_LESSONS_PATH"], !path.isEmpty else {
            return nil
        }
        return path
    }
    static let topK         = 3
    static let minScore     = 0.35   // cosine similarity threshold — tuned: greetings score <0.35, domain queries >0.35
    static let sectionSep   = "\n---\n"

    // MARK: - Lesson cache (auto-reloads on file modification)
    // nonisolated(unsafe): this binary is single-shot (one request per invocation),
    // so there is no actual concurrent access to this cache.
    nonisolated(unsafe) private static var _cache: (mtime: Date, lessons: [String]) = (.distantPast, [])

    private static var lessons: [String] {
        let fm = FileManager.default
        guard let lessonsPath,
              let attrs = try? fm.attributesOfItem(atPath: lessonsPath),
              let mtime = attrs[.modificationDate] as? Date
        else { return [] }

        if mtime == _cache.mtime { return _cache.lessons }

        guard let raw = try? String(contentsOfFile: lessonsPath, encoding: .utf8) else { return [] }
        let parsed = raw
            .components(separatedBy: sectionSep)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        _cache = (mtime, parsed)
        return parsed
    }

    // MARK: - Public API

    /// Returns an enriched system prompt with injected relevant context, or the
    /// original system string unchanged if no lessons match above minScore.
    static func inject(query: String, into system: String?) -> String {
        let base = system ?? "You are a helpful assistant."
        let bank = lessons
        guard !bank.isEmpty else { return base }

        let ranked = rank(query: query, against: bank)
        let top = ranked
            .prefix(topK)
            .filter { $0.score >= minScore }
            .map { $0.text }

        guard !top.isEmpty else { return base }

        let block = top.joined(separator: "\n\n")
        return "\(base)\n\n[Relevant Context]\n\(block)"
    }

    /// Returns the count of lessons that would be injected for a given query.
    static func previewInjection(query: String) -> [(text: String, score: Double)] {
        let bank = lessons
        guard !bank.isEmpty else { return [] }
        return rank(query: query, against: bank)
            .prefix(topK)
            .filter { $0.score >= minScore }
            .map { ($0.text, $0.score) }
    }

    // MARK: - Ranking

    private struct Scored { let text: String; let score: Double }

    private static func rank(query: String, against bank: [String]) -> [Scored] {
        // Score using only the first line of each lesson (the label/title).
        // Full lesson paragraphs dilute the embedding vector — short anchor lines score much higher.
        // The full text is still what gets injected into the system prompt.
        if let emb = NLEmbedding.sentenceEmbedding(for: .english),
           let qVec = emb.vector(for: query) {
            return bank.compactMap { lesson -> Scored? in
                let anchor = lesson.components(separatedBy: ".").first.map { String($0.prefix(120)) } ?? lesson // first sentence, capped
                guard let lVec = emb.vector(for: anchor) else { return nil }
                return Scored(text: lesson, score: cosine(qVec, lVec))
            }.sorted { $0.score > $1.score }
        }

        // Fallback: Jaccard on first line only
        return bank.map { lesson in
            let anchor = String(lesson.prefix(while: { $0 != "\n" }))
            return Scored(text: lesson, score: jaccard(query, anchor))
        }.sorted { $0.score > $1.score }
    }

    // MARK: - Math helpers

    private static func cosine(_ a: [Double], _ b: [Double]) -> Double {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        var dot = 0.0, na = 0.0, nb = 0.0
        for i in 0..<a.count {
            dot += a[i] * b[i]
            na  += a[i] * a[i]
            nb  += b[i] * b[i]
        }
        let denom = sqrt(na) * sqrt(nb)
        return denom > 0 ? dot / denom : 0
    }

    private static func jaccard(_ a: String, _ b: String) -> Double {
        let tokA = Set(a.lowercased().components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty })
        let tokB = Set(b.lowercased().components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty })
        guard !tokA.isEmpty || !tokB.isEmpty else { return 0 }
        let inter = Double(tokA.intersection(tokB).count)
        let union = Double(tokA.union(tokB).count)
        return union > 0 ? inter / union : 0
    }
}
