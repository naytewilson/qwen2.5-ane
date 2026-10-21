import Foundation

public final class NativeTokenizer: @unchecked Sendable {
    private var vocab: [String: Int] = [:]
    private var mergeRankPairs: [UInt64: Int32] = [:]
    private var mergedTokenId: [UInt64: Int] = [:]
    private var reverseVocabBytes: [Int: [UInt8]] = [:]
    private let cacheLock = NSLock()
    private var bpeCache: [String: [Int]] = [:]

    private let specialTokenMap: [String: Int]
    private let specialIdMap: [Int: String]

    private let byteToUnicode: [UInt32]
    private let unicodeToByteArr: [UInt8?]
    private let regex: NSRegularExpression

    // Special tokens: IDs 151643-151664 (22 total)
    private static let SPECIAL_TOKENS: [(String, Int)] = [
        ("<|endoftext|>", 151643),
        ("<|im_start|>", 151644),
        ("<|im_end|>", 151645),
        ("<|object_ref_start|>", 151646),
        ("<|object_ref_end|>", 151647),
        ("<|box_start|>", 151648),
        ("<|box_end|>", 151649),
        ("<|quad_start|>", 151650),
        ("<|quad_end|>", 151651),
        ("<|vision_start|>", 151652),
        ("<|vision_end|>", 151653),
        ("<|vision_pad|>", 151654),
        ("<|image_pad|>", 151655),
        ("<|video_pad|>", 151656),
        ("<tool_call>", 151657),
        ("</tool_call>", 151658),
        ("<|fim_prefix|>", 151659),
        ("<|fim_middle|>", 151660),
        ("<|fim_suffix|>", 151661),
        ("<|fim_pad|>", 151662),
        ("<|repo_name|>", 151663),
        ("<|file_sep|>", 151664)
    ]

    public var vocabCount: Int { vocab.count }

    public init(tokenizerDirectory: String) throws {
        fputs("[NativeTokenizer] Initializing...\n", stderr)
        fflush(stderr)

        // Build GPT-2 byte encoding tables
        fputs("[NativeTokenizer] Building byte tables...\n", stderr)
        fflush(stderr)
        self.byteToUnicode = Self.buildByteToUnicode()
        self.unicodeToByteArr = Self.buildUnicodeToByteArr()

        // Build special token maps
        fputs("[NativeTokenizer] Building special token maps...\n", stderr)
        fflush(stderr)
        var specialMap: [String: Int] = [:]
        var reverseSpecialMap: [Int: String] = [:]
        for (token, id) in Self.SPECIAL_TOKENS {
            specialMap[token] = id
            reverseSpecialMap[id] = token
        }
        self.specialTokenMap = specialMap
        self.specialIdMap = reverseSpecialMap

        // Compile regex once (expensive operation)
        fputs("[NativeTokenizer] Compiling regex...\n", stderr)
        fflush(stderr)
        let tikTokenPattern = "(?i:'s|'t|'re|'ve|'m|'ll|'d)| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
        self.regex = try NSRegularExpression(pattern: tikTokenPattern, options: [])

        // Load vocab (sequential for now, optimized later)
        fputs("[NativeTokenizer] Loading vocab.json...\n", stderr)
        fflush(stderr)
        let vocabPath = "\(tokenizerDirectory)/vocab.json"
        let vocabData = try Data(contentsOf: URL(fileURLWithPath: vocabPath))
        self.vocab = try JSONDecoder().decode([String: Int].self, from: vocabData)
        fputs("[NativeTokenizer] Vocab loaded: \(vocab.count) tokens\n", stderr)
        fflush(stderr)

        // Load merges
        fputs("[NativeTokenizer] Loading merges.txt...\n", stderr)
        fflush(stderr)
        let mergesPath = "\(tokenizerDirectory)/merges.txt"
        let content = try String(contentsOfFile: mergesPath, encoding: .utf8)
        let normalizedContent = content.replacingOccurrences(of: "\r\n", with: "\n")
        let lines = normalizedContent.split(separator: "\n", omittingEmptySubsequences: true)
        
        self.mergeRankPairs.reserveCapacity(lines.count)
        self.mergedTokenId.reserveCapacity(lines.count)
        
        for (idx, line) in lines.enumerated() {
            let pair = String(line).trimmingCharacters(in: .newlines)
            let parts = pair.split(separator: " ")
            if parts.count == 2 {
                let token1 = String(parts[0])
                let token2 = String(parts[1])
                if let id1 = vocab[token1], let id2 = vocab[token2] {
                    let mergedToken = token1 + token2
                    if let mergedId = vocab[mergedToken] {
                        let key = (UInt64(id1) << 32) | UInt64(id2)
                        self.mergeRankPairs[key] = Int32(idx)
                        self.mergedTokenId[key] = mergedId
                    }
                }
            }
        }
        fputs("[NativeTokenizer] Merges loaded: \(mergeRankPairs.count) pairs\n", stderr)
        fflush(stderr)

        // Build reverseVocabBytes: pre-compute UTF-8 bytes for each token ID
        fputs("[NativeTokenizer] Building reverse vocab...\n", stderr)
        fflush(stderr)
        buildReverseVocab()
        fputs("[NativeTokenizer] Ready\n", stderr)
        fflush(stderr)
    }

    /// Standard GPT-2 bytes_to_unicode mapping (OpenAI algorithm)
    /// Maps 256 byte values to visible Unicode characters to avoid raw bytes in BPE
    private static func buildByteToUnicode() -> [UInt32] {
        var bs = [UInt8]()

        // Printable ASCII: 33-126 ("!" to "~")
        for byte in 33...126 {
            bs.append(UInt8(byte))
        }

        // Latin-1: 161-172
        for byte in 161...172 {
            bs.append(UInt8(byte))
        }

        // Latin-1: 174-255
        for byte in 174...255 {
            bs.append(UInt8(byte))
        }

        var cs = [Int]()
        cs.append(contentsOf: bs.map { Int($0) })

        // Fill remaining bytes 0-255 that aren't in bs, map to Unicode 256+
        var n = 0
        for b in 0..<256 {
            let byte = UInt8(b)
            if !bs.contains(byte) {
                bs.append(byte)
                cs.append(256 + n)
                n += 1
            }
        }

        // Build result table: result[byte] = Unicode codepoint
        var result = [UInt32](repeating: 0, count: 256)
        for (i, b) in bs.enumerated() {
            result[Int(b)] = UInt32(cs[i])
        }

        return result
    }

    /// Reverse Unicode-to-Byte mapping (properly inverts buildByteToUnicode)
    private static func buildUnicodeToByteArr() -> [UInt8?] {
        var table = [UInt8?](repeating: nil, count: 512)
        let byteToUnicode = buildByteToUnicode()
        for b in 0..<256 {
            let u = byteToUnicode[b]
            if Int(u) < table.count {
                table[Int(u)] = UInt8(b)
            }
        }
        return table
    }

    /// Pre-compute bytes for each vocab token ID
    private func buildReverseVocab() {
        for (token, id) in vocab {
            // Each token string contains Unicode codepoints from byteToUnicode
            // We need to convert those back to bytes
            var bytes = [UInt8]()
            for scalar in token.unicodeScalars {
                let val = scalar.value
                // Try to find the inverse mapping in unicodeToByteArr
                if val < UInt32(unicodeToByteArr.count) {
                    if let byte = unicodeToByteArr[Int(val)] {
                        bytes.append(byte)
                    } else if val < 256 {
                        // Direct mapping for ASCII-range values
                        bytes.append(UInt8(val))
                    }
                } else if val < 256 {
                    bytes.append(UInt8(val))
                }
            }
            reverseVocabBytes[id] = bytes
        }
    }

    /// BPE algorithm using flat array-backed linked list (O(1) merges) - Optimized Zero-Allocation
    private func bpeWord(_ byteEncodedWord: String) -> [Int] {
        cacheLock.lock()
        let cached = bpeCache[byteEncodedWord]
        cacheLock.unlock()
        if let res = cached {
            return res
        }

        let chars = Array(byteEncodedWord)
        if chars.isEmpty {
            return []
        }

        // BPENode struct: value type, no ARC overhead, and no String allocations
        struct BPENode {
            var tokenId: Int
            var prev: Int  // index of previous node (-1 if head)
            var next: Int  // index of next node (-1 if tail)
        }

        // Pre-allocate contiguous array of BPENodes
        var nodes = [BPENode]()
        nodes.reserveCapacity(chars.count)
        for i in 0..<chars.count {
            let charStr = String(chars[i])
            guard let tokenId = vocab[charStr] else {
                continue
            }
            nodes.append(BPENode(
                tokenId: tokenId,
                prev: i > 0 ? i - 1 : -1,
                next: i < chars.count - 1 ? i + 1 : -1
            ))
        }

        let headIdx = 0

        // Build initial pair ranks dictionary (key: Packed UInt64 indices, value: merge rank)
        // Key format: (UInt64(i) << 32) | UInt64(j)
        var pairRanks = [UInt64: Int32]()

        for i in 0..<nodes.count - 1 {
            if nodes[i].next != -1 {
                let j = nodes[i].next
                let id1 = nodes[i].tokenId
                let id2 = nodes[j].tokenId
                let pairKey = (UInt64(id1) << 32) | UInt64(id2)
                if let rank = mergeRankPairs[pairKey] {
                    let key = (UInt64(i) << 32) | UInt64(j)
                    pairRanks[key] = rank
                }
            }
        }

        // BPE merge loop: O(1) index updates, no array shifts, 100% zero string allocations!
        while !pairRanks.isEmpty {
            // Find pair with minimum rank (best match to merge)
            let (bestKey, _) = pairRanks.min { $0.value < $1.value }!
            let i = Int(bestKey >> 32)
            let j = Int(bestKey & 0xFFFFFFFF)

            let pi = nodes[i].prev
            let ni = nodes[j].next

            // Merge: i absorbs j, j becomes dead
            let id1 = nodes[i].tokenId
            let id2 = nodes[j].tokenId
            let pairKey = (UInt64(id1) << 32) | UInt64(id2)
            guard let mergedId = mergedTokenId[pairKey] else {
                break
            }
            nodes[i].tokenId = mergedId
            nodes[i].next = ni

            if ni != -1 {
                nodes[ni].prev = i
            }

            // Remove the merged pair from tracking
            pairRanks.removeValue(forKey: bestKey)

            // Remove old (j, ni) since j is dead
            if ni != -1 {
                let keyJN = (UInt64(j) << 32) | UInt64(ni)
                pairRanks.removeValue(forKey: keyJN)
            }

            // Update pair (pi, i)
            if pi != -1 {
                let keyPiI = (UInt64(pi) << 32) | UInt64(i)
                pairRanks.removeValue(forKey: keyPiI)
                let idPi = nodes[pi].tokenId
                let idI = nodes[i].tokenId
                let pairKeyPiI = (UInt64(idPi) << 32) | UInt64(idI)
                if let rank = mergeRankPairs[pairKeyPiI] {
                    pairRanks[keyPiI] = rank
                }
            }

            // Update pair (i, ni)
            if ni != -1 {
                let keyINi = (UInt64(i) << 32) | UInt64(ni)
                pairRanks.removeValue(forKey: keyINi)
                let idI = nodes[i].tokenId
                let idNi = nodes[ni].tokenId
                let pairKeyINi = (UInt64(idI) << 32) | UInt64(idNi)
                if let rank = mergeRankPairs[pairKeyINi] {
                    pairRanks[keyINi] = rank
                }
            }
        }

        // Walk the final linked list and collect token IDs
        var result = [Int]()
        var idx = headIdx
        while idx != -1 {
            result.append(nodes[idx].tokenId)
            idx = nodes[idx].next
        }

        cacheLock.lock()
        bpeCache[byteEncodedWord] = result
        cacheLock.unlock()
        return result
    }

    /// Tokenize text into token IDs, correctly handling special tokens anywhere in the text
    public func tokenize(_ text: String) -> [Int] {
        // 1. NFC normalization
        let normalizedText = text.precomposedStringWithCanonicalMapping
        
        var result = [Int]()
        var currentText = normalizedText
        
        while !currentText.isEmpty {
            // Find the earliest occurrence of any special token
            var earliestIndex: String.Index? = nil
            var matchedSpecial: String? = nil
            var matchedId: Int? = nil
            
            for (specialToken, id) in specialTokenMap {
                if let range = currentText.range(of: specialToken) {
                    if earliestIndex == nil || range.lowerBound < earliestIndex! {
                        earliestIndex = range.lowerBound
                        matchedSpecial = specialToken
                        matchedId = id
                    }
                }
            }
            
            if let index = earliestIndex, let special = matchedSpecial, let id = matchedId {
                // We found a special token. First tokenize the text before it.
                let prefix = String(currentText[..<index])
                if !prefix.isEmpty {
                    result.append(contentsOf: tokenizeRegular(prefix))
                }
                
                // Append the special token ID
                result.append(id)
                
                // Suffix is after the special token
                let suffixStart = currentText.index(index, offsetBy: special.count)
                currentText = String(currentText[suffixStart...])
            } else {
                // No special tokens left in the text
                result.append(contentsOf: tokenizeRegular(currentText))
                break
            }
        }
        
        return result
    }

    /// Tokenize regular text (no special tokens) using regex and BPE
    private func tokenizeRegular(_ text: String) -> [Int] {
        var result = [Int]()
        let nsString = text as NSString
        let matches = regex.matches(in: text, range: NSRange(location: 0, length: nsString.length))

        for match in matches {
            let matchStr = nsString.substring(with: match.range)

            // Byte-level encode
            let byteEncoded = encodeToByteString(matchStr)

            // BPE merge
            let ids = bpeWord(byteEncoded)
            result.append(contentsOf: ids)
        }

        return result
    }

    /// Encode string to byte representation using GPT-2 scheme
    private func encodeToByteString(_ str: String) -> String {
        var result = ""
        let utf8Bytes = Array(str.utf8)
        for byte in utf8Bytes {
            let unicode = byteToUnicode[Int(byte)]
            if let scalar = UnicodeScalar(unicode) {
                result.append(Character(scalar))
            }
        }
        return result
    }

    /// Decode token IDs back to text
    public func decode(tokens: [Int], skipSpecialTokens: Bool = true) -> String {
        var bytes = [UInt8]()
        bytes.reserveCapacity(tokens.count * 4)

        for id in tokens {
            if let special = specialIdMap[id] {
                if !skipSpecialTokens {
                    bytes.append(contentsOf: special.utf8)
                }
            } else if let tokenBytes = reverseVocabBytes[id] {
                bytes.append(contentsOf: tokenBytes)
            }
        }

        return String(bytes: bytes, encoding: .utf8) ?? ""
    }

    /// Format prompt using Qwen2.5 ChatML template
    public func formatChatML(system: String?, user: String) -> String {
        let sys = system ?? "You are a helpful assistant."
        return "<|im_start|>system\n\(sys)<|im_end|>\n<|im_start|>user\n\(user)<|im_end|>\n<|im_start|>assistant\n"
    }
}
