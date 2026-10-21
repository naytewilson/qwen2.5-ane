import Foundation

/// Token-level format enforcement for CortexEngine's decode loop.
///
/// Two mechanisms — no logit access required:
///   1. Prefix forcing: override the first N tokens with a mandatory sequence.
///      The model then continues naturally from that prefix (e.g. `{` → valid JSON).
///   2. Token banning: after an optional trigger pattern appears in output,
///      refuse any token in the ban set and replace with the next decode step.
///
/// Integration point: call `enforcer.next(rawToken:decodedText:)` after
/// `extractArgmax` and before `generatedIds.append / onToken`.
public struct FormatEnforcer {

    // MARK: - Preset constraints

    /// Force JSON object output: prefix with `{`, ban EOS until closing `}` seen.
    public static func json(tokenizer: NativeTokenizer) -> FormatEnforcer {
        let prefix = tokenizer.tokenize("{").map { Int32($0) }
        return FormatEnforcer(forcedPrefix: prefix)
    }

    /// Ban Python shebangs anywhere in output (no #!/usr/bin/env python).
    public static func noPython(tokenizer: NativeTokenizer) -> FormatEnforcer {
        // Encode representative shebang fragments to get their token IDs.
        // The tokenizer may merge these differently; we ban the individual pieces.
        let fragments = ["#!/usr/bin/env python", "#!/usr/bin/python",
                         "#!/usr/bin/env python3", "import os\nimport sys"]
        var banned = Set<Int32>()
        for frag in fragments {
            tokenizer.tokenize(frag).forEach { banned.insert(Int32($0)) }
        }
        return FormatEnforcer(bannedTokenIds: banned,
                              banViolationMessage: "[BLOCKED: Python shebang forbidden in ~/bin/. Use bash + jq or Swift.]")
    }

    // MARK: - State

    /// Tokens to emit before the model's own output. Consumed one per step.
    private var forcedPrefix: [Int32]

    /// Token IDs that are always forbidden. When hit, emit the violation message instead.
    private let bannedTokenIds: Set<Int32>

    /// If non-nil, inject this text into the output stream when a ban fires.
    private let banViolationMessage: String?

    private var prefixCursor: Int = 0

    // MARK: - Init

    public init(forcedPrefix: [Int32] = [],
         bannedTokenIds: Set<Int32> = [],
         banViolationMessage: String? = nil) {
        // Qwen EOS tokens: 151643 (<|endoftext|>), 151645 (<|im_end|>)
        assert(!forcedPrefix.contains(where: { $0 == 151643 || $0 == 151645 }),
               "FormatEnforcer: forced prefix contains EOS token — decode terminates immediately")
        self.forcedPrefix = forcedPrefix
        self.bannedTokenIds = bannedTokenIds
        self.banViolationMessage = banViolationMessage
    }

    // MARK: - Core decision

    /// Call this after extractArgmax, before committing the token.
    ///
    /// Returns the token to actually emit, or nil if the step should be
    /// suppressed and a violation message should be streamed instead.
    public mutating func next(rawToken: Int32) -> Decision {
        // 1. Prefix forcing — override the raw token.
        if prefixCursor < forcedPrefix.count {
            let forced = forcedPrefix[prefixCursor]
            prefixCursor += 1
            return .emit(forced)
        }

        // 2. Ban check.
        if bannedTokenIds.contains(rawToken) {
            if let msg = banViolationMessage {
                return .injectText(msg)
            }
            return .skip
        }

        return .emit(rawToken)
    }

    public enum Decision {
        case emit(Int32)        // use this token ID
        case injectText(String) // stream this literal string instead, advance position normally
        case skip               // drop this token silently, advance position
    }
}
