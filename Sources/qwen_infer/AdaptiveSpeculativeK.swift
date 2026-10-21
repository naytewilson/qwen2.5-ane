//
// AdaptiveSpeculativeK.swift
// Adaptive K-parameter for Qwen2.5-1.5B speculative decoding (EAGLE-3)
// Monitors temperature, recent acceptance rate, and token-level entropy
// K \in {1,2,3,4} - switches at step boundaries; instant adaptation

import Foundation
import Accelerate

public final class AdaptiveSpeculativeK {
    // MARK: - Constants
    private let windowSize = 12          // rolling acceptance history
    private let minK = 1
    private let maxK = 4
    
    // MARK: - State
    private var acceptHistory: [Bool] = []
    private var recentEntropy: [Float] = []
    private var currentTemperature: Float = 1.0
    
    // MARK: - Configuration (tuned for 1.5B ANE)
    private let highAcceptThreshold: Float = 0.70
    private let lowAcceptThreshold: Float  = 0.30
    private let lowTempBound: Float        = 0.8
    private let highTempBound: Float       = 1.5
    private let entropyHigh: Float         = 0.6   // high entropy -> cut K
    private let entropyLow: Float          = 0.3   // low entropy -> boost K
    
    public init() {}
    
    /// Called every time the base model evaluates a draft token.
    /// - Parameter accepted: true if the draft token was accepted
    /// - Parameter draftLogProbabilities: log-probabilities of the draft head output (for entropy)
    public func record(accepted: Bool, draftLogProbabilities: [Float]) {
        // Update acceptance history
        acceptHistory.append(accepted)
        if acceptHistory.count > windowSize {
            acceptHistory.removeFirst()
        }
        
        // Compute entropy of the draft log-probabilities (normalised to [0,1])
        let softmax = softmax(draftLogProbabilities)
        let entropy = computeEntropy(softmax)
        recentEntropy.append(entropy)
        if recentEntropy.count > windowSize {
            recentEntropy.removeFirst()
        }
    }
    
    /// Set the generation temperature (called when config changes)
    public func setTemperature(_ temp: Float) {
        currentTemperature = temp
    }
    
    /// Compute the next K value based on current state.
    /// Should be called before drafting the next set of tokens.
    /// - Returns: K in [1, 4]
    public func computeK() -> Int {
        let acceptRate = acceptHistory.isEmpty ? 0.5 : Float(acceptHistory.filter { $0 }.count) / Float(acceptHistory.count)
        let avgEntropy = recentEntropy.isEmpty ? 0.5 : recentEntropy.reduce(0, +) / Float(recentEntropy.count)
        
        // Decision logic
        if acceptRate >= highAcceptThreshold && currentTemperature <= lowTempBound && avgEntropy <= entropyLow {
            return 4
        } else if acceptRate >= highAcceptThreshold && currentTemperature <= highTempBound && avgEntropy <= entropyHigh {
            return 3
        } else if acceptRate <= lowAcceptThreshold || currentTemperature >= highTempBound || avgEntropy >= entropyHigh {
            return 1
        } else {
            return 2   // default moderate
        }
    }
    
    // MARK: - Helpers
    private func softmax(_ logits: [Float]) -> [Float] {
        // Numerically stable softmax
        let maxVal = logits.max() ?? 0
        var exps = [Float](repeating: 0, count: logits.count)
        var sum: Float = 0
        for i in logits.indices {
            exps[i] = exp(logits[i] - maxVal)
            sum += exps[i]
        }
        if sum > 0 {
            for i in exps.indices { exps[i] /= sum }
        }
        return exps
    }
    
    private func computeEntropy(_ probs: [Float]) -> Float {
        // Normalised entropy: H / log(N)
        let n = Float(probs.count)
        guard n > 1 else { return 0 }
        var h: Float = 0
        for p in probs where p > 0 {
            h -= p * log(p)
        }
        return h / log(n)
    }
}
