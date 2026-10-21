import Foundation
@preconcurrency import CoreML

/// Zero-Copy Speculative Streaming (Shared Memory IPC)
/// Maps a 64KB POSIX file to bypass TCP/HTTP overhead completely.
///
/// Memory Layout:
/// - Offset 0: magic (UInt32) 0x5157454E "QWEN"
/// - Offset 4: state (UInt32) (0=IDLE, 1=PYTHON_WRITING, 2=SWIFT_PROCESSING, 3=SWIFT_WRITING, 4=DONE, 5=ERROR)
/// - Offset 8: prompt_length (UInt32)
/// - Offset 12: output_head (UInt32)
/// - Offset 16: max_tokens (UInt32)
/// - Offset 20: prompt_buffer (4096 * Int32)
/// - Offset 16404: output_buffer (4096 * Int32)
final class IPCServer: @unchecked Sendable {
    private let filePath: String
    private let fd: Int32
    private let mapSize = 65536 // 64KB
    private var mmapPtr: UnsafeMutableRawPointer?
    private let actor: InferenceActor
    private var isRunning = true
    private let queue = DispatchQueue(label: "com.qwen-ane.qwen.ipc", qos: .userInteractive)
    
    init(filePath: String = "/tmp/qwen_ipc.mmap", actor: InferenceActor) throws {
        self.filePath = filePath
        self.actor = actor
        
        fd = open(filePath, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR)
        guard fd != -1 else {
            throw NSError(domain: "IPCServer", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to open mmap file"])
        }
        
        ftruncate(fd, off_t(mapSize))
        
        mmapPtr = mmap(nil, mapSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)
        guard mmapPtr != MAP_FAILED else {
            close(fd)
            throw NSError(domain: "IPCServer", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to mmap file"])
        }
        
        // Initialize if not already initialized
        let magicPtr = mmapPtr!.assumingMemoryBound(to: UInt32.self)
        if magicPtr.pointee != 0x5157454E {
            memset(mmapPtr!, 0, mapSize)
            magicPtr.pointee = 0x5157454E
        }
    }
    
    deinit {
        isRunning = false
        if let p = mmapPtr, p != MAP_FAILED {
            munmap(p, mapSize)
        }
        if fd != -1 {
            close(fd)
        }
    }
    
    public func start() {
        fputs("[IPCServer] Listening on mmap: \(filePath)\n", stderr)
        queue.async { [weak self] in
            self?.pollLoop()
        }
    }
    
    private func pollLoop() {
        guard let mmapPtr = self.mmapPtr else { return }
        
        let statePtr = mmapPtr.advanced(by: 4).assumingMemoryBound(to: UInt32.self)
        let promptLenPtr = mmapPtr.advanced(by: 8).assumingMemoryBound(to: UInt32.self)
        let outputHeadPtr = mmapPtr.advanced(by: 12).assumingMemoryBound(to: UInt32.self)
        let maxTokensPtr = mmapPtr.advanced(by: 16).assumingMemoryBound(to: UInt32.self)
        let promptBufferPtr = mmapPtr.advanced(by: 20).assumingMemoryBound(to: Int32.self)
        let outputBufferPtr = mmapPtr.advanced(by: 16404).assumingMemoryBound(to: Int32.self)
        
        var lastLoggedState: UInt32 = 999
        while isRunning {
            var state: UInt32 = 0
            let bytesRead = pread(fd, &state, 4, 4)
            if state != lastLoggedState {
                fputs("[IPCServer] State changed: \(state) (bytesRead=\(bytesRead), fd=\(fd))\n", stderr)
                lastLoggedState = state
            }
            if state == 2 { // SWIFT_PROCESSING
                let promptLen = Int(promptLenPtr.pointee)
                let maxTokens = Int(maxTokensPtr.pointee)
                
                var promptIds = [Int32]()
                promptIds.reserveCapacity(promptLen)
                for i in 0..<promptLen {
                    promptIds.append(promptBufferPtr[i])
                }
                
                statePtr.pointee = 3 // SWIFT_WRITING
                outputHeadPtr.pointee = 0
                
                let outHeadAddr = Int(bitPattern: outputHeadPtr)
                let outBufAddr = Int(bitPattern: outputBufferPtr)
                let localActor = self.actor
                let localStateAddr = Int(bitPattern: statePtr)
                let localPromptIds = promptIds
                let localMaxTokens = maxTokens
                
                // Spin up an isolated Task to interact with the actor safely
                Task { @Sendable [localActor, localStateAddr, localPromptIds, localMaxTokens, outHeadAddr, outBufAddr] in
                    do {
                        fputs("[IPCServer] Generating \(localMaxTokens) tokens for prompt len: \(localPromptIds.count)...\n", stderr)
                        
                        try await localActor.inferRaw(
                            promptIds: localPromptIds,
                            maxTokens: localMaxTokens
                        ) { @Sendable [outHeadAddr, outBufAddr] tokenId in
                            let outHeadPtr = UnsafeMutablePointer<UInt32>(bitPattern: outHeadAddr)!
                            let outBufPtr = UnsafeMutablePointer<Int32>(bitPattern: outBufAddr)!
                            
                            let head = Int(outHeadPtr.pointee)
                            if head < 4096 {
                                outBufPtr[head] = tokenId
                                outHeadPtr.pointee = UInt32(head + 1)
                            }
                        }
                        
                        let sPtr = UnsafeMutablePointer<UInt32>(bitPattern: localStateAddr)!
                        sPtr.pointee = 4 // DONE
                        fputs("[IPCServer] Generation DONE\n", stderr)
                    } catch {
                        fputs("[IPCServer] Generation failed: \(error)\n", stderr)
                        let sPtr = UnsafeMutablePointer<UInt32>(bitPattern: localStateAddr)!
                        sPtr.pointee = 5 // ERROR
                    }
                }
                
                // Sleep for a moment to let the state fully process before we continue polling
                usleep(100000)
            } else {
                usleep(1000) // Sleep 1ms to prevent 100% CPU on spin
            }
        }
    }
}
