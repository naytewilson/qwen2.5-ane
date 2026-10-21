// swift-tools-version: 6.1
import PackageDescription

// Zero external dependencies.
//
// The ANE inference path (CortexEngine, BatchServing, the HTTP server) uses only
// Foundation, CoreML, Accelerate, Network and NaturalLanguage. The upstream
// research package declared MLX / swift-transformers products, but no target
// imports them — `MLXServerBackend` is a plain URLSession proxy to an external
// mlx_lm.server, not an MLX client. They were removed so the release builds
// offline in about a minute instead of pulling ~1.6 GB of checkouts.
let package = Package(
    name: "qwen-ane",
    platforms: [
        .macOS(.v15)
    ],
    products: [
        .executable(name: "qwen-ane", targets: ["qwen_infer"])
    ],
    targets: [
        .target(
            name: "CNeon",
            path: "Sources/CNeon"
        ),
        .executableTarget(
            name: "qwen_infer",
            dependencies: ["CNeon"],
            path: "Sources/qwen_infer",
            linkerSettings: [.linkedFramework("Accelerate")]
        )
    ]
)
