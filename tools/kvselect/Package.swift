// swift-tools-version:5.9
// kvselect — rewrite the dynamic-begin KV slice_update ops (ANE compile status -14) in a
// multi-function ANEMLL package's `prefill` into a static-bounds one-hot placement + blend.
import PackageDescription
let package = Package(
    name: "kvselect",
    platforms: [.macOS(.v14)],
    dependencies: [.package(url: "https://github.com/apple/swift-protobuf.git", from: "1.27.0")],
    targets: [.executableTarget(name: "kvselect",
                                dependencies: [.product(name: "SwiftProtobuf", package: "swift-protobuf")],
                                path: "Sources/kvselect")]
)
