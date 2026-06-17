// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "tauri-plugin-litert",
    platforms: [
        // Xcode 26 PackageDescription dropped constants below macOS 14 / iOS 15;
        // LiteRT-LM itself requires iOS 15+.
        .macOS(.v14),
        .iOS(.v15),
    ],
    products: [
        .library(
            name: "tauri-plugin-litert",
            type: .static,
            targets: ["tauri-plugin-litert"]),
    ],
    dependencies: [
        .package(name: "Tauri", path: "../.tauri/tauri-api"),
        // Local checkout (litert-ios/LiteRT-LM). Must be a *path* dependency:
        // the LiteRTLM target uses unsafeFlags (-all_load), which SwiftPM only
        // permits for root/local packages, not versioned remote dependencies.
        .package(name: "LiteRTLM", path: "../../LiteRT-LM"),
    ],
    targets: [
        .target(
            name: "tauri-plugin-litert",
            dependencies: [
                .byName(name: "Tauri"),
                .product(name: "LiteRTLM", package: "LiteRTLM"),
            ],
            path: "Sources")
    ]
)
