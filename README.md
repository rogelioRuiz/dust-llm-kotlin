<p align="center">
  <img alt="dust" src="assets/dust_banner.png" width="400">
</p>

<p align="center">
  <strong>Device Unified Serving Toolkit</strong><br>
  <a href="https://github.com/rogelioRuiz/dust">dust ecosystem</a> · v0.1.0 · Apache 2.0
</p>

<p align="center">
  <a href="https://github.com/rogelioRuiz/dust/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
  <img alt="Version" src="https://img.shields.io/badge/version-0.1.0-informational">
  <img alt="Maven" src="https://img.shields.io/badge/Maven-io.t6x.dust%3Adust--llm-blue">
  <a href="https://developer.android.com/studio/releases/platforms"><img alt="API" src="https://img.shields.io/badge/API-26+-green.svg"></a>
  <a href="https://kotlinlang.org"><img alt="Kotlin" src="https://img.shields.io/badge/Kotlin-2.1-purple.svg"></a>
  <img alt="GGUF" src="https://img.shields.io/badge/GGUF-llama.cpp-blueviolet">
  <a href="https://github.com/rogelioRuiz/dust-llm-kotlin/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/rogelioRuiz/dust-llm-kotlin/actions/workflows/ci.yml/badge.svg?branch=main"></a>
</p>

---

<p align="center">
<strong>dust ecosystem</strong> —
<a href="../capacitor-core/README.md">capacitor-core</a> ·
<a href="../capacitor-llm/README.md">capacitor-llm</a> ·
<a href="../capacitor-onnx/README.md">capacitor-onnx</a> ·
<a href="../capacitor-serve/README.md">capacitor-serve</a> ·
<a href="../capacitor-embeddings/README.md">capacitor-embeddings</a>
<br>
<a href="../dust-core-kotlin/README.md">dust-core-kotlin</a> ·
<strong>dust-llm-kotlin</strong> ·
<a href="../dust-onnx-kotlin/README.md">dust-onnx-kotlin</a> ·
<a href="../dust-embeddings-kotlin/README.md">dust-embeddings-kotlin</a> ·
<a href="../dust-serve-kotlin/README.md">dust-serve-kotlin</a>
<br>
<a href="../dust-core-swift/README.md">dust-core-swift</a> ·
<a href="../dust-llm-swift/README.md">dust-llm-swift</a> ·
<a href="../dust-onnx-swift/README.md">dust-onnx-swift</a> ·
<a href="../dust-embeddings-swift/README.md">dust-embeddings-swift</a> ·
<a href="../dust-serve-swift/README.md">dust-serve-swift</a>
</p>

---

# dust-llm-kotlin

Android GGUF/llama.cpp inference and chat runtime with JNI bindings.

**Version: 0.1.0**

## Overview

`dust-llm-kotlin` provides a Kotlin-first API for running GGUF large language models on Android via llama.cpp. It builds on [dust-core-kotlin](../dust-core-kotlin) and includes a JNI layer that compiles llama.cpp for arm64-v8a:

- **LlamaEngine** — load GGUF models and manage llama.cpp context
- **ChatSession** — multi-turn chat with message history and system prompts
- **GenerationSession** — single-shot text generation with sampling parameters
- **StreamingHandler** — token-by-token streaming callbacks
- **VisionProcessor** — multimodal image+text inference (LLaVA-style)
- **LlmRegistry** — thread-safe model registration and lookup

> **Note:** The JNI binary is built for `arm64-v8a` only. Unit tests use `MockLlamaEngine` and do not require the native library.

## Architecture

```
dust-llm-kotlin/
├── src/main/kotlin/io/t6x/dust/llm/   # Kotlin API
│   ├── LlamaEngine.kt
│   ├── ChatSession.kt
│   ├── GenerationSession.kt
│   ├── StreamingHandler.kt
│   ├── VisionProcessor.kt
│   └── LlmRegistry.kt
└── src/main/cpp/                        # JNI layer (CMake → llama.cpp)
```

## Install

### Gradle — local project dependency

```groovy
// settings.gradle
include ':dust-llm-kotlin'
project(':dust-llm-kotlin').projectDir = new File('../dust-llm-kotlin')

// Also include the contract library
include ':dust-core-kotlin'
project(':dust-core-kotlin').projectDir = new File('../dust-core-kotlin')

// build.gradle
dependencies {
    implementation project(':dust-llm-kotlin')
}
```

### Gradle — Maven (when published)

```groovy
dependencies {
    implementation 'io.t6x.dust:dust-llm:0.1.0'
}
```

## Usage

```kotlin
import io.t6x.dust.llm.*

// 1. Load a GGUF model
val engine = LlamaEngine(modelPath = "/data/model.gguf", nThreads = 4)

// 2. Start a chat session
val chat = ChatSession(engine, systemPrompt = "You are a helpful assistant.")
val reply = chat.send("What is 2 + 2?")

// 3. Or stream tokens
val streaming = StreamingHandler { token -> print(token) }
val gen = GenerationSession(engine, handler = streaming)
gen.generate("Once upon a time")

// 4. Clean up
engine.close()
```

## Test

```bash
./gradlew test    # 52 JUnit tests (6 suites)
```

| Suite | Tests | Coverage |
|-------|-------|----------|
| `ChatSessionTest` | 8 | Multi-turn, system prompt, history management |
| `GenerationSessionTest` | 8 | Sampling params, stop tokens, max length |
| `LlmRegistryTest` | 9 | Register/resolve, thread safety |
| `LlamaEngineTest` | 10 | Load/close, context config, mock engine |
| `StreamingHandlerTest` | 9 | Token callbacks, cancellation, buffering |
| `VisionProcessorTest` | 8 | Image encoding, multimodal prompt assembly |

No emulator needed — all tests run on the JVM with `MockLlamaEngine`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding conventions, and PR guidelines.

## License

Copyright 2026 Rogelio Ruiz Perez. Licensed under the [Apache License 2.0](LICENSE).
