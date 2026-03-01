# Contributing to dust-llm-kotlin

Thanks for your interest in contributing! This guide will help you get set up and understand our development workflow.

## Prerequisites

- **JDK 17**
- **Android SDK** with NDK installed (CMake native build in `src/main/cpp/`)
- **Git**
- **dust-core-kotlin** cloned as a sibling directory (`../dust-core-kotlin`)

## Getting Started

```bash
# Clone both repos side-by-side
git clone https://github.com/rogelioRuiz/dust-core-kotlin.git
git clone https://github.com/rogelioRuiz/dust-llm-kotlin.git

cd dust-llm-kotlin

# Run tests
./gradlew test
```

## Project Structure

```
src/main/java/io/t6x/dust/llm/
  ChatTemplateEngine.kt      # Chat prompt formatting
  LlamaContextWrapper.kt     # Native context lifecycle wrapper
  LlamaEngine.kt             # llama.cpp inference engine
  LlamaJNI.kt                # JNI bindings for llama.cpp
  LlamaSession.kt            # Single inference session
  LLMConfig.kt               # Model configuration
  LLMSessionManager.kt       # Session lifecycle and caching
  VisionEncoder.kt           # Multimodal vision input encoding

src/main/cpp/
  CMakeLists.txt             # NDK build for llama.cpp

src/test/java/io/t6x/dust/llm/
  LLMChatTemplateTest.kt    # 8 tests
  LLMGenerationTest.kt      # 8 tests
  LLMRegistryTest.kt        # 9 tests
  LLMSessionManagerTest.kt  # 10 tests
  LLMStreamingTest.kt       # 9 tests
  LLMVisionTest.kt          # 8 tests
```

## Making Changes

### 1. Create a branch

```bash
git checkout -b feat/my-feature
```

### 2. Make your changes

- Follow existing Kotlin conventions in the codebase
- JNI changes require updating both the Kotlin declarations in `LlamaJNI.kt` and the native C++ code
- Add tests for new functionality

### 3. Add the license header

All `.kt` files must include the Apache 2.0 header:

```kotlin
//
// Copyright 2026 T6X
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
```

### 4. Run checks

```bash
./gradlew test      # All 52 tests must pass
./gradlew build     # Clean build
```

### 5. Commit with a conventional message

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add LoRA adapter loading support
fix: correct KV cache eviction under memory pressure
docs: update README usage examples
chore(deps): bump dust-core-kotlin to 0.2.0
```

### 6. Open a pull request

Push your branch and open a PR against `main`.

## Reporting Issues

- **Bugs**: Open an issue with steps to reproduce
- **Features**: Open an issue describing the use case and proposed API

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
