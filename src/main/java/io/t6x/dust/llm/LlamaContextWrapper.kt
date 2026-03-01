package io.t6x.dust.llm

import io.t6x.dust.core.DustCoreError
import java.io.File
import java.util.concurrent.atomic.AtomicLong

data class LLMModelMetadata(
    val name: String?,
    val chatTemplate: String?,
    val hasVision: Boolean,
)

open class LlamaContextWrapper(
    private val handleRef: AtomicLong,
    val metadata: LLMModelMetadata,
    override val nCtx: Int,
    private val batchSize: Int,
) : AutoCloseable, LlamaEngine {

    val handle: Long
        get() = handleRef.get()

    override fun close() {
        val handle = handleRef.getAndSet(0L)
        if (handle != 0L) {
            LlamaJNI.nativeFree(handle)
        }
    }

    override fun tokenize(text: String, addSpecial: Boolean): IntArray {
        return LlamaJNI.nativeTokenize(handle, text, addSpecial)
            ?: throw DustCoreError.InferenceFailed("Tokenization failed")
    }

    override fun detokenize(tokens: IntArray): String {
        return LlamaJNI.nativeDetokenize(handle, tokens)
            ?: throw DustCoreError.InferenceFailed("Detokenization failed")
    }

    override fun getEmbedding(promptTokens: IntArray): FloatArray {
        return LlamaJNI.nativeGetEmbedding(handle, promptTokens)
            ?: throw DustCoreError.InferenceFailed("Embedding extraction failed")
    }

    override val embeddingDims: Int
        get() = LlamaJNI.nativeGetEmbeddingDims(handle)

    override fun generate(promptTokens: IntArray, maxTokens: Int, sampler: SamplerConfig): GenerateEngineResult {
        val raw = LlamaJNI.nativeGenerate(
            handle = handle,
            promptTokens = promptTokens,
            maxTokens = maxTokens,
            temperature = sampler.temperature,
            topK = sampler.topK,
            topP = sampler.topP,
            minP = sampler.minP,
            repeatPenalty = sampler.repeatPenalty,
            repeatLastN = sampler.repeatLastN,
            seed = sampler.seed,
        ) ?: throw DustCoreError.InferenceFailed("Generation failed")

        return parseGenerateResult(raw)
    }

    override fun generateWithVision(
        promptTokens: IntArray,
        imageEmbedding: ImageEmbedding,
        visionEncoder: VisionEncoderEngine,
        maxTokens: Int,
        sampler: SamplerConfig,
    ): GenerateEngineResult {
        val raw = LlamaJNI.nativeGenerateWithVision(
            handle = handle,
            promptTokens = promptTokens,
            embedHandle = imageEmbedding.nativeHandle,
            maxTokens = maxTokens,
            temperature = sampler.temperature,
            topK = sampler.topK,
            topP = sampler.topP,
            minP = sampler.minP,
            repeatPenalty = sampler.repeatPenalty,
            repeatLastN = sampler.repeatLastN,
            seed = sampler.seed,
        ) ?: throw DustCoreError.InferenceFailed("Generation failed")

        return parseGenerateResult(raw)
    }

    override fun generateStreamingWithVision(
        promptTokens: IntArray,
        imageEmbedding: ImageEmbedding,
        visionEncoder: VisionEncoderEngine,
        maxTokens: Int,
        sampler: SamplerConfig,
        isCancelled: () -> Boolean,
        onToken: (Int) -> Unit,
    ): StopReason {
        val rawStopReason = LlamaJNI.nativeGenerateStreamingWithVision(
            handle = handle,
            promptTokens = promptTokens,
            embedHandle = imageEmbedding.nativeHandle,
            maxTokens = maxTokens,
            temperature = sampler.temperature,
            topK = sampler.topK,
            topP = sampler.topP,
            minP = sampler.minP,
            repeatPenalty = sampler.repeatPenalty,
            repeatLastN = sampler.repeatLastN,
            seed = sampler.seed,
            callback = StreamingCallback { tokenId ->
                if (isCancelled()) {
                    false
                } else {
                    onToken(tokenId)
                    true
                }
            },
        ) ?: throw DustCoreError.InferenceFailed("Streaming generation failed")

        return when (rawStopReason) {
            StopReason.EOS.wireValue -> StopReason.EOS
            StopReason.CANCELLED.wireValue -> StopReason.CANCELLED
            else -> StopReason.MAX_TOKENS
        }
    }

    private fun parseGenerateResult(raw: Array<String>): GenerateEngineResult {
        if (raw.size < 3) {
            throw DustCoreError.InferenceFailed("Generation failed")
        }

        val tokens = if (raw[0].isBlank()) {
            intArrayOf()
        } else {
            raw[0].split(",")
                .filter { it.isNotBlank() }
                .map { it.toInt() }
                .toIntArray()
        }

        val stopReason = when (raw[2]) {
            StopReason.EOS.wireValue -> StopReason.EOS
            else -> StopReason.MAX_TOKENS
        }

        return GenerateEngineResult(tokens = tokens, stopReason = stopReason)
    }

    override fun generateStreaming(
        promptTokens: IntArray,
        maxTokens: Int,
        sampler: SamplerConfig,
        isCancelled: () -> Boolean,
        onToken: (Int) -> Unit,
    ): StopReason {
        val rawStopReason = LlamaJNI.nativeGenerateStreaming(
            handle = handle,
            promptTokens = promptTokens,
            maxTokens = maxTokens,
            temperature = sampler.temperature,
            topK = sampler.topK,
            topP = sampler.topP,
            minP = sampler.minP,
            repeatPenalty = sampler.repeatPenalty,
            repeatLastN = sampler.repeatLastN,
            seed = sampler.seed,
            callback = StreamingCallback { tokenId ->
                if (isCancelled()) {
                    false
                } else {
                    onToken(tokenId)
                    true
                }
            },
        ) ?: throw DustCoreError.InferenceFailed("Streaming generation failed")

        return when (rawStopReason) {
            StopReason.EOS.wireValue -> StopReason.EOS
            StopReason.CANCELLED.wireValue -> StopReason.CANCELLED
            else -> StopReason.MAX_TOKENS
        }
    }

    companion object {
        fun load(path: String, config: LLMConfig): LlamaContextWrapper {
            if (!File(path).exists()) {
                throw DustCoreError.InferenceFailed("Model file not found: $path")
            }

            val handle = LlamaJNI.nativeLoad(
                path,
                config.nGpuLayers,
                config.contextSize,
                config.batchSize,
            )

            if (handle == 0L) {
                throw DustCoreError.InferenceFailed("Failed to load GGUF model: $path")
            }

            val metadata = LLMModelMetadata(
                name = LlamaJNI.nativeGetMetadata(handle, "general.name"),
                chatTemplate = LlamaJNI.nativeGetMetadata(handle, "tokenizer.chat_template"),
                hasVision = LlamaJNI.nativeGetMetadata(handle, "clip.vision.image_size") != null,
            )

            return LlamaContextWrapper(AtomicLong(handle), metadata, config.contextSize, config.batchSize)
        }
    }
}
