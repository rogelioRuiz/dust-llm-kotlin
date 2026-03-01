package io.t6x.dust.llm

enum class StopReason(val wireValue: String) {
    MAX_TOKENS("max_tokens"),
    STOP_SEQUENCE("stop_sequence"),
    EOS("eos"),
    CANCELLED("cancelled"),
}

interface LlamaEngine {
    val nCtx: Int

    fun tokenize(text: String, addSpecial: Boolean): IntArray

    fun detokenize(tokens: IntArray): String

    fun getEmbedding(promptTokens: IntArray): FloatArray =
        throw LlamaError.UnsupportedOperation("embedding extraction is not available for this engine")

    val embeddingDims: Int
        get() = 0

    fun generate(promptTokens: IntArray, maxTokens: Int, sampler: SamplerConfig): GenerateEngineResult

    fun generateWithVision(
        promptTokens: IntArray,
        imageEmbedding: ImageEmbedding,
        visionEncoder: VisionEncoderEngine,
        maxTokens: Int,
        sampler: SamplerConfig,
    ): GenerateEngineResult = throw LlamaError.UnsupportedOperation("vision generation is not available for this engine")

    fun generateStreaming(
        promptTokens: IntArray,
        maxTokens: Int,
        sampler: SamplerConfig,
        isCancelled: () -> Boolean,
        onToken: (Int) -> Unit,
    ): StopReason

    fun generateStreamingWithVision(
        promptTokens: IntArray,
        imageEmbedding: ImageEmbedding,
        visionEncoder: VisionEncoderEngine,
        maxTokens: Int,
        sampler: SamplerConfig,
        isCancelled: () -> Boolean,
        onToken: (Int) -> Unit,
    ): StopReason = throw LlamaError.UnsupportedOperation("vision generation is not available for this engine")
}

sealed class LlamaError(override val message: String) : Exception(message) {
    class ModelEvicted : LlamaError("Model was evicted from memory")
    class UnsupportedOperation(detail: String) : LlamaError(detail)
}

data class SamplerConfig(
    val temperature: Float = 0.8f,
    val topK: Int = 40,
    val topP: Float = 0.95f,
    val minP: Float = 0.05f,
    val repeatPenalty: Float = 1.1f,
    val repeatLastN: Int = 64,
    val seed: Int = 0,
)

data class GenerateEngineResult(
    val tokens: IntArray,
    val stopReason: StopReason,
)

data class GenerateResult(
    val text: String,
    val tokenCount: Int,
    val stopReason: StopReason,
)
