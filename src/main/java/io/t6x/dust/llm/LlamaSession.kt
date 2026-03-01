package io.t6x.dust.llm

import io.t6x.dust.core.DustCoreError
import io.t6x.dust.core.DustInputTensor
import io.t6x.dust.core.DustOutputTensor
import io.t6x.dust.core.ModelSession
import io.t6x.dust.core.ModelStatus
import io.t6x.dust.core.SessionPriority
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

class LlamaSession(
    val sessionId: String,
    private var engine: LlamaEngine?,
    private var visionEncoderRef: VisionEncoderEngine?,
    val metadata: LLMModelMetadata,
    private val sessionPriority: SessionPriority,
) : ModelSession {
    constructor(
        sessionId: String,
        engine: LlamaEngine?,
        metadata: LLMModelMetadata,
        sessionPriority: SessionPriority,
    ) : this(
        sessionId = sessionId,
        engine = engine,
        visionEncoderRef = null,
        metadata = metadata,
        sessionPriority = sessionPriority,
    )

    constructor(
        sessionId: String,
        contextWrapper: LlamaContextWrapper,
        sessionPriority: SessionPriority,
    ) : this(
        sessionId = sessionId,
        engine = contextWrapper,
        visionEncoderRef = null,
        metadata = contextWrapper.metadata,
        sessionPriority = sessionPriority,
    )

    constructor(
        sessionId: String,
        contextWrapper: LlamaContextWrapper,
        sessionPriority: SessionPriority,
        visionEncoder: VisionEncoderEngine?,
    ) : this(
        sessionId = sessionId,
        engine = contextWrapper,
        visionEncoderRef = visionEncoder,
        metadata = contextWrapper.metadata,
        sessionPriority = sessionPriority,
    )

    constructor(
        sessionId: String,
        metadata: LLMModelMetadata,
        sessionPriority: SessionPriority,
    ) : this(
        sessionId = sessionId,
        engine = null,
        visionEncoderRef = null,
        metadata = metadata,
        sessionPriority = sessionPriority,
    )

    private val lock = ReentrantLock()
    private val cancelRequested = AtomicBoolean(false)
    private var currentStatus: ModelStatus = ModelStatus.Ready
    private var evicted = false
    private var isGenerating = false
    private val chatMessages = mutableListOf<ChatMessage>()
    var contextUsed: Int = 0
        private set
    private val templateEngine = ChatTemplateEngine(metadata.chatTemplate)
    internal val visionEncoder: VisionEncoderEngine?
        get() = lock.withLock { visionEncoderRef }
    val isModelEvicted: Boolean
        get() = lock.withLock { evicted }

    override suspend fun predict(inputs: List<DustInputTensor>): List<DustOutputTensor> {
        throw DustCoreError.InferenceFailed("predict is not implemented in L1")
    }

    override fun status(): ModelStatus = lock.withLock { currentStatus }

    override fun priority(): SessionPriority = sessionPriority

    override suspend fun close() {
        closeContext()
    }

    suspend fun evict() {
        releaseResources(markEvicted = true)
    }

    fun tokenize(text: String, addSpecial: Boolean): IntArray {
        val current = activeEngine()
        return current.tokenize(text, addSpecial)
    }

    fun detokenize(tokens: IntArray): String {
        val current = activeEngine()
        return current.detokenize(tokens)
    }

    fun countTokens(text: String): Int = tokenize(text, addSpecial = true).size

    fun getEmbedding(text: String): FloatArray {
        val current = activeEngine()
        val tokens = current.tokenize(text, addSpecial = true)
        return current.getEmbedding(tokens)
    }

    val embeddingDims: Int
        get() = activeEngine().embeddingDims

    fun generate(
        prompt: String,
        maxTokens: Int,
        stopSequences: List<String>,
        sampler: SamplerConfig,
    ): GenerateResult = generate(
        prompt = prompt,
        maxTokens = maxTokens,
        stopSequences = stopSequences,
        sampler = sampler,
        imageBytes = null,
    )

    fun generate(
        prompt: String,
        maxTokens: Int,
        stopSequences: List<String>,
        sampler: SamplerConfig,
        imageBytes: ByteArray?,
    ): GenerateResult {
        val current = activeEngine()
        val promptTokens = current.tokenize(prompt, addSpecial = true)
        val currentVisionEncoder = visionEncoder

        if (promptTokens.size >= current.nCtx) {
            throw DustCoreError.InvalidInput("Prompt has ${promptTokens.size} tokens but context size is ${current.nCtx}")
        }

        if (imageBytes != null && currentVisionEncoder == null) {
            throw LlamaError.UnsupportedOperation("vision input requires a vision-capable model")
        }

        beginGeneration()
        try {
            val generated = if (imageBytes != null && currentVisionEncoder != null) {
                val imageEmbedding = currentVisionEncoder.encode(imageBytes)
                try {
                    val promptTokenCount = promptTokens.size + imageEmbedding.tokenCount
                    if (promptTokenCount >= current.nCtx) {
                        throw DustCoreError.InvalidInput("Prompt has $promptTokenCount tokens but context size is ${current.nCtx}")
                    }

                    current.generateWithVision(promptTokens, imageEmbedding, currentVisionEncoder, maxTokens, sampler)
                } finally {
                    currentVisionEncoder.freeEmbedding(imageEmbedding)
                }
            } else {
                current.generate(promptTokens, maxTokens, sampler)
            }
            var text = current.detokenize(generated.tokens)
            var stopReason = generated.stopReason

            if (stopReason != StopReason.EOS) {
                for (sequence in stopSequences) {
                    val matchIndex = text.indexOf(sequence)
                    if (matchIndex >= 0) {
                        text = text.substring(0, matchIndex)
                        stopReason = StopReason.STOP_SEQUENCE
                        break
                    }
                }
            }

            return GenerateResult(
                text = text,
                tokenCount = generated.tokens.size,
                stopReason = stopReason,
            )
        } finally {
            endGeneration()
        }
    }

    fun applyTemplate(
        messages: List<ChatMessage>,
        addGenerationPrompt: Boolean,
    ): Pair<String, Int> {
        val prompt = templateEngine.apply(messages, addGenerationPrompt)
        val tokens = tokenize(prompt, addSpecial = true)
        return prompt to tokens.size
    }

    fun generateChat(
        messages: List<ChatMessage>,
        maxTokens: Int,
        stopSequences: List<String>,
        sampler: SamplerConfig,
    ): Pair<GenerateResult, Int> {
        val current = activeEngine()
        var allMessages = lock.withLock { chatMessages.toList() } + messages
        allMessages = trimHistory(allMessages, maxTokens, current.nCtx)

        val prompt = templateEngine.apply(allMessages, addGenerationPrompt = true)
        val result = generate(prompt, maxTokens, stopSequences, sampler)

        val updatedHistory = allMessages + ChatMessage(role = "assistant", content = result.text)
        val fullPrompt = templateEngine.apply(updatedHistory, addGenerationPrompt = false)
        val updatedContextUsed = countTokens(fullPrompt)

        lock.withLock {
            chatMessages.clear()
            chatMessages += updatedHistory
            contextUsed = updatedContextUsed
        }

        return result to updatedContextUsed
    }

    fun streamGenerate(
        prompt: String,
        maxTokens: Int,
        stopSequences: List<String>,
        sampler: SamplerConfig,
        onToken: (tokenIndex: Int, tokenId: Int, text: String) -> Unit,
        onComplete: (fullText: String, tokenCount: Int, tokensPerSecond: Double, stopReason: StopReason) -> Unit,
        onError: (Throwable, Int) -> Unit,
    ) {
        streamGenerate(
            prompt = prompt,
            maxTokens = maxTokens,
            stopSequences = stopSequences,
            sampler = sampler,
            imageBytes = null,
            onToken = onToken,
            onComplete = { fullText, tokenCount, _, tokensPerSecond, stopReason ->
                onComplete(fullText, tokenCount, tokensPerSecond, stopReason)
            },
            onError = onError,
        )
    }

    fun streamGenerate(
        prompt: String,
        maxTokens: Int,
        stopSequences: List<String>,
        sampler: SamplerConfig,
        imageBytes: ByteArray?,
        onToken: (tokenIndex: Int, tokenId: Int, text: String) -> Unit,
        onComplete: (fullText: String, tokenCount: Int, promptTokenCount: Int, tokensPerSecond: Double, stopReason: StopReason) -> Unit,
        onError: (Throwable, Int) -> Unit,
    ) {
        val current = try {
            activeEngine()
        } catch (error: Throwable) {
            onError(error, 0)
            return
        }

        val promptTokens = try {
            current.tokenize(prompt, addSpecial = true)
        } catch (error: Throwable) {
            onError(error, 0)
            return
        }

        if (promptTokens.size >= current.nCtx) {
            onError(
                DustCoreError.InvalidInput("Prompt has ${promptTokens.size} tokens but context size is ${current.nCtx}"),
                0,
            )
            return
        }

        val currentVisionEncoder = visionEncoder
        if (imageBytes != null && currentVisionEncoder == null) {
            onError(LlamaError.UnsupportedOperation("vision input requires a vision-capable model"), 0)
            return
        }

        try {
            beginGeneration()
        } catch (error: Throwable) {
            onError(error, 0)
            return
        }

        val generatedTokens = mutableListOf<Int>()
        var lastDetokenizedText = ""
        var completionText = ""
        var stoppedBySequence = false
        var streamingFailure: Throwable? = null
        var promptTokenCount = promptTokens.size

        try {
            val startTime = System.nanoTime()
            var emittedText = ""

            val rawStopReason = if (imageBytes != null && currentVisionEncoder != null) {
                val imageEmbedding = currentVisionEncoder.encode(imageBytes)
                try {
                    promptTokenCount += imageEmbedding.tokenCount
                    if (promptTokenCount >= current.nCtx) {
                        throw DustCoreError.InvalidInput(
                            "Prompt has $promptTokenCount tokens but context size is ${current.nCtx}",
                        )
                    }

                    current.generateStreamingWithVision(
                        promptTokens = promptTokens,
                        imageEmbedding = imageEmbedding,
                        visionEncoder = currentVisionEncoder,
                        maxTokens = maxTokens,
                        sampler = sampler,
                        isCancelled = { cancelRequested.get() },
                        onToken = tokenLoop@{ token ->
                            if (streamingFailure != null) {
                                requestCancellation()
                                return@tokenLoop
                            }

                            generatedTokens += token

                            val detokenizedText = try {
                                current.detokenize(generatedTokens.toIntArray())
                            } catch (error: Throwable) {
                                streamingFailure = error
                                requestCancellation()
                                return@tokenLoop
                            }

                            lastDetokenizedText = detokenizedText
                            var nextCompletionText = detokenizedText

                            for (sequence in stopSequences) {
                                val matchIndex = detokenizedText.indexOf(sequence)
                                if (matchIndex >= 0) {
                                    nextCompletionText = detokenizedText.substring(0, matchIndex)
                                    stoppedBySequence = true
                                    requestCancellation()
                                    break
                                }
                            }

                            val tokenText = when {
                                nextCompletionText.startsWith(emittedText) -> {
                                    val delta = nextCompletionText.substring(emittedText.length)
                                    emittedText = nextCompletionText
                                    delta
                                }
                                !stoppedBySequence -> {
                                    val delta = if (detokenizedText.startsWith(emittedText)) {
                                        detokenizedText.substring(emittedText.length)
                                    } else {
                                        ""
                                    }
                                    emittedText = detokenizedText
                                    delta
                                }
                                else -> ""
                            }

                            completionText = nextCompletionText
                            onToken(generatedTokens.lastIndex, token, tokenText)
                        },
                    )
                } finally {
                    currentVisionEncoder.freeEmbedding(imageEmbedding)
                }
            } else {
                current.generateStreaming(
                    promptTokens = promptTokens,
                    maxTokens = maxTokens,
                    sampler = sampler,
                    isCancelled = { cancelRequested.get() },
                    onToken = tokenLoop@{ token ->
                        if (streamingFailure != null) {
                            requestCancellation()
                            return@tokenLoop
                        }

                        generatedTokens += token

                        val detokenizedText = try {
                            current.detokenize(generatedTokens.toIntArray())
                        } catch (error: Throwable) {
                            streamingFailure = error
                            requestCancellation()
                            return@tokenLoop
                        }

                        lastDetokenizedText = detokenizedText
                        var nextCompletionText = detokenizedText

                        for (sequence in stopSequences) {
                            val matchIndex = detokenizedText.indexOf(sequence)
                            if (matchIndex >= 0) {
                                nextCompletionText = detokenizedText.substring(0, matchIndex)
                                stoppedBySequence = true
                                requestCancellation()
                                break
                            }
                        }

                        val tokenText = when {
                            nextCompletionText.startsWith(emittedText) -> {
                                val delta = nextCompletionText.substring(emittedText.length)
                                emittedText = nextCompletionText
                                delta
                            }
                            !stoppedBySequence -> {
                                val delta = if (detokenizedText.startsWith(emittedText)) {
                                    detokenizedText.substring(emittedText.length)
                                } else {
                                    ""
                                }
                                emittedText = detokenizedText
                                delta
                            }
                            else -> ""
                        }

                        completionText = nextCompletionText
                        onToken(generatedTokens.lastIndex, token, tokenText)
                    },
                )
            }

            streamingFailure?.let {
                onError(it, generatedTokens.size)
                return
            }

            val elapsedNanos = (System.nanoTime() - startTime).coerceAtLeast(1L)
            val tokenCount = generatedTokens.size
            val finalText = if (stoppedBySequence) completionText else lastDetokenizedText
            val stopReason = if (stoppedBySequence) StopReason.STOP_SEQUENCE else rawStopReason
            val tokensPerSecond = if (tokenCount == 0) {
                0.0
            } else {
                tokenCount.toDouble() / (elapsedNanos.toDouble() / 1_000_000_000.0)
            }

            onComplete(finalText, tokenCount, promptTokenCount, tokensPerSecond, stopReason)
        } catch (error: Throwable) {
            onError(streamingFailure ?: error, generatedTokens.size)
        } finally {
            endGeneration()
        }
    }

    fun cancelGeneration() {
        requestCancellation()
    }

    fun clearHistory() {
        lock.withLock {
            chatMessages.clear()
            contextUsed = 0
        }
    }

    /** Synchronous resource cleanup — safe to call from non-suspend contexts (e.g. race-loser cleanup). */
    internal fun closeContext() {
        releaseResources(markEvicted = false)
    }

    private fun trimHistory(
        messages: List<ChatMessage>,
        maxTokens: Int,
        contextSize: Int,
    ): List<ChatMessage> {
        val prompt = templateEngine.apply(messages, addGenerationPrompt = true)
        val promptTokens = countTokens(prompt)

        if (promptTokens + maxTokens <= contextSize) {
            return messages
        }

        val nonSystemIndexes = messages.mapIndexedNotNull { index, message ->
            if (message.role == "system") {
                null
            } else {
                index
            }
        }

        if (nonSystemIndexes.size <= 1) {
            throw DustCoreError.InvalidInput("Prompt has $promptTokens tokens but context size is $contextSize")
        }

        var trimmedCount = 2
        while (trimmedCount <= nonSystemIndexes.size) {
            val removedIndexes = nonSystemIndexes.take(trimmedCount).toSet()
            val candidate = messages.filterIndexed { index, _ -> index !in removedIndexes }
            val candidatePrompt = templateEngine.apply(candidate, addGenerationPrompt = true)
            val candidateTokens = countTokens(candidatePrompt)

            if (candidateTokens + maxTokens <= contextSize) {
                return candidate
            }

            trimmedCount += 2
        }

        val lastNonSystemIndex = nonSystemIndexes.last()
        val candidate = messages.filterIndexed { index, message ->
            message.role == "system" || index == lastNonSystemIndex
        }
        val candidatePrompt = templateEngine.apply(candidate, addGenerationPrompt = true)
        val candidateTokens = countTokens(candidatePrompt)

        if (candidateTokens + maxTokens <= contextSize) {
            return candidate
        }

        throw DustCoreError.InvalidInput("Prompt has $candidateTokens tokens but context size is $contextSize")
    }

    private fun beginGeneration() {
        lock.withLock {
            if (evicted) {
                throw LlamaError.ModelEvicted()
            }

            if (currentStatus != ModelStatus.Ready || isGenerating) {
                throw DustCoreError.ModelNotReady
            }

            isGenerating = true
            cancelRequested.set(false)
        }
    }

    private fun endGeneration() {
        lock.withLock {
            isGenerating = false
            cancelRequested.set(false)
        }
    }

    private fun requestCancellation() {
        lock.withLock {
            if (isGenerating) {
                cancelRequested.set(true)
            }
        }
    }

    private fun activeEngine(): LlamaEngine = lock.withLock {
        if (evicted) {
            throw LlamaError.ModelEvicted()
        }

        return engine ?: throw DustCoreError.SessionClosed
    }

    private fun releaseResources(markEvicted: Boolean) {
        val (current, currentVisionEncoder) = lock.withLock {
            if (markEvicted) {
                evicted = true
            } else {
                currentStatus = ModelStatus.Unloading
            }
            isGenerating = false
            cancelRequested.set(false)
            chatMessages.clear()
            contextUsed = 0
            val active = engine
            val activeVisionEncoder = visionEncoderRef
            engine = null
            visionEncoderRef = null
            active to activeVisionEncoder
        }

        if (current is AutoCloseable) {
            current.close()
        }
        currentVisionEncoder?.close()
    }
}
