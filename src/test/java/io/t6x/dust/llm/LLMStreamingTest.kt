package io.t6x.dust.llm

import io.t6x.dust.core.DustCoreError
import io.t6x.dust.core.ModelStatus
import io.t6x.dust.core.SessionPriority
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

class LLMStreamingTest {
    @Test
    fun l3T1StreamGenerateEmitsIncrementingTokenIndexes() {
        val engine = StreamingMockLlamaEngine().apply {
            tokenizeResult = intArrayOf(1, 2, 3)
            streamingTokens = intArrayOf(10, 11, 12, 13, 14)
            detokenizeHandler = { tokens -> "x".repeat(tokens.size) }
        }
        val session = makeSession(engine)

        val tokenIndexes = mutableListOf<Int>()

        session.streamGenerate(
            prompt = "prompt",
            maxTokens = 5,
            stopSequences = emptyList(),
            sampler = SamplerConfig(),
            onToken = { tokenIndex, _, _ -> tokenIndexes += tokenIndex },
            onComplete = { _, _, _, _ -> },
            onError = { error, _ -> fail("Unexpected error: $error") },
        )

        assertEquals(listOf(0, 1, 2, 3, 4), tokenIndexes)
    }

    @Test
    fun l3T2InferenceCompleteReportsCompletionTokensAfterLastToken() {
        val engine = StreamingMockLlamaEngine().apply {
            tokenizeResult = intArrayOf(1, 2)
            streamingTokens = intArrayOf(10, 11, 12, 13, 14)
            detokenizeHandler = { tokens -> "x".repeat(tokens.size) }
        }
        val session = makeSession(engine)

        var tokenEvents = 0
        var completionCount = 0
        var completionTokens = -1

        session.streamGenerate(
            prompt = "prompt",
            maxTokens = 5,
            stopSequences = emptyList(),
            sampler = SamplerConfig(),
            onToken = { _, _, _ -> tokenEvents += 1 },
            onComplete = { _, tokenCount, _, _ ->
                completionCount += 1
                completionTokens = tokenCount
                assertEquals(5, tokenEvents)
            },
            onError = { error, _ -> fail("Unexpected error: $error") },
        )

        assertEquals(1, completionCount)
        assertEquals(5, completionTokens)
    }

    @Test
    fun l3T3InferenceCompleteReportsPositiveTokensPerSecond() {
        val engine = StreamingMockLlamaEngine().apply {
            tokenizeResult = intArrayOf(1)
            streamingTokens = intArrayOf(10, 11, 12)
            detokenizeHandler = { tokens -> "x".repeat(tokens.size) }
        }
        val session = makeSession(engine)

        var reportedTokensPerSecond = 0.0

        session.streamGenerate(
            prompt = "prompt",
            maxTokens = 3,
            stopSequences = emptyList(),
            sampler = SamplerConfig(),
            onToken = { _, _, _ -> },
            onComplete = { _, _, tokensPerSecond, _ ->
                reportedTokensPerSecond = tokensPerSecond
            },
            onError = { error, _ -> fail("Unexpected error: $error") },
        )

        assertTrue(reportedTokensPerSecond > 0.0)
    }

    @Test
    fun l3T4CancelGenerationMidStreamStopsWithCancelledReason() {
        val engine = StreamingMockLlamaEngine().apply {
            tokenizeResult = intArrayOf(1)
            streamingTokens = intArrayOf(10, 11, 12, 13, 14)
            detokenizeHandler = { tokens -> "x".repeat(tokens.size) }
        }
        val session = makeSession(engine)

        var tokenEvents = 0
        var completionStopReason: StopReason? = null

        session.streamGenerate(
            prompt = "prompt",
            maxTokens = 5,
            stopSequences = emptyList(),
            sampler = SamplerConfig(),
            onToken = { tokenIndex, _, _ ->
                tokenEvents += 1
                if (tokenIndex == 2) {
                    session.cancelGeneration()
                }
            },
            onComplete = { _, _, _, stopReason ->
                completionStopReason = stopReason
            },
            onError = { error, _ -> fail("Unexpected error: $error") },
        )

        assertEquals(StopReason.CANCELLED, completionStopReason)
        assertTrue(tokenEvents <= 4)
    }

    @Test
    fun l3T5CancelGenerationWhileIdleIsNoOpAndSessionRemainsUsable() {
        val engine = StreamingMockLlamaEngine().apply {
            tokenizeResult = intArrayOf(1, 2)
            generateResult = GenerateEngineResult(intArrayOf(10, 11), StopReason.MAX_TOKENS)
            detokenizeResult = "ok"
        }
        val session = makeSession(engine)

        session.cancelGeneration()

        assertEquals(ModelStatus.Ready, session.status())

        val result = session.generate("prompt", 2, emptyList(), SamplerConfig())

        assertEquals("ok", result.text)
        assertEquals(2, result.tokenCount)
    }

    @Test
    fun l3T6SecondStreamGenerateWhileBusyReportsModelNotReady() {
        val engine = StreamingMockLlamaEngine().apply {
            tokenizeResult = intArrayOf(1)
            streamingTokens = intArrayOf(10, 11)
            detokenizeHandler = { tokens -> "x".repeat(tokens.size) }
        }
        val session = makeSession(engine)

        val firstTokenReady = CountDownLatch(1)
        val releaseFirstStream = CountDownLatch(1)
        val firstStreamFinished = CountDownLatch(1)

        val worker = Thread {
            session.streamGenerate(
                prompt = "prompt",
                maxTokens = 2,
                stopSequences = emptyList(),
                sampler = SamplerConfig(),
                onToken = { tokenIndex, _, _ ->
                    if (tokenIndex == 0) {
                        firstTokenReady.countDown()
                        releaseFirstStream.await(1, TimeUnit.SECONDS)
                    }
                },
                onComplete = { _, _, _, _ -> firstStreamFinished.countDown() },
                onError = { _, _ -> firstStreamFinished.countDown() },
            )
        }
        worker.start()

        assertTrue(firstTokenReady.await(1, TimeUnit.SECONDS))

        var concurrentError: Throwable? = null
        session.streamGenerate(
            prompt = "prompt",
            maxTokens = 1,
            stopSequences = emptyList(),
            sampler = SamplerConfig(),
            onToken = { _, _, _ -> fail("Second stream should not emit tokens") },
            onComplete = { _, _, _, _ -> fail("Second stream should not complete successfully") },
            onError = { error, _ -> concurrentError = error },
        )

        releaseFirstStream.countDown()
        assertTrue(firstStreamFinished.await(1, TimeUnit.SECONDS))
        worker.join(1_000)

        assertTrue(concurrentError is DustCoreError.ModelNotReady)
    }

    @Test
    fun l3T7MidStreamErrorReportsFailureAndSessionRemainsUsable() {
        val engine = StreamingMockLlamaEngine().apply {
            tokenizeResult = intArrayOf(1)
            streamingTokens = intArrayOf(10, 11, 12, 13, 14)
            streamingError = DustCoreError.InferenceFailed("decode failed")
            streamingErrorAfterTokens = 3
            detokenizeHandler = { tokens -> "x".repeat(tokens.size) }
        }
        val session = makeSession(engine)

        var tokenEvents = 0
        var failureTokenCount = -1
        var reportedError: Throwable? = null

        session.streamGenerate(
            prompt = "prompt",
            maxTokens = 5,
            stopSequences = emptyList(),
            sampler = SamplerConfig(),
            onToken = { _, _, _ -> tokenEvents += 1 },
            onComplete = { _, _, _, _ -> fail("Expected streaming failure") },
            onError = { error, tokenCount ->
                reportedError = error
                failureTokenCount = tokenCount
            },
        )

        assertEquals(3, tokenEvents)
        assertEquals(3, failureTokenCount)
        assertNotNull(reportedError)

        engine.streamingError = null
        engine.generateResult = GenerateEngineResult(intArrayOf(20, 21), StopReason.MAX_TOKENS)
        engine.detokenizeHandler = null
        engine.detokenizeResult = "recovered"

        val result = session.generate("prompt", 2, emptyList(), SamplerConfig())

        assertEquals("recovered", result.text)
        assertEquals(2, result.tokenCount)
    }

    @Test
    fun l3T8SecondStreamAfterCancelSucceeds() {
        val engine = StreamingMockLlamaEngine().apply {
            tokenizeResult = intArrayOf(1)
            streamingTokens = intArrayOf(10, 11, 12)
            detokenizeHandler = { tokens -> "x".repeat(tokens.size) }
        }
        val session = makeSession(engine)

        var firstStopReason: StopReason? = null
        session.streamGenerate(
            prompt = "prompt",
            maxTokens = 3,
            stopSequences = emptyList(),
            sampler = SamplerConfig(),
            onToken = { tokenIndex, _, _ ->
                if (tokenIndex == 0) {
                    session.cancelGeneration()
                }
            },
            onComplete = { _, _, _, stopReason ->
                firstStopReason = stopReason
            },
            onError = { error, _ -> fail("Unexpected error: $error") },
        )

        engine.streamingTokens = intArrayOf(20, 21)
        engine.streamingStopReason = StopReason.MAX_TOKENS
        engine.detokenizeHandler = { tokens -> "y".repeat(tokens.size) }

        var secondStopReason: StopReason? = null
        var secondTokenCount = -1
        session.streamGenerate(
            prompt = "prompt",
            maxTokens = 2,
            stopSequences = emptyList(),
            sampler = SamplerConfig(),
            onToken = { _, _, _ -> },
            onComplete = { _, tokenCount, _, stopReason ->
                secondStopReason = stopReason
                secondTokenCount = tokenCount
            },
            onError = { error, _ -> fail("Unexpected error: $error") },
        )

        assertEquals(StopReason.CANCELLED, firstStopReason)
        assertEquals(StopReason.MAX_TOKENS, secondStopReason)
        assertEquals(2, secondTokenCount)
    }

    @Test
    fun l3T9MultiByteEmojiAssemblesWithoutReplacementCharacters() {
        val engine = StreamingMockLlamaEngine().apply {
            tokenizeResult = intArrayOf(1)
            streamingTokens = intArrayOf(100, 101, 102, 103)
            detokenizeHandler = { tokens ->
                when {
                    tokens.contentEquals(intArrayOf(100)) -> "Hello "
                    tokens.contentEquals(intArrayOf(100, 101)) -> "Hello "
                    tokens.contentEquals(intArrayOf(100, 101, 102)) -> "Hello 😀"
                    tokens.contentEquals(intArrayOf(100, 101, 102, 103)) -> "Hello 😀!"
                    else -> ""
                }
            }
        }
        val session = makeSession(engine)

        val tokenTexts = mutableListOf<String>()
        var completedText = ""

        session.streamGenerate(
            prompt = "prompt",
            maxTokens = 4,
            stopSequences = emptyList(),
            sampler = SamplerConfig(),
            onToken = { _, _, text -> tokenTexts += text },
            onComplete = { fullText, _, _, _ -> completedText = fullText },
            onError = { error, _ -> fail("Unexpected error: $error") },
        )

        assertEquals(listOf("Hello ", "", "😀", "!"), tokenTexts)
        assertEquals("Hello 😀!", tokenTexts.joinToString(separator = ""))
        assertEquals("Hello 😀!", completedText)
        assertFalse(completedText.contains("\uFFFD"))
    }

    private fun makeSession(engine: StreamingMockLlamaEngine): LlamaSession {
        return LlamaSession(
            sessionId = "test-session",
            engine = engine,
            metadata = LLMModelMetadata(name = "mock", chatTemplate = null, hasVision = false),
            sessionPriority = SessionPriority.INTERACTIVE,
        )
    }
}

private class StreamingMockLlamaEngine : LlamaEngine {
    override var nCtx: Int = 64
    var tokenizeResult: IntArray = intArrayOf()
    var detokenizeResult: String = ""
    var generateResult: GenerateEngineResult = GenerateEngineResult(intArrayOf(), StopReason.MAX_TOKENS)
    var streamingTokens: IntArray = intArrayOf()
    var streamingStopReason: StopReason = StopReason.MAX_TOKENS
    var streamingError: Throwable? = null
    var streamingErrorAfterTokens: Int = 0
    var detokenizeHandler: ((IntArray) -> String)? = null

    var lastTokenizeText: String? = null
    var lastGeneratePromptTokens: IntArray? = null
    var lastGenerateMaxTokens: Int? = null
    var lastGenerateSampler: SamplerConfig? = null

    override fun tokenize(text: String, addSpecial: Boolean): IntArray {
        lastTokenizeText = text
        return tokenizeResult
    }

    override fun detokenize(tokens: IntArray): String {
        return detokenizeHandler?.invoke(tokens) ?: detokenizeResult
    }

    override fun generate(promptTokens: IntArray, maxTokens: Int, sampler: SamplerConfig): GenerateEngineResult {
        lastGeneratePromptTokens = promptTokens
        lastGenerateMaxTokens = maxTokens
        lastGenerateSampler = sampler
        return generateResult
    }

    override fun generateStreaming(
        promptTokens: IntArray,
        maxTokens: Int,
        sampler: SamplerConfig,
        isCancelled: () -> Boolean,
        onToken: (Int) -> Unit,
    ): StopReason {
        lastGeneratePromptTokens = promptTokens
        lastGenerateMaxTokens = maxTokens
        lastGenerateSampler = sampler

        var emittedTokenCount = 0

        for (token in streamingTokens.take(maxTokens)) {
            if (isCancelled()) {
                return StopReason.CANCELLED
            }

            streamingError?.let { error ->
                if (emittedTokenCount >= streamingErrorAfterTokens) {
                    throw error
                }
            }

            onToken(token)
            emittedTokenCount += 1
        }

        return streamingStopReason
    }
}
