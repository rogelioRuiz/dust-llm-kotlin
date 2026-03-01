package io.t6x.dust.llm

import io.t6x.dust.core.DustCoreError
import io.t6x.dust.core.SessionPriority
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.fail
import org.junit.Test

class LLMGenerationTest {
    @Test
    fun l2T1TokenizeKnownStringReturnsExpectedTokens() {
        val engine = MockLlamaEngine().apply {
            tokenizeResult = intArrayOf(10, 20, 30, 40, 50)
        }
        val session = makeSession(engine)

        val tokens = session.tokenize("hello", addSpecial = true)

        assertArrayEquals(intArrayOf(10, 20, 30, 40, 50), tokens)
        assertEquals(5, tokens.size)
        assertEquals("hello", engine.lastTokenizeText)
    }

    @Test
    fun l2T2RoundTripTokenizeDetokenize() {
        val engine = MockLlamaEngine().apply {
            tokenizeResult = intArrayOf(10, 20, 30)
            detokenizeResult = "hello world"
        }
        val session = makeSession(engine)

        val tokens = session.tokenize("hello world", addSpecial = true)
        val text = session.detokenize(tokens)

        assertArrayEquals(intArrayOf(10, 20, 30), tokens)
        assertEquals("hello world", text)
    }

    @Test
    fun l2T3GenerateReturnsNonEmptyString() {
        val engine = MockLlamaEngine().apply {
            tokenizeResult = intArrayOf(1, 10, 11)
            generateResult = GenerateEngineResult(intArrayOf(5, 6, 7), StopReason.MAX_TOKENS)
            detokenizeResult = "output text"
        }
        val session = makeSession(engine)

        val result = session.generate("prompt", 3, emptyList(), SamplerConfig())

        assertEquals("output text", result.text)
        assertEquals(3, result.tokenCount)
        assertEquals(StopReason.MAX_TOKENS, result.stopReason)
    }

    @Test
    fun l2T4TemperatureZeroForwardsGreedySamplerSetting() {
        val engine = MockLlamaEngine().apply {
            tokenizeResult = intArrayOf(1, 10, 11)
            generateResult = GenerateEngineResult(intArrayOf(5, 6, 7), StopReason.MAX_TOKENS)
            detokenizeResult = "output text"
        }
        val session = makeSession(engine)

        session.generate("prompt", 3, emptyList(), SamplerConfig(temperature = 0f))

        assertEquals(0f, engine.lastGenerateSampler?.temperature)
    }

    @Test
    fun l2T5GenerateForwardsMaxTokens() {
        val engine = MockLlamaEngine().apply {
            tokenizeResult = intArrayOf(1, 10, 11)
            generateResult = GenerateEngineResult(intArrayOf(5, 6, 7, 8, 9), StopReason.MAX_TOKENS)
            detokenizeResult = "five tokens"
        }
        val session = makeSession(engine)

        val result = session.generate("prompt", 5, emptyList(), SamplerConfig())

        assertEquals(5, result.tokenCount)
        assertEquals(StopReason.MAX_TOKENS, result.stopReason)
        assertEquals(5, engine.lastGenerateMaxTokens)
    }

    @Test
    fun l2T6StopSequenceTruncatesText() {
        val engine = MockLlamaEngine().apply {
            tokenizeResult = intArrayOf(1, 10, 11)
            generateResult = GenerateEngineResult(intArrayOf(5, 6, 7), StopReason.MAX_TOKENS)
            detokenizeResult = "hello STOP world"
        }
        val session = makeSession(engine)

        val result = session.generate("prompt", 3, listOf("STOP"), SamplerConfig())

        assertEquals("hello ", result.text)
        assertEquals(StopReason.STOP_SEQUENCE, result.stopReason)
    }

    @Test
    fun l2T7EosStopReasonPassesThrough() {
        val engine = MockLlamaEngine().apply {
            tokenizeResult = intArrayOf(1, 10, 11)
            generateResult = GenerateEngineResult(intArrayOf(5, 6), StopReason.EOS)
            detokenizeResult = "done"
        }
        val session = makeSession(engine)

        val result = session.generate("prompt", 2, listOf("STOP"), SamplerConfig())

        assertEquals("done", result.text)
        assertEquals(StopReason.EOS, result.stopReason)
    }

    @Test
    fun l2T8PromptOverflowThrowsContextOverflowEquivalent() {
        val engine = MockLlamaEngine().apply {
            nCtx = 4
            tokenizeResult = intArrayOf(1, 2, 3, 4, 5)
        }
        val session = makeSession(engine)

        try {
            session.generate("prompt", 1, emptyList(), SamplerConfig())
            fail("Expected InvalidInput")
        } catch (error: DustCoreError) {
            require(error is DustCoreError.InvalidInput)
            assertEquals("Invalid input: Prompt has 5 tokens but context size is 4", error.message)
        }
    }

    private fun makeSession(engine: MockLlamaEngine): LlamaSession {
        return LlamaSession(
            sessionId = "test-session",
            engine = engine,
            metadata = LLMModelMetadata(name = "mock", chatTemplate = null, hasVision = false),
            sessionPriority = SessionPriority.INTERACTIVE,
        )
    }
}

private class MockLlamaEngine : LlamaEngine {
    override var nCtx: Int = 64
    var tokenizeResult: IntArray = intArrayOf()
    var detokenizeResult: String = ""
    var generateResult: GenerateEngineResult = GenerateEngineResult(intArrayOf(), StopReason.MAX_TOKENS)

    var lastTokenizeText: String? = null
    var lastGeneratePromptTokens: IntArray? = null
    var lastGenerateMaxTokens: Int? = null
    var lastGenerateSampler: SamplerConfig? = null

    override fun tokenize(text: String, addSpecial: Boolean): IntArray {
        lastTokenizeText = text
        return tokenizeResult
    }

    override fun detokenize(tokens: IntArray): String {
        return detokenizeResult
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

        for (token in generateResult.tokens.take(maxTokens)) {
            if (isCancelled()) {
                return StopReason.CANCELLED
            }
            onToken(token)
        }

        return generateResult.stopReason
    }
}
