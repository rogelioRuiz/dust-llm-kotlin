package io.t6x.dust.llm

import io.t6x.dust.core.DustCoreError
import io.t6x.dust.core.SessionPriority
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Assume.assumeTrue
import org.junit.Test

class LLMVisionTest {
    @Test
    fun l6T1VisionModelSessionCanHoldVisionEncoder() = runTest {
        val manager = LLMSessionManager(
            sessionFactory = { _, modelId, _, priority ->
                LlamaSession(
                    modelId,
                    VisionTestMockLlamaEngine(),
                    MockVisionEncoderEngine(),
                    LLMModelMetadata(name = "vision", chatTemplate = null, hasVision = true),
                    priority,
                )
            },
        )

        val session = manager.loadModel("/tmp/vision.gguf", "vision", LLMConfig(), SessionPriority.INTERACTIVE)

        assertTrue(session.metadata.hasVision)
        assertNotNull(session.visionEncoder)
    }

    @Test
    fun l6T2TextOnlySessionHasNoVisionEncoder() {
        val session = LlamaSession(
            sessionId = "text-only",
            engine = VisionTestMockLlamaEngine(),
            metadata = LLMModelMetadata(name = "text", chatTemplate = null, hasVision = false),
            sessionPriority = SessionPriority.INTERACTIVE,
        )

        assertTrue(!session.metadata.hasVision)
        assertNull(session.visionEncoder)
    }

    @Test
    fun l6T3ImageToTextOnlyModelThrowsUnsupportedOperation() {
        val session = LlamaSession(
            sessionId = "text-only",
            engine = VisionTestMockLlamaEngine(),
            metadata = LLMModelMetadata(name = "text", chatTemplate = null, hasVision = false),
            sessionPriority = SessionPriority.INTERACTIVE,
        )

        try {
            session.generate("describe", 2, emptyList(), SamplerConfig(), pngBytes())
            fail("Expected unsupportedOperation")
        } catch (error: LlamaError) {
            assertTrue(error is LlamaError.UnsupportedOperation)
        }
    }

    @Test
    fun l6T4RealMtmdEncodeReturnsEmbeddingWhenEnvIsSet() {
        val mmprojPath = System.getenv("LLAVA_MMPROJ_PATH")
        val modelPath = System.getenv("LLAVA_MODEL_PATH")
        assumeTrue("LLAVA_MMPROJ_PATH and LLAVA_MODEL_PATH must both be set",
            !mmprojPath.isNullOrEmpty() && !modelPath.isNullOrEmpty())

        try {
            val context = LlamaContextWrapper.load(modelPath!!, LLMConfig())
            val encoder = VisionEncoder(mmprojPath!!, context.handle)
            try {
                val embedding = encoder.encode(pngBytes())
                assertTrue(embedding.tokenCount > 0)
                encoder.freeEmbedding(embedding)
            } finally {
                encoder.close()
                context.close()
            }
        } catch (_: UnsatisfiedLinkError) {
            // Local JVM unit tests do not always load JNI artifacts; treat that as a skip.
        }
    }

    @Test
    fun l6T5VisionEncoderFailurePropagatesAsInferenceError() {
        val engine = VisionTestMockLlamaEngine()
        val session = makeSession(engine, MockVisionEncoderEngine(encodeError = DustCoreError.InferenceFailed("Failed to encode image")))

        try {
            session.generate("describe", 1, emptyList(), SamplerConfig(), "bad".encodeToByteArray())
            fail("Expected inference failure")
        } catch (error: DustCoreError) {
            assertTrue(error is DustCoreError.InferenceFailed)
        }
    }

    @Test
    fun l6T6ImageEmbeddingIsInjectedAfterPromptTokens() {
        val engine = VisionTestMockLlamaEngine()
        val visionEncoder = MockVisionEncoderEngine(imageTokenCount = 32)
        val session = makeSession(engine, visionEncoder)

        session.generate("describe", 2, emptyList(), SamplerConfig(), pngBytes())

        assertEquals(listOf(11, 12, 13), engine.lastVisionPromptTokens)
    }

    @Test
    fun l6T7OversizedImageDoesNotCrashWithMockEncoder() {
        val engine = VisionTestMockLlamaEngine()
        val session = makeSession(engine, MockVisionEncoderEngine(imageTokenCount = 64))

        val result = session.generate(
            prompt = "describe",
            maxTokens = 2,
            stopSequences = emptyList(),
            sampler = SamplerConfig(),
            imageBytes = ByteArray(64 * 1024) { 0x7f },
        )

        assertEquals("ok", result.text)
        assertEquals(2, result.tokenCount)
    }

    @Test
    fun l6T8StreamGenerateWithImageReportsPromptTokens() {
        val mmprojPath = System.getenv("LLAVA_MMPROJ_PATH")
        val modelPath = System.getenv("LLAVA_MODEL_PATH")
        assumeTrue("LLAVA_MMPROJ_PATH and LLAVA_MODEL_PATH must both be set",
            !mmprojPath.isNullOrEmpty() && !modelPath.isNullOrEmpty())

        try {
            val engine = VisionTestMockLlamaEngine().apply {
                shouldCallVisionEval = false
            }
            val context = LlamaContextWrapper.load(modelPath!!, LLMConfig())
            val encoder = VisionEncoder(mmprojPath!!, context.handle)
            val session = makeSession(engine, encoder)
            var observedPromptTokens = 0
            val tokenTexts = mutableListOf<String>()

            session.streamGenerate(
                prompt = "describe",
                maxTokens = 2,
                stopSequences = emptyList(),
                sampler = SamplerConfig(),
                imageBytes = pngBytes(),
                onToken = { _, _, text -> tokenTexts += text },
                onComplete = { _, tokenCount, promptTokens, _, _ ->
                    assertEquals(2, tokenCount)
                    observedPromptTokens = promptTokens
                },
                onError = { error, _ -> fail("Unexpected error: $error") },
            )

            assertTrue(tokenTexts.isNotEmpty())
            assertTrue(observedPromptTokens > 3)
            encoder.close()
        } catch (_: UnsatisfiedLinkError) {
            // Local JVM unit tests do not always load JNI artifacts; treat that as a skip.
        }
    }

    private fun makeSession(engine: VisionTestMockLlamaEngine, visionEncoder: VisionEncoderEngine): LlamaSession {
        engine.tokenizeResult = intArrayOf(11, 12, 13)
        engine.generateResult = GenerateEngineResult(intArrayOf(101, 102), StopReason.MAX_TOKENS)
        engine.streamingTokens = intArrayOf(101, 102)
        engine.detokenizeMap = mapOf(
            listOf(101) to "o",
            listOf(101, 102) to "ok",
        )

        return LlamaSession(
            "vision-session",
            engine,
            visionEncoder,
            LLMModelMetadata(name = "vision", chatTemplate = null, hasVision = true),
            SessionPriority.INTERACTIVE,
        )
    }

    private fun pngBytes(): ByteArray {
        return byteArrayOf(
            137.toByte(), 80, 78, 71, 13, 10, 26, 10,
            0, 0, 0, 13, 73, 72, 68, 82,
            0, 0, 0, 1, 0, 0, 0, 1,
            8, 4, 0, 0, 0, 181.toByte(), 28, 12,
            2, 0, 0, 0, 11, 73, 68, 65,
            84, 120, 218.toByte(), 99, 252.toByte(), 255.toByte(), 31, 0,
            3, 3, 2, 0, 239.toByte(), 166.toByte(), 229.toByte(), 177.toByte(),
            0, 0, 0, 0, 73, 69, 78, 68,
            174.toByte(), 66, 96, 130.toByte(),
        )
    }
}

private class VisionTestMockLlamaEngine : LlamaEngine {
    override var nCtx: Int = 2048
    var tokenizeResult: IntArray = intArrayOf()
    var detokenizeMap: Map<List<Int>, String> = emptyMap()
    var generateResult: GenerateEngineResult = GenerateEngineResult(intArrayOf(), StopReason.MAX_TOKENS)
    var streamingTokens: IntArray = intArrayOf()
    var shouldCallVisionEval = true
    var lastVisionPromptTokens: List<Int>? = null

    override fun tokenize(text: String, addSpecial: Boolean): IntArray {
        return tokenizeResult
    }

    override fun detokenize(tokens: IntArray): String {
        return detokenizeMap[tokens.toList()] ?: ""
    }

    override fun generate(promptTokens: IntArray, maxTokens: Int, sampler: SamplerConfig): GenerateEngineResult {
        return generateResult
    }

    override fun generateWithVision(
        promptTokens: IntArray,
        imageEmbedding: ImageEmbedding,
        visionEncoder: VisionEncoderEngine,
        maxTokens: Int,
        sampler: SamplerConfig,
    ): GenerateEngineResult {
        lastVisionPromptTokens = promptTokens.toList()
        return generateResult
    }

    override fun generateStreaming(
        promptTokens: IntArray,
        maxTokens: Int,
        sampler: SamplerConfig,
        isCancelled: () -> Boolean,
        onToken: (Int) -> Unit,
    ): StopReason {
        for (token in streamingTokens) {
            if (isCancelled()) {
                return StopReason.CANCELLED
            }
            onToken(token)
        }
        return generateResult.stopReason
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
        generateWithVision(promptTokens, imageEmbedding, visionEncoder, maxTokens, sampler)
        return generateStreaming(promptTokens, maxTokens, sampler, isCancelled, onToken)
    }
}

private class MockVisionEncoderEngine(
    override val imageTokenCount: Int = 576,
    private val encodeError: Throwable? = null,
) : VisionEncoderEngine {

    override fun encode(imageBytes: ByteArray): ImageEmbedding {
        if (encodeError != null) {
            throw encodeError
        }

        return ImageEmbedding(chunksHandle = 1L, mtmdCtxHandle = 1L, tokenCount = imageTokenCount)
    }

    override fun freeEmbedding(embedding: ImageEmbedding) = Unit

    override fun close() = Unit
}
