package io.t6x.dust.llm

import io.t6x.dust.core.DustCoreError
import io.t6x.dust.core.SessionPriority
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

class LLMChatTemplateTest {
    @Test
    fun l4T1ChatMLTemplateAppliesThreeMessageConversation() {
        val engine = ChatTemplateEngine(ChatTemplateEngine.CHAT_ML_TEMPLATE)
        val messages = listOf(
            ChatMessage(role = "system", content = "You are a helpful assistant"),
            ChatMessage(role = "user", content = "Hello"),
            ChatMessage(role = "assistant", content = "Hi there"),
        )

        val output = engine.apply(messages, addGenerationPrompt = true)

        assertEquals(
            "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi there<|im_end|>\n<|im_start|>assistant\n",
            output,
        )
    }

    @Test
    fun l4T2NilTemplateFallsBackToChatML() {
        val engine = ChatTemplateEngine(null)

        val output = engine.apply(
            messages = listOf(ChatMessage(role = "user", content = "Hello")),
            addGenerationPrompt = true,
        )

        assertTrue(output.contains("<|im_start|>"))
        assertTrue(output.contains("<|im_end|>"))
        assertTrue(output.contains("Hello"))
    }

    @Test
    fun l4T3TrimHistoryPreservesSystemAndEvictsOldestPair() {
        val engine = ChatTemplateMockLlamaEngine().apply {
            nCtx = 200
            tokenizeHandler = { text, _ -> IntArray(text.toByteArray(Charsets.UTF_8).size) { 1 } }
            val completions = mapOf(11 to "B", 12 to "D", 13 to "F")
            detokenizeHandler = { tokens -> completions[tokens.lastOrNull()] ?: "" }
            var nextToken = 11
            generateHandler = { _, _ ->
                GenerateEngineResult(intArrayOf(nextToken++), StopReason.MAX_TOKENS)
            }
        }
        val session = makeSession(engine)

        session.generateChat(
            messages = listOf(
                ChatMessage(role = "system", content = "Hi"),
                ChatMessage(role = "user", content = "A"),
            ),
            maxTokens = 10,
            stopSequences = emptyList(),
            sampler = SamplerConfig(),
        )
        session.generateChat(
            messages = listOf(ChatMessage(role = "user", content = "C")),
            maxTokens = 10,
            stopSequences = emptyList(),
            sampler = SamplerConfig(),
        )
        session.generateChat(
            messages = listOf(ChatMessage(role = "user", content = "E")),
            maxTokens = 10,
            stopSequences = emptyList(),
            sampler = SamplerConfig(),
        )

        val trimmedPrompt = engine.generatePromptTexts.last()
        assertTrue(trimmedPrompt.contains("<|im_start|>system\nHi"))
        assertFalse(trimmedPrompt.contains("<|im_start|>user\nA"))
        assertFalse(trimmedPrompt.contains("<|im_start|>assistant\nB"))
        assertTrue(trimmedPrompt.contains("<|im_start|>user\nC"))
        assertTrue(trimmedPrompt.contains("<|im_start|>assistant\nD"))
        assertTrue(trimmedPrompt.contains("<|im_start|>user\nE"))
    }

    @Test
    fun l4T4SingleMessageExactFitSucceeds() {
        val engine = ChatTemplateMockLlamaEngine().apply {
            nCtx = 100
            tokenizeHandler = { _, _ -> IntArray(80) { 1 } }
            generateResult = GenerateEngineResult(intArrayOf(21), StopReason.MAX_TOKENS)
            detokenizeHandler = { "ok" }
        }
        val session = makeSession(engine)

        val result = session.generateChat(
            messages = listOf(ChatMessage(role = "user", content = "test")),
            maxTokens = 20,
            stopSequences = emptyList(),
            sampler = SamplerConfig(),
        )

        assertEquals("ok", result.first.text)
        assertEquals(StopReason.MAX_TOKENS, result.first.stopReason)
    }

    @Test
    fun l4T5SingleMessageOverflowThrowsContextOverflowEquivalent() {
        val engine = ChatTemplateMockLlamaEngine().apply {
            nCtx = 50
            tokenizeHandler = { _, _ -> IntArray(60) { 1 } }
        }
        val session = makeSession(engine)

        try {
            session.generateChat(
                messages = listOf(ChatMessage(role = "user", content = "huge")),
                maxTokens = 10,
                stopSequences = emptyList(),
                sampler = SamplerConfig(),
            )
            fail("Expected InvalidInput")
        } catch (error: DustCoreError) {
            require(error is DustCoreError.InvalidInput)
            assertEquals("Invalid input: Prompt has 60 tokens but context size is 50", error.message)
        }
    }

    @Test
    fun l4T6ClearHistoryResetsContextUsed() {
        val engine = ChatTemplateMockLlamaEngine().apply {
            tokenizeHandler = { text, _ -> IntArray(text.toByteArray(Charsets.UTF_8).size) { 1 } }
            val completions = mapOf(31 to "Hello", 32 to "Again")
            detokenizeHandler = { tokens -> completions[tokens.lastOrNull()] ?: "" }
            var nextToken = 31
            generateHandler = { _, _ ->
                GenerateEngineResult(intArrayOf(nextToken++), StopReason.MAX_TOKENS)
            }
        }
        val session = makeSession(engine)

        session.generateChat(
            messages = listOf(ChatMessage(role = "user", content = "Hi")),
            maxTokens = 8,
            stopSequences = emptyList(),
            sampler = SamplerConfig(),
        )

        assertTrue(session.contextUsed > 0)

        session.clearHistory()

        assertEquals(0, session.contextUsed)

        val secondTurn = session.generateChat(
            messages = listOf(ChatMessage(role = "user", content = "Fresh")),
            maxTokens = 8,
            stopSequences = emptyList(),
            sampler = SamplerConfig(),
        )

        assertEquals("Again", secondTurn.first.text)
        assertTrue(session.contextUsed > 0)
    }

    @Test
    fun l4T7ContextUsedIncreasesAcrossTurns() {
        val engine = ChatTemplateMockLlamaEngine().apply {
            nCtx = 512
            tokenizeHandler = { text, _ -> IntArray(text.toByteArray(Charsets.UTF_8).size) { 1 } }
            val completions = mapOf(41 to "Hi", 42 to "Fine")
            detokenizeHandler = { tokens -> completions[tokens.lastOrNull()] ?: "" }
            var nextToken = 41
            generateHandler = { _, _ ->
                GenerateEngineResult(intArrayOf(nextToken++), StopReason.MAX_TOKENS)
            }
        }
        val session = makeSession(engine)

        val turnOne = session.generateChat(
            messages = listOf(ChatMessage(role = "user", content = "Hello")),
            maxTokens = 8,
            stopSequences = emptyList(),
            sampler = SamplerConfig(),
        )
        val turnTwo = session.generateChat(
            messages = listOf(ChatMessage(role = "user", content = "How are you?")),
            maxTokens = 8,
            stopSequences = emptyList(),
            sampler = SamplerConfig(),
        )

        assertTrue(turnOne.second > 0)
        assertTrue(turnTwo.second > turnOne.second)
    }

    @Test
    fun l4T8AddGenerationPromptAddsAssistantPrefix() {
        val engine = ChatTemplateEngine(ChatTemplateEngine.CHAT_ML_TEMPLATE)
        val messages = listOf(ChatMessage(role = "user", content = "Hello"))

        val withPrompt = engine.apply(messages, addGenerationPrompt = true)
        val withoutPrompt = engine.apply(messages, addGenerationPrompt = false)

        assertTrue(withPrompt.endsWith("<|im_start|>assistant\n"))
        assertFalse(withoutPrompt.endsWith("<|im_start|>assistant\n"))
    }

    private fun makeSession(engine: ChatTemplateMockLlamaEngine): LlamaSession {
        return LlamaSession(
            sessionId = "chat-template-session",
            engine = engine,
            metadata = LLMModelMetadata(name = "mock", chatTemplate = null, hasVision = false),
            sessionPriority = SessionPriority.INTERACTIVE,
        )
    }
}

private class ChatTemplateMockLlamaEngine : LlamaEngine {
    override var nCtx: Int = 256
    var tokenizeResult: IntArray = intArrayOf()
    var detokenizeResult: String = ""
    var generateResult: GenerateEngineResult = GenerateEngineResult(intArrayOf(), StopReason.MAX_TOKENS)

    var tokenizeHandler: ((String, Boolean) -> IntArray)? = null
    var detokenizeHandler: ((IntArray) -> String)? = null
    var generateHandler: ((IntArray, Int) -> GenerateEngineResult)? = null

    var lastTokenizeText: String? = null
    val generatePromptTexts = mutableListOf<String>()

    override fun tokenize(text: String, addSpecial: Boolean): IntArray {
        lastTokenizeText = text
        return tokenizeHandler?.invoke(text, addSpecial) ?: tokenizeResult
    }

    override fun detokenize(tokens: IntArray): String {
        return detokenizeHandler?.invoke(tokens) ?: detokenizeResult
    }

    override fun generate(promptTokens: IntArray, maxTokens: Int, sampler: SamplerConfig): GenerateEngineResult {
        generatePromptTexts += lastTokenizeText.orEmpty()
        return generateHandler?.invoke(promptTokens, maxTokens) ?: generateResult
    }

    override fun generateStreaming(
        promptTokens: IntArray,
        maxTokens: Int,
        sampler: SamplerConfig,
        isCancelled: () -> Boolean,
        onToken: (Int) -> Unit,
    ): StopReason {
        return StopReason.MAX_TOKENS
    }
}
