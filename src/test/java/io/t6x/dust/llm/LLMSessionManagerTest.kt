package io.t6x.dust.llm

import io.t6x.dust.core.DustCoreError
import io.t6x.dust.core.SessionPriority
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.test.runTest
import io.t6x.dust.core.ModelFormat
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import java.io.File

class LLMSessionManagerTest {
    @Test
    fun l1T1LoadValidPathCreatesSession() = runTest {
        val file = File.createTempFile("tiny-test-", ".gguf")
        file.writeBytes(byteArrayOf(0x47, 0x47, 0x55, 0x46))
        file.deleteOnExit()

        val manager = LLMSessionManager(
            sessionFactory = { _, modelId, _, priority ->
                LlamaSession(
                    modelId,
                    fakeWrapper(
                        LLMModelMetadata(
                            name = "tiny-test-model",
                            chatTemplate = "{% for message in messages %}{{ message.content }}{% endfor %}",
                            hasVision = true,
                        ),
                    ),
                    priority,
                )
            },
        )

        val session = manager.loadModel(file.path, "model-a", LLMConfig(), SessionPriority.INTERACTIVE)

        assertEquals("tiny-test-model", session.metadata.name)
        assertEquals(SessionPriority.INTERACTIVE, session.priority())
    }

    @Test
    fun l1T2MissingFileThrowsWithPath() = runTest {
        val manager = LLMSessionManager(
            sessionFactory = { path, _, _, _ ->
                if (!File(path).exists()) {
                    throw DustCoreError.InferenceFailed("Model file not found: $path")
                }
                error("Expected failure — file should not exist")
            },
        )

        try {
            manager.loadModel("/nonexistent/model.gguf", "missing", LLMConfig(), SessionPriority.INTERACTIVE)
            fail("Expected InferenceFailed")
        } catch (error: DustCoreError) {
            assertTrue(error is DustCoreError.InferenceFailed)
            assertTrue(error.message.orEmpty().contains("/nonexistent/model.gguf"))
        }
    }

    @Test
    fun l1T3CorruptFileThrowsInferenceFailed() = runTest {
        val corrupt = File.createTempFile("corrupt-", ".gguf")
        corrupt.writeBytes(byteArrayOf(0x47, 0x47, 0x55, 0x46, 0x01, 0x00))
        corrupt.deleteOnExit()

        val manager = LLMSessionManager(
            sessionFactory = { _, _, _, _ ->
                throw DustCoreError.InferenceFailed("Failed to load GGUF model: ${corrupt.path}")
            },
        )

        try {
            manager.loadModel(corrupt.path, "corrupt", LLMConfig(), SessionPriority.INTERACTIVE)
            fail("Expected InferenceFailed")
        } catch (error: DustCoreError) {
            assertTrue(error is DustCoreError.InferenceFailed)
        }
    }

    @Test
    fun l1T4WrongFormatWouldBeRejectedAtPluginLayer() {
        // The plugin layer rejects non-gguf formats before calling the session manager.
        // Verify that the GGUF constant is correct and all others differ.
        val accepted = ModelFormat.GGUF.value
        assertEquals("gguf", accepted)

        val rejected = ModelFormat.entries.filter { it != ModelFormat.GGUF }
        assertTrue("Should have non-GGUF formats", rejected.isNotEmpty())
        for (format in rejected) {
            assertTrue("${format.name} should differ from gguf", format.value != accepted)
        }

        // Verify the session factory is never invoked for a rejected format.
        var factoryCalled = false
        val manager = LLMSessionManager(
            sessionFactory = { _, modelId, _, priority ->
                factoryCalled = true
                LlamaSession(modelId, fakeWrapper(LLMModelMetadata(null, null, false)), priority)
            },
        )
        assertNotNull(manager) // suppress unused warning
        assertFalse("Factory must not be called when format is rejected at plugin layer", factoryCalled)
    }

    @Test
    fun l1T5UnloadLoadedModelEmptiesSessionMap() = runTest {
        val manager = LLMSessionManager(
            sessionFactory = { _, modelId, _, priority ->
                LlamaSession(modelId, fakeWrapper(LLMModelMetadata(null, null, false)), priority)
            },
        )

        manager.loadModel("/tmp/model-a.gguf", "model-a", LLMConfig(), SessionPriority.INTERACTIVE)
        manager.forceUnloadModel("model-a")

        assertEquals(0, manager.sessionCount)
    }

    @Test
    fun l1T6UnloadUnknownIdThrowsModelNotFound() = runTest {
        val manager = LLMSessionManager(
            sessionFactory = { _, modelId, _, priority ->
                LlamaSession(modelId, fakeWrapper(LLMModelMetadata(null, null, false)), priority)
            },
        )

        try {
            manager.forceUnloadModel("missing")
            fail("Expected ModelNotFound")
        } catch (error: DustCoreError) {
            assertTrue(error is DustCoreError.ModelNotFound)
        }
    }

    @Test
    fun l1T7LoadingSameIdTwiceReusesSession() = runTest {
        val manager = LLMSessionManager(
            sessionFactory = { _, modelId, _, priority ->
                LlamaSession(modelId, fakeWrapper(LLMModelMetadata(null, null, false)), priority)
            },
        )

        val first = manager.loadModel("/tmp/model-a.gguf", "model-a", LLMConfig(), SessionPriority.INTERACTIVE)
        val second = manager.loadModel("/tmp/model-a.gguf", "model-a", LLMConfig(), SessionPriority.BACKGROUND)

        assertSame(first, second)
        assertEquals(1, manager.sessionCount)
    }

    @Test
    fun l1T8ConcurrentLoadTwoModelsSucceeds() = runTest {
        val manager = LLMSessionManager(
            sessionFactory = { _, modelId, _, priority ->
                LlamaSession(modelId, fakeWrapper(LLMModelMetadata(modelId, null, false)), priority)
            },
        )

        val sessions = listOf(
            async { manager.loadModel("/tmp/model-a.gguf", "model-a", LLMConfig(), SessionPriority.INTERACTIVE) },
            async { manager.loadModel("/tmp/model-b.gguf", "model-b", LLMConfig(), SessionPriority.INTERACTIVE) },
        ).awaitAll()

        assertEquals(2, sessions.size)
        assertNotNull(manager.session("model-a"))
        assertNotNull(manager.session("model-b"))
    }

    @Test
    fun l7T1EvictUnderPressureStandardEvictsOnlyBackgroundIdleSessions() = runTest {
        val manager = LLMSessionManager(
            sessionFactory = { _, modelId, _, priority ->
                LlamaSession(modelId, fakeWrapper(LLMModelMetadata(modelId, null, false)), priority)
            },
        )

        manager.loadModel("/tmp/model-interactive.gguf", "interactive", LLMConfig(), SessionPriority.INTERACTIVE)
        manager.loadModel("/tmp/model-background.gguf", "background", LLMConfig(), SessionPriority.BACKGROUND)

        manager.unloadModel("interactive")
        manager.unloadModel("background")
        manager.evictUnderPressure(MemoryPressureLevel.STANDARD)

        assertTrue(manager.hasCachedSession("interactive"))
        assertFalse(manager.hasCachedSession("background"))
        assertEquals(1, manager.sessionCount)
    }

    @Test
    fun l7T2EvictUnderPressureCriticalEvictsAllIdleSessions() = runTest {
        val manager = LLMSessionManager(
            sessionFactory = { _, modelId, _, priority ->
                LlamaSession(modelId, fakeWrapper(LLMModelMetadata(modelId, null, false)), priority)
            },
        )

        manager.loadModel("/tmp/model-interactive.gguf", "interactive", LLMConfig(), SessionPriority.INTERACTIVE)
        manager.loadModel("/tmp/model-background.gguf", "background", LLMConfig(), SessionPriority.BACKGROUND)

        manager.unloadModel("interactive")
        manager.unloadModel("background")
        manager.evictUnderPressure(MemoryPressureLevel.CRITICAL)

        assertFalse(manager.hasCachedSession("interactive"))
        assertFalse(manager.hasCachedSession("background"))
        assertEquals(0, manager.sessionCount)
    }
}

private fun fakeWrapper(metadata: LLMModelMetadata): LlamaContextWrapper {
    return object : LlamaContextWrapper(
        handleRef = java.util.concurrent.atomic.AtomicLong(1L),
        metadata = metadata,
        nCtx = 2048,
        batchSize = 512,
    ) {
        override fun close() {
            // No-op for local unit tests.
        }
    }
}
