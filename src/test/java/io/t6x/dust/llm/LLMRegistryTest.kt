package io.t6x.dust.llm

import io.t6x.dust.core.DustCoreError
import io.t6x.dust.core.DustCoreRegistry
import io.t6x.dust.core.ModelDescriptor
import io.t6x.dust.core.ModelFormat
import io.t6x.dust.core.ModelStatus
import io.t6x.dust.core.SessionPriority
import kotlinx.coroutines.test.runTest
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Before
import org.junit.Test
import java.io.File

class LLMRegistryTest {

    @Before
    fun setUp() {
        DustCoreRegistry.resetForTesting()
    }

    @After
    fun tearDown() {
        DustCoreRegistry.resetForTesting()
    }

    @Test
    fun l5T1RegistryRegistrationMakesManagerResolvable() {
        val manager = makeManager()

        DustCoreRegistry.getInstance().registerModelServer(manager)

        assertSame(manager, DustCoreRegistry.getInstance().resolveModelServer())
    }

    @Test
    fun l5T2LoadModelForReadyDescriptorCreatesSessionAndRefCount() = runTest {
        val file = makeTempModelFile()
        val manager = makeManager()
        val descriptor = makeDescriptor("model-a", file.path)
        manager.register(descriptor)

        val session = manager.loadModel(descriptor, SessionPriority.INTERACTIVE)

        assertEquals(ModelStatus.Ready, session.status())
        assertEquals(1, manager.refCount("model-a"))

        file.delete()
    }

    @Test
    fun l5T3LoadModelForNotLoadedDescriptorThrowsModelNotReady() = runTest {
        val manager = makeManager()
        val descriptor = makeDescriptor("model-a", "/tmp/missing-model.gguf")
        manager.register(descriptor)

        try {
            manager.loadModel(descriptor, SessionPriority.INTERACTIVE)
            fail("Expected ModelNotReady")
        } catch (error: DustCoreError) {
            assertTrue(error is DustCoreError.ModelNotReady)
        }
    }

    @Test
    fun l5T4LoadModelForUnregisteredIdThrowsModelNotFound() = runTest {
        val manager = makeManager()
        val descriptor = makeDescriptor("ghost", "/tmp/ghost.gguf")

        try {
            manager.loadModel(descriptor, SessionPriority.INTERACTIVE)
            fail("Expected ModelNotFound")
        } catch (error: DustCoreError) {
            assertTrue(error is DustCoreError.ModelNotFound)
        }
    }

    @Test
    fun l5T5UnloadModelDecrementsRefCountAndKeepsSessionCached() = runTest {
        val file = makeTempModelFile()
        val manager = makeManager()
        val descriptor = makeDescriptor("model-a", file.path)
        manager.register(descriptor)

        manager.loadModel(descriptor, SessionPriority.INTERACTIVE)
        manager.unloadModel("model-a")

        assertEquals(0, manager.refCount("model-a"))
        assertTrue(manager.hasCachedSession("model-a"))

        file.delete()
    }

    @Test
    fun l5T6LoadModelTwiceReusesSameSessionAndIncrementsRefCount() = runTest {
        val file = makeTempModelFile()
        val manager = makeManager()
        val descriptor = makeDescriptor("model-a", file.path)
        manager.register(descriptor)

        val first = manager.loadModel(descriptor, SessionPriority.INTERACTIVE)
        val second = manager.loadModel(descriptor, SessionPriority.BACKGROUND)

        assertSame(first, second)
        assertEquals(2, manager.refCount("model-a"))

        file.delete()
    }

    @Test
    fun l5T7EvictOnZeroRefCountSessionInvalidatesSession() = runTest {
        val file = makeTempModelFile()
        val manager = makeManager()
        val descriptor = makeDescriptor("model-a", file.path)
        manager.register(descriptor)

        val session = manager.loadModel(descriptor, SessionPriority.INTERACTIVE) as LlamaSession
        manager.unloadModel("model-a")
        manager.evict("model-a")

        assertTrue(session.isModelEvicted)
        assertFalse(manager.hasCachedSession("model-a"))

        file.delete()
    }

    @Test
    fun l5T8GenerateOnEvictedSessionThrowsModelEvicted() = runTest {
        val file = makeTempModelFile()
        val manager = makeManager(useWrapperSession = true)
        val descriptor = makeDescriptor("model-a", file.path)
        manager.register(descriptor)

        val session = manager.loadModel(descriptor, SessionPriority.INTERACTIVE) as LlamaSession
        manager.unloadModel("model-a")
        manager.evict("model-a")

        try {
            session.generate("prompt", 1, emptyList(), SamplerConfig())
            fail("Expected ModelEvicted")
        } catch (error: LlamaError) {
            assertTrue(error is LlamaError.ModelEvicted)
        }

        file.delete()
    }

    @Test
    fun l5T9AllModelIdsReturnsOnlyLiveSessionsAfterEviction() = runTest {
        val fileA = makeTempModelFile()
        val fileB = makeTempModelFile()
        val manager = makeManager()
        val descriptorA = makeDescriptor("model-a", fileA.path)
        val descriptorB = makeDescriptor("model-b", fileB.path)
        manager.register(descriptorA)
        manager.register(descriptorB)

        manager.loadModel(descriptorA, SessionPriority.INTERACTIVE)
        manager.loadModel(descriptorB, SessionPriority.INTERACTIVE)

        assertEquals(listOf("model-a", "model-b"), manager.allModelIds())

        manager.unloadModel("model-a")
        manager.evict("model-a")

        assertEquals(listOf("model-b"), manager.allModelIds())

        fileA.delete()
        fileB.delete()
    }

    private fun makeManager(useWrapperSession: Boolean = false): LLMSessionManager {
        return LLMSessionManager(
            sessionFactory = { _, modelId, _, priority ->
                if (useWrapperSession) {
                    LlamaSession(modelId, registryFakeWrapper(LLMModelMetadata(modelId, null, false)), priority)
                } else {
                    LlamaSession(
                        sessionId = modelId,
                        metadata = LLMModelMetadata(modelId, null, false),
                        sessionPriority = priority,
                    )
                }
            },
        )
    }

    private fun makeDescriptor(id: String, path: String): ModelDescriptor {
        return ModelDescriptor(
            id = id,
            name = id,
            format = ModelFormat.GGUF,
            sizeBytes = 1,
            version = "1.0.0",
            metadata = mapOf("localPath" to path),
        )
    }

    private fun makeTempModelFile(): File {
        val file = File.createTempFile("llm-registry-", ".gguf")
        file.writeBytes(byteArrayOf(0x47, 0x47, 0x55, 0x46))
        file.deleteOnExit()
        return file
    }
}

private fun registryFakeWrapper(metadata: LLMModelMetadata): LlamaContextWrapper {
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
