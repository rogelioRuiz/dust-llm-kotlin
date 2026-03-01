package io.t6x.dust.llm

import io.t6x.dust.core.DustCoreError
import io.t6x.dust.core.ModelDescriptor
import io.t6x.dust.core.ModelFormat
import io.t6x.dust.core.ModelServer
import io.t6x.dust.core.ModelSession
import io.t6x.dust.core.ModelStatus
import io.t6x.dust.core.SessionPriority
import java.io.File
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

enum class MemoryPressureLevel {
    STANDARD,
    CRITICAL,
}

class LLMSessionManager(
    private val sessionFactory: ((path: String, modelId: String, config: LLMConfig, priority: SessionPriority) -> LlamaSession)? = null,
    private val visionEncoderFactory: (mmprojPath: String) -> VisionEncoderEngine = { mmprojPath ->
        VisionEncoder(mmprojPath)
    },
) : ModelServer {
    private val lock = ReentrantLock()
    private val descriptors = mutableMapOf<String, ModelDescriptor>()
    private val statuses = mutableMapOf<String, ModelStatus>()
    private val configs = mutableMapOf<String, LLMConfig>()
    private val cachedSessions = mutableMapOf<String, CachedSession>()

    fun register(descriptor: ModelDescriptor, config: LLMConfig? = null) {
        val status = initialStatus(descriptor)

        lock.withLock {
            descriptors[descriptor.id] = descriptor
            statuses[descriptor.id] = status
            configs[descriptor.id] = config ?: configs[descriptor.id] ?: LLMConfig()
        }
    }

    fun setStatus(status: ModelStatus, id: String) {
        lock.withLock {
            statuses[id] = status
        }
    }

    override suspend fun loadModel(descriptor: ModelDescriptor, priority: SessionPriority): ModelSession {
        val config = lock.withLock {
            if (!descriptors.containsKey(descriptor.id)) {
                throw DustCoreError.ModelNotFound
            }
            configs[descriptor.id] ?: LLMConfig()
        }

        return loadModelWithConfig(descriptor, config, priority)
    }

    fun loadModelWithConfig(
        descriptor: ModelDescriptor,
        config: LLMConfig,
        priority: SessionPriority,
    ): LlamaSession {
        incrementCachedRefCount(descriptor.id)?.let { return it }

        val (registeredDescriptor, status) = lock.withLock {
            descriptors[descriptor.id] to (statuses[descriptor.id] ?: ModelStatus.NotLoaded)
        }

        val storedDescriptor = registeredDescriptor ?: throw DustCoreError.ModelNotFound
        if (status != ModelStatus.Ready) {
            throw DustCoreError.ModelNotReady
        }

        val path = resolvedPath(storedDescriptor)
            ?: throw DustCoreError.InvalidInput("descriptor.url or descriptor.metadata.localPath is required")

        val created = createSession(path, descriptor.id, config, priority)

        val winner = lock.withLock {
            val existing = cachedSessions[descriptor.id]
            if (existing != null) {
                existing.refCount += 1
                existing.lastAccessTime = System.nanoTime()
                existing.session
            } else {
                cachedSessions[descriptor.id] = CachedSession(
                    session = created,
                    priority = priority,
                    refCount = 1,
                    lastAccessTime = System.nanoTime(),
                )
                created
            }.also {
                statuses[descriptor.id] = ModelStatus.Ready
                configs[descriptor.id] = config
            }
        }

        if (winner !== created) {
            created.closeContext()
        }

        return winner
    }

    fun loadModel(
        path: String,
        modelId: String,
        config: LLMConfig,
        priority: SessionPriority,
    ): LlamaSession {
        val descriptor = legacyDescriptor(path, modelId)
        register(descriptor, config)
        setStatus(ModelStatus.Ready, modelId)
        return loadModelWithConfig(descriptor, config, priority)
    }

    override suspend fun unloadModel(id: String) {
        val didDecrement = lock.withLock {
            val cached = cachedSessions[id]
            if (cached == null || cached.refCount == 0) {
                false
            } else {
                cached.refCount -= 1
                cached.lastAccessTime = System.nanoTime()
                true
            }
        }

        if (!didDecrement) {
            throw DustCoreError.ModelNotFound
        }
    }

    suspend fun forceUnloadModel(id: String) {
        val session = lock.withLock { cachedSessions.remove(id)?.session } ?: throw DustCoreError.ModelNotFound
        session.close()
    }

    suspend fun evict(modelId: String): LlamaSession? {
        val session = lock.withLock { cachedSessions.remove(modelId)?.session }
        session?.evict()
        return session
    }

    suspend fun evictUnderPressure(level: MemoryPressureLevel) {
        val evicted = lock.withLock {
            val eligible = cachedSessions.filter { (_, cached) ->
                cached.refCount == 0 && when (level) {
                    MemoryPressureLevel.STANDARD -> cached.priority == SessionPriority.BACKGROUND
                    MemoryPressureLevel.CRITICAL -> true
                }
            }
            val sorted = eligible.entries.sortedBy { it.value.lastAccessTime }
            val sessions = sorted.map { it.value.session }
            for ((id, _) in sorted) {
                cachedSessions.remove(id)
            }
            sessions
        }

        for (session in evicted) {
            session.evict()
        }
    }

    override suspend fun listModels(): List<ModelDescriptor> = allDescriptors()

    override suspend fun modelStatus(id: String): ModelStatus = lock.withLock {
        statuses[id] ?: ModelStatus.NotLoaded
    }

    fun refCount(id: String): Int = lock.withLock {
        cachedSessions[id]?.refCount ?: 0
    }

    fun hasCachedSession(id: String): Boolean = lock.withLock {
        cachedSessions.containsKey(id)
    }

    fun session(id: String): LlamaSession? = lock.withLock {
        cachedSessions[id]?.session
    }

    fun allModelIds(): List<String> = lock.withLock { cachedSessions.keys.sorted() }

    fun allDescriptors(): List<ModelDescriptor> = lock.withLock {
        descriptors.values.sortedBy { it.id }
    }

    val sessionCount: Int
        get() = lock.withLock { cachedSessions.size }

    private fun incrementCachedRefCount(id: String): LlamaSession? {
        return lock.withLock {
            val cached = cachedSessions[id] ?: return null
            cached.refCount += 1
            cached.lastAccessTime = System.nanoTime()
            cached.session
        }
    }

    private fun initialStatus(descriptor: ModelDescriptor): ModelStatus {
        val path = resolvedPath(descriptor) ?: return ModelStatus.NotLoaded
        return if (File(path).exists()) {
            ModelStatus.Ready
        } else {
            ModelStatus.NotLoaded
        }
    }

    private fun resolvedPath(descriptor: ModelDescriptor): String? {
        val localPath = descriptor.metadata?.get("localPath")
        if (!localPath.isNullOrEmpty()) {
            return localPath
        }

        if (!descriptor.url.isNullOrEmpty()) {
            return descriptor.url
        }

        return null
    }

    private fun legacyDescriptor(path: String, modelId: String): ModelDescriptor {
        val file = File(path)
        val sizeBytes = if (file.exists()) file.length() else 0L

        return ModelDescriptor(
            id = modelId,
            name = modelId,
            format = ModelFormat.GGUF,
            sizeBytes = sizeBytes,
            version = "legacy",
            url = path,
        )
    }

    private fun createSession(
        path: String,
        modelId: String,
        config: LLMConfig,
        priority: SessionPriority,
    ): LlamaSession {
        sessionFactory?.let { factory ->
            return factory(path, modelId, config, priority)
        }

        val context = LlamaContextWrapper.load(path, config)
        val visionEncoder = if (context.metadata.hasVision) {
            visionEncoderFactory(resolveMMProjPath(path, config))
        } else {
            null
        }

        return LlamaSession(modelId, context, priority, visionEncoder)
    }

    private fun resolveMMProjPath(modelPath: String, config: LLMConfig): String {
        val explicit = config.mmprojPath
        if (!explicit.isNullOrEmpty()) {
            return explicit
        }

        return if (modelPath.endsWith(".gguf")) {
            modelPath.removeSuffix(".gguf") + "-mmproj.gguf"
        } else {
            "$modelPath-mmproj.gguf"
        }
    }
}

private data class CachedSession(
    val session: LlamaSession,
    val priority: SessionPriority,
    var refCount: Int,
    var lastAccessTime: Long,
)
