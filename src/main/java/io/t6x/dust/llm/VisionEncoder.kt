package io.t6x.dust.llm

import io.t6x.dust.core.DustCoreError
import java.io.File

interface VisionEncoderEngine {
    val imageTokenCount: Int

    fun encode(imageBytes: ByteArray): ImageEmbedding

    fun freeEmbedding(embedding: ImageEmbedding)

    fun close()
}

data class ImageEmbedding(
    val chunksHandle: Long,
    val mtmdCtxHandle: Long,
    val tokenCount: Int,
)

class VisionEncoder(mmprojPath: String, llamaHandle: Long) : VisionEncoderEngine, AutoCloseable {
    private var mtmdHandle: Long

    init {
        if (!File(mmprojPath).exists()) {
            throw DustCoreError.InferenceFailed("mmproj not found at $mmprojPath")
        }

        mtmdHandle = LlamaJNI.nativeMtmdLoad(mmprojPath, llamaHandle)
        if (mtmdHandle == 0L) {
            throw DustCoreError.InferenceFailed("Failed to load mtmd model: $mmprojPath")
        }
    }

    override val imageTokenCount: Int
        get() = 0  // With mtmd, token count is determined per-image during tokenization

    override fun encode(imageBytes: ByteArray): ImageEmbedding {
        if (mtmdHandle == 0L) {
            throw LlamaError.ModelEvicted()
        }

        val chunksHandle = LlamaJNI.nativeMtmdEncodeImage(mtmdHandle, imageBytes)
        if (chunksHandle == 0L) {
            throw DustCoreError.InferenceFailed("Failed to encode image")
        }

        val tokenCount = LlamaJNI.nativeMtmdGetTokenCount(chunksHandle)

        return ImageEmbedding(
            chunksHandle = chunksHandle,
            mtmdCtxHandle = mtmdHandle,
            tokenCount = tokenCount,
        )
    }

    override fun freeEmbedding(embedding: ImageEmbedding) {
        if (embedding.chunksHandle != 0L) {
            LlamaJNI.nativeMtmdFreeChunks(embedding.chunksHandle)
        }
    }

    override fun close() {
        val handle = mtmdHandle
        if (handle != 0L) {
            mtmdHandle = 0L
            LlamaJNI.nativeMtmdFree(handle)
        }
    }
}
