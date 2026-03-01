package io.t6x.dust.llm

import io.t6x.dust.core.DustCoreError
import java.io.File

interface VisionEncoderEngine {
    val imageTokenCount: Int

    fun encode(imageBytes: ByteArray): ImageEmbedding

    fun evalImageEmbed(embedding: ImageEmbedding, llamaHandle: Long, batchSize: Int, nPast: IntArray)

    fun freeEmbedding(embedding: ImageEmbedding)

    fun close()
}

data class ImageEmbedding(
    val nativeHandle: Long,
    val tokenCount: Int,
)

class VisionEncoder(mmprojPath: String) : VisionEncoderEngine, AutoCloseable {
    private var clipHandle: Long

    init {
        if (!File(mmprojPath).exists()) {
            throw DustCoreError.InferenceFailed("mmproj not found at $mmprojPath")
        }

        clipHandle = LlamaJNI.nativeClipLoad(mmprojPath, 1)
        if (clipHandle == 0L) {
            throw DustCoreError.InferenceFailed("Failed to load CLIP model: $mmprojPath")
        }
    }

    override val imageTokenCount: Int
        get() = if (clipHandle == 0L) {
            0
        } else {
            LlamaJNI.nativeClipImageTokenCount(clipHandle)
        }

    override fun encode(imageBytes: ByteArray): ImageEmbedding {
        if (clipHandle == 0L) {
            throw LlamaError.ModelEvicted()
        }

        val embedHandle = LlamaJNI.nativeClipEncodeImage(clipHandle, imageBytes, 4)
        if (embedHandle == 0L) {
            throw DustCoreError.InferenceFailed("Failed to encode image")
        }

        return ImageEmbedding(
            nativeHandle = embedHandle,
            tokenCount = imageTokenCount,
        )
    }

    override fun evalImageEmbed(embedding: ImageEmbedding, llamaHandle: Long, batchSize: Int, nPast: IntArray) {
        val ok = LlamaJNI.nativeClipEvalImageEmbed(
            llamaHandle = llamaHandle,
            embedHandle = embedding.nativeHandle,
            batchSize = batchSize,
            nPast = nPast,
        )

        if (!ok) {
            throw DustCoreError.InferenceFailed("Failed to evaluate image embedding")
        }
    }

    override fun freeEmbedding(embedding: ImageEmbedding) {
        if (embedding.nativeHandle != 0L) {
            LlamaJNI.nativeClipFreeEmbed(embedding.nativeHandle)
        }
    }

    override fun close() {
        val handle = clipHandle
        if (handle != 0L) {
            clipHandle = 0L
            LlamaJNI.nativeClipFree(handle)
        }
    }
}
