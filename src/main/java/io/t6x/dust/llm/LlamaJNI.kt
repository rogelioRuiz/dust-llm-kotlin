package io.t6x.dust.llm

fun interface StreamingCallback {
    fun onToken(tokenId: Int): Boolean
}

object LlamaJNI {
    init {
        System.loadLibrary("llama_jni")
    }

    external fun nativeLoad(path: String, nGpuLayers: Int, nCtx: Int, nBatch: Int): Long

    external fun nativeFree(handle: Long)

    external fun nativeGetMetadata(handle: Long, key: String): String?

    external fun nativeTokenize(handle: Long, text: String, addSpecial: Boolean): IntArray?

    external fun nativeDetokenize(handle: Long, tokens: IntArray): String?

    external fun nativeGetEmbedding(handle: Long, promptTokens: IntArray): FloatArray?

    external fun nativeGetEmbeddingDims(handle: Long): Int

    external fun nativeGenerate(
        handle: Long,
        promptTokens: IntArray,
        maxTokens: Int,
        temperature: Float,
        topK: Int,
        topP: Float,
        minP: Float,
        repeatPenalty: Float,
        repeatLastN: Int,
        seed: Int,
    ): Array<String>?

    external fun nativeGenerateWithVision(
        handle: Long,
        promptTokens: IntArray,
        mtmdCtxHandle: Long,
        chunksHandle: Long,
        maxTokens: Int,
        temperature: Float,
        topK: Int,
        topP: Float,
        minP: Float,
        repeatPenalty: Float,
        repeatLastN: Int,
        seed: Int,
    ): Array<String>?

    external fun nativeGenerateStreaming(
        handle: Long,
        promptTokens: IntArray,
        maxTokens: Int,
        temperature: Float,
        topK: Int,
        topP: Float,
        minP: Float,
        repeatPenalty: Float,
        repeatLastN: Int,
        seed: Int,
        callback: StreamingCallback,
    ): String?

    external fun nativeGenerateStreamingWithVision(
        handle: Long,
        promptTokens: IntArray,
        mtmdCtxHandle: Long,
        chunksHandle: Long,
        maxTokens: Int,
        temperature: Float,
        topK: Int,
        topP: Float,
        minP: Float,
        repeatPenalty: Float,
        repeatLastN: Int,
        seed: Int,
        callback: StreamingCallback,
    ): String?

    external fun nativeMtmdLoad(mmprojPath: String, llamaHandle: Long): Long

    external fun nativeMtmdFree(handle: Long)

    external fun nativeMtmdEncodeImage(mtmdHandle: Long, imageBytes: ByteArray): Long

    external fun nativeMtmdGetTokenCount(chunksHandle: Long): Int

    external fun nativeMtmdFreeChunks(chunksHandle: Long)
}
