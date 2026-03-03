package io.t6x.dust.llm

object GpuInfo {
    val isGpuAvailable: Boolean by lazy {
        try { LlamaJNI.nativeIsGpuAvailable() } catch (_: UnsatisfiedLinkError) { false }
    }
}
