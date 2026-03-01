package io.t6x.dust.llm

data class LLMConfig(
    val nGpuLayers: Int = 0,
    val contextSize: Int = 2048,
    val batchSize: Int = 512,
    val mmprojPath: String? = null,
)
