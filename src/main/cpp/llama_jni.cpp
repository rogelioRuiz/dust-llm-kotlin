#include <jni.h>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <string>
#include <vector>

#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

namespace {

struct LlamaState {
    llama_model * model = nullptr;
    llama_context * context = nullptr;
};

std::once_flag g_backend_init;

void ensure_backend_init() {
    std::call_once(g_backend_init, []() {
        llama_backend_init();
    });
}

void fill_batch(llama_batch & batch, const llama_token * tokens, int32_t n_tokens, int32_t start_pos) {
    batch.n_tokens = 0;
    for (int32_t i = 0; i < n_tokens; ++i) {
        const int index = batch.n_tokens;
        batch.token[index] = tokens[i];
        batch.pos[index] = start_pos + i;
        batch.n_seq_id[index] = 1;
        batch.seq_id[index][0] = 0;
        batch.logits[index] = (i == n_tokens - 1) ? 1 : 0;
        batch.n_tokens += 1;
    }
}

llama_sampler * build_sampler(
    float temperature,
    int32_t top_k,
    float top_p,
    float min_p,
    float repeat_penalty,
    int32_t repeat_last_n,
    uint32_t seed
) {
    const llama_sampler_chain_params params = llama_sampler_chain_default_params();
    llama_sampler * chain = llama_sampler_chain_init(params);

    if (repeat_penalty != 1.0f && repeat_last_n > 0) {
        llama_sampler_chain_add(chain, llama_sampler_init_penalties(repeat_last_n, repeat_penalty, 0.0f, 0.0f));
    }

    if (temperature <= 0.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_greedy());
        return chain;
    }

    if (top_k > 0) {
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(top_k));
    }
    if (top_p < 1.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(top_p, 1));
    }
    if (min_p > 0.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_min_p(min_p, 1));
    }

    llama_sampler_chain_add(chain, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));
    return chain;
}

std::string serialize_tokens(const std::vector<llama_token> & tokens) {
    std::string result;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) {
            result.push_back(',');
        }
        result += std::to_string(tokens[i]);
    }
    return result;
}

jobjectArray make_string_array(JNIEnv * env, const std::string & first, const std::string & second, const std::string & third) {
    jclass string_class = env->FindClass("java/lang/String");
    if (string_class == nullptr) {
        return nullptr;
    }

    jobjectArray result = env->NewObjectArray(3, string_class, nullptr);
    if (result == nullptr) {
        return nullptr;
    }

    env->SetObjectArrayElement(result, 0, env->NewStringUTF(first.c_str()));
    env->SetObjectArrayElement(result, 1, env->NewStringUTF(second.c_str()));
    env->SetObjectArrayElement(result, 2, env->NewStringUTF(third.c_str()));
    return result;
}

bool read_prompt_tokens(JNIEnv * env, jintArray prompt_tokens, std::vector<llama_token> & prompt) {
    const jsize prompt_count = env->GetArrayLength(prompt_tokens);
    if (prompt_count <= 0) {
        return false;
    }

    jint * raw_prompt = env->GetIntArrayElements(prompt_tokens, nullptr);
    if (raw_prompt == nullptr) {
        return false;
    }

    prompt.resize(prompt_count);
    std::copy(raw_prompt, raw_prompt + prompt_count, prompt.begin());
    env->ReleaseIntArrayElements(prompt_tokens, raw_prompt, JNI_ABORT);
    return true;
}

jobjectArray run_generation(
    JNIEnv * env,
    LlamaState * state,
    const std::vector<llama_token> & prompt,
    mtmd_context * mtmd_ctx,
    mtmd_input_chunks * image_chunks,
    jint max_tokens,
    jfloat temperature,
    jint top_k,
    jfloat top_p,
    jfloat min_p,
    jfloat repeat_penalty,
    jint repeat_last_n,
    jint seed
) {
    const int32_t prompt_count = static_cast<int32_t>(prompt.size());
    const int32_t image_token_count = (image_chunks != nullptr)
        ? static_cast<int32_t>(mtmd_helper_get_n_tokens(image_chunks))
        : 0;
    const int32_t total_prompt_count = prompt_count + image_token_count;
    const int32_t context_size = static_cast<int32_t>(llama_n_ctx(state->context));
    if (total_prompt_count >= context_size) {
        return nullptr;
    }

    llama_memory_clear(llama_get_memory(state->context), true);

    llama_batch batch = llama_batch_init(std::max<int32_t>(prompt_count, 1), 0, 1);
    if (prompt_count > 0) {
        fill_batch(batch, prompt.data(), prompt_count, 0);

        if (llama_decode(state->context, batch) != 0) {
            llama_batch_free(batch);
            return nullptr;
        }
    }

    int32_t next_pos = prompt_count;
    if (image_chunks != nullptr && mtmd_ctx != nullptr) {
        llama_pos new_n_past = static_cast<llama_pos>(next_pos);
        const int32_t batch_size = static_cast<int32_t>(llama_n_batch(state->context));
        if (mtmd_helper_eval_chunks(mtmd_ctx, state->context, image_chunks,
                                    new_n_past, 0, batch_size, true, &new_n_past) != 0) {
            llama_batch_free(batch);
            return nullptr;
        }
        next_pos = static_cast<int32_t>(new_n_past);
    }

    llama_sampler * sampler = build_sampler(
        temperature,
        top_k,
        top_p,
        min_p,
        repeat_penalty,
        repeat_last_n,
        static_cast<uint32_t>(seed)
    );

    const llama_vocab * vocab = llama_model_get_vocab(state->model);
    std::vector<llama_token> generated;
    std::string stop_reason = "max_tokens";

    for (int i = 0; i < max_tokens; ++i) {
        const llama_token token = llama_sampler_sample(sampler, state->context, -1);
        generated.push_back(token);
        llama_sampler_accept(sampler, token);

        if (llama_vocab_is_eog(vocab, token)) {
            stop_reason = "eos";
            break;
        }

        fill_batch(batch, &token, 1, next_pos);
        next_pos += 1;

        if (llama_decode(state->context, batch) != 0) {
            llama_sampler_free(sampler);
            llama_batch_free(batch);
            return nullptr;
        }
    }

    llama_sampler_free(sampler);
    llama_batch_free(batch);

    return make_string_array(
        env,
        serialize_tokens(generated),
        std::to_string(generated.size()),
        stop_reason
    );
}

jstring run_generation_streaming(
    JNIEnv * env,
    LlamaState * state,
    const std::vector<llama_token> & prompt,
    mtmd_context * mtmd_ctx,
    mtmd_input_chunks * image_chunks,
    jint max_tokens,
    jfloat temperature,
    jint top_k,
    jfloat top_p,
    jfloat min_p,
    jfloat repeat_penalty,
    jint repeat_last_n,
    jint seed,
    jobject callback
) {
    if (callback == nullptr) {
        return nullptr;
    }

    const int32_t prompt_count = static_cast<int32_t>(prompt.size());
    const int32_t image_token_count = (image_chunks != nullptr)
        ? static_cast<int32_t>(mtmd_helper_get_n_tokens(image_chunks))
        : 0;
    const int32_t total_prompt_count = prompt_count + image_token_count;
    const int32_t context_size = static_cast<int32_t>(llama_n_ctx(state->context));
    if (total_prompt_count >= context_size) {
        return nullptr;
    }

    llama_memory_clear(llama_get_memory(state->context), true);

    llama_batch batch = llama_batch_init(std::max<int32_t>(prompt_count, 1), 0, 1);
    if (prompt_count > 0) {
        fill_batch(batch, prompt.data(), prompt_count, 0);

        if (llama_decode(state->context, batch) != 0) {
            llama_batch_free(batch);
            return nullptr;
        }
    }

    int32_t next_pos = prompt_count;
    if (image_chunks != nullptr && mtmd_ctx != nullptr) {
        llama_pos new_n_past = static_cast<llama_pos>(next_pos);
        const int32_t batch_size = static_cast<int32_t>(llama_n_batch(state->context));
        if (mtmd_helper_eval_chunks(mtmd_ctx, state->context, image_chunks,
                                    new_n_past, 0, batch_size, true, &new_n_past) != 0) {
            llama_batch_free(batch);
            return nullptr;
        }
        next_pos = static_cast<int32_t>(new_n_past);
    }

    llama_sampler * sampler = build_sampler(
        temperature,
        top_k,
        top_p,
        min_p,
        repeat_penalty,
        repeat_last_n,
        static_cast<uint32_t>(seed)
    );
    if (sampler == nullptr) {
        llama_batch_free(batch);
        return nullptr;
    }

    jclass callback_class = env->GetObjectClass(callback);
    if (callback_class == nullptr) {
        llama_sampler_free(sampler);
        llama_batch_free(batch);
        return nullptr;
    }

    jmethodID on_token = env->GetMethodID(callback_class, "onToken", "(I)Z");
    env->DeleteLocalRef(callback_class);
    if (on_token == nullptr) {
        if (env->ExceptionCheck()) {
            env->ExceptionClear();
        }
        llama_sampler_free(sampler);
        llama_batch_free(batch);
        return nullptr;
    }

    const llama_vocab * vocab = llama_model_get_vocab(state->model);
    std::string stop_reason = "max_tokens";

    for (int i = 0; i < max_tokens; ++i) {
        const llama_token token = llama_sampler_sample(sampler, state->context, -1);
        llama_sampler_accept(sampler, token);

        const jboolean should_continue = env->CallBooleanMethod(
            callback,
            on_token,
            static_cast<jint>(token)
        );
        if (env->ExceptionCheck()) {
            env->ExceptionClear();
            stop_reason = "cancelled";
            break;
        }
        if (should_continue == JNI_FALSE) {
            stop_reason = "cancelled";
            break;
        }

        if (llama_vocab_is_eog(vocab, token)) {
            stop_reason = "eos";
            break;
        }

        fill_batch(batch, &token, 1, next_pos);
        next_pos += 1;

        if (llama_decode(state->context, batch) != 0) {
            llama_sampler_free(sampler);
            llama_batch_free(batch);
            return nullptr;
        }
    }

    llama_sampler_free(sampler);
    llama_batch_free(batch);
    return env->NewStringUTF(stop_reason.c_str());
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_io_t6x_dust_llm_LlamaJNI_nativeLoad(
    JNIEnv * env,
    jclass,
    jstring path,
    jint n_gpu_layers,
    jint n_ctx,
    jint n_batch
) {
    if (path == nullptr) {
        return 0;
    }

    ensure_backend_init();

    const char * raw_path = env->GetStringUTFChars(path, nullptr);
    if (raw_path == nullptr) {
        return 0;
    }

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;

    llama_model * model = llama_model_load_from_file(raw_path, model_params);
    env->ReleaseStringUTFChars(path, raw_path);

    if (model == nullptr) {
        return 0;
    }

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = static_cast<uint32_t>(n_ctx);
    context_params.n_batch = static_cast<uint32_t>(n_batch);
    context_params.n_ubatch = static_cast<uint32_t>(n_batch);

    llama_context * context = llama_init_from_model(model, context_params);
    if (context == nullptr) {
        llama_model_free(model);
        return 0;
    }

    auto * state = new LlamaState();
    state->model = model;
    state->context = context;
    return reinterpret_cast<jlong>(state);
}

extern "C" JNIEXPORT void JNICALL
Java_io_t6x_dust_llm_LlamaJNI_nativeFree(
    JNIEnv *,
    jclass,
    jlong handle
) {
    if (handle == 0) {
        return;
    }

    auto * state = reinterpret_cast<LlamaState *>(handle);
    if (state->context != nullptr) {
        llama_free(state->context);
    }
    if (state->model != nullptr) {
        llama_model_free(state->model);
    }
    delete state;
}

extern "C" JNIEXPORT jstring JNICALL
Java_io_t6x_dust_llm_LlamaJNI_nativeGetMetadata(
    JNIEnv * env,
    jclass,
    jlong handle,
    jstring key
) {
    if (handle == 0 || key == nullptr) {
        return nullptr;
    }

    auto * state = reinterpret_cast<LlamaState *>(handle);
    if (state->model == nullptr) {
        return nullptr;
    }

    const char * raw_key = env->GetStringUTFChars(key, nullptr);
    if (raw_key == nullptr) {
        return nullptr;
    }

    char buffer[1024] = {0};
    const int32_t length = llama_model_meta_val_str(state->model, raw_key, buffer, sizeof(buffer));
    env->ReleaseStringUTFChars(key, raw_key);

    if (length < 0) {
        return nullptr;
    }

    return env->NewStringUTF(buffer);
}

extern "C" JNIEXPORT jintArray JNICALL
Java_io_t6x_dust_llm_LlamaJNI_nativeTokenize(
    JNIEnv * env,
    jclass,
    jlong handle,
    jstring text,
    jboolean add_special
) {
    if (handle == 0 || text == nullptr) {
        return nullptr;
    }

    auto * state = reinterpret_cast<LlamaState *>(handle);
    if (state->model == nullptr) {
        return nullptr;
    }

    const llama_vocab * vocab = llama_model_get_vocab(state->model);
    const char * raw = env->GetStringUTFChars(text, nullptr);
    if (raw == nullptr) {
        return nullptr;
    }

    const int32_t text_len = env->GetStringUTFLength(text);
    int32_t needed = llama_tokenize(vocab, raw, text_len, nullptr, 0, add_special, false);
    needed = needed < 0 ? -needed : needed;

    std::vector<llama_token> tokens(needed);
    const int32_t actual = llama_tokenize(vocab, raw, text_len, tokens.data(), needed, add_special, false);
    env->ReleaseStringUTFChars(text, raw);

    if (actual < 0) {
        return nullptr;
    }

    jintArray result = env->NewIntArray(actual);
    if (result == nullptr) {
        return nullptr;
    }

    env->SetIntArrayRegion(result, 0, actual, reinterpret_cast<const jint *>(tokens.data()));
    return result;
}

extern "C" JNIEXPORT jstring JNICALL
Java_io_t6x_dust_llm_LlamaJNI_nativeDetokenize(
    JNIEnv * env,
    jclass,
    jlong handle,
    jintArray tokens
) {
    if (handle == 0 || tokens == nullptr) {
        return nullptr;
    }

    auto * state = reinterpret_cast<LlamaState *>(handle);
    if (state->model == nullptr) {
        return nullptr;
    }

    const llama_vocab * vocab = llama_model_get_vocab(state->model);
    const jsize token_count = env->GetArrayLength(tokens);
    jint * data = env->GetIntArrayElements(tokens, nullptr);
    if (data == nullptr) {
        return nullptr;
    }

    int32_t buffer_size = std::max<int32_t>(token_count * 8, 64);
    std::vector<char> buffer(buffer_size, 0);
    int32_t length = llama_detokenize(
        vocab,
        reinterpret_cast<llama_token *>(data),
        token_count,
        buffer.data(),
        static_cast<int32_t>(buffer.size()),
        false,
        false
    );

    while (length < 0) {
        buffer_size = -length + 1;
        buffer.assign(buffer_size, 0);
        length = llama_detokenize(
            vocab,
            reinterpret_cast<llama_token *>(data),
            token_count,
            buffer.data(),
            static_cast<int32_t>(buffer.size()),
            false,
            false
        );
    }

    env->ReleaseIntArrayElements(tokens, data, JNI_ABORT);
    buffer[length] = '\0';
    return env->NewStringUTF(buffer.data());
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_io_t6x_dust_llm_LlamaJNI_nativeGenerate(
    JNIEnv * env,
    jclass,
    jlong handle,
    jintArray prompt_tokens,
    jint max_tokens,
    jfloat temperature,
    jint top_k,
    jfloat top_p,
    jfloat min_p,
    jfloat repeat_penalty,
    jint repeat_last_n,
    jint seed
) {
    if (handle == 0 || prompt_tokens == nullptr) {
        return nullptr;
    }

    auto * state = reinterpret_cast<LlamaState *>(handle);
    if (state->model == nullptr || state->context == nullptr) {
        return nullptr;
    }

    std::vector<llama_token> prompt;
    if (!read_prompt_tokens(env, prompt_tokens, prompt)) {
        return nullptr;
    }

    return run_generation(
        env,
        state,
        prompt,
        nullptr,
        nullptr,
        max_tokens,
        temperature,
        top_k,
        top_p,
        min_p,
        repeat_penalty,
        repeat_last_n,
        seed
    );
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_io_t6x_dust_llm_LlamaJNI_nativeGetEmbedding(
    JNIEnv * env,
    jclass,
    jlong handle,
    jintArray prompt_tokens
) {
    if (handle == 0 || prompt_tokens == nullptr) {
        return nullptr;
    }

    auto * state = reinterpret_cast<LlamaState *>(handle);
    if (state->model == nullptr || state->context == nullptr) {
        return nullptr;
    }

    llama_set_embeddings(state->context, true);

    std::vector<llama_token> prompt;
    if (!read_prompt_tokens(env, prompt_tokens, prompt)) {
        llama_set_embeddings(state->context, false);
        return nullptr;
    }

    llama_memory_clear(llama_get_memory(state->context), true);

    llama_batch batch = llama_batch_init(static_cast<int32_t>(prompt.size()), 0, 1);
    batch.n_tokens = 0;
    for (int32_t i = 0; i < static_cast<int32_t>(prompt.size()); ++i) {
        const int index = batch.n_tokens;
        batch.token[index] = prompt[i];
        batch.pos[index] = i;
        batch.n_seq_id[index] = 1;
        batch.seq_id[index][0] = 0;
        batch.logits[index] = 1;
        batch.n_tokens += 1;
    }

    if (llama_decode(state->context, batch) != 0) {
        llama_batch_free(batch);
        llama_set_embeddings(state->context, false);
        return nullptr;
    }
    llama_batch_free(batch);

    const int32_t n_embd = llama_model_n_embd(state->model);
    const float * embd = llama_get_embeddings_seq(state->context, 0);
    if (embd == nullptr) {
        embd = llama_get_embeddings_ith(state->context, -1);
    }
    if (embd == nullptr) {
        llama_set_embeddings(state->context, false);
        return nullptr;
    }

    std::vector<float> normalized(n_embd);
    float norm = 0.0f;
    for (int32_t i = 0; i < n_embd; ++i) {
        norm += embd[i] * embd[i];
    }
    norm = std::sqrt(norm);
    const float scale = norm > 1e-12f ? (1.0f / norm) : 0.0f;
    for (int32_t i = 0; i < n_embd; ++i) {
        normalized[i] = embd[i] * scale;
    }

    jfloatArray result = env->NewFloatArray(n_embd);
    if (result != nullptr) {
        env->SetFloatArrayRegion(result, 0, n_embd, normalized.data());
    }

    llama_set_embeddings(state->context, false);
    return result;
}

extern "C" JNIEXPORT jint JNICALL
Java_io_t6x_dust_llm_LlamaJNI_nativeGetEmbeddingDims(
    JNIEnv *,
    jclass,
    jlong handle
) {
    if (handle == 0) {
        return 0;
    }

    auto * state = reinterpret_cast<LlamaState *>(handle);
    if (state->model == nullptr) {
        return 0;
    }

    return static_cast<jint>(llama_model_n_embd(state->model));
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_io_t6x_dust_llm_LlamaJNI_nativeGenerateWithVision(
    JNIEnv * env,
    jclass,
    jlong handle,
    jintArray prompt_tokens,
    jlong mtmd_ctx_handle,
    jlong chunks_handle,
    jint max_tokens,
    jfloat temperature,
    jint top_k,
    jfloat top_p,
    jfloat min_p,
    jfloat repeat_penalty,
    jint repeat_last_n,
    jint seed
) {
    if (handle == 0 || prompt_tokens == nullptr || mtmd_ctx_handle == 0 || chunks_handle == 0) {
        return nullptr;
    }

    auto * state = reinterpret_cast<LlamaState *>(handle);
    if (state->model == nullptr || state->context == nullptr) {
        return nullptr;
    }

    std::vector<llama_token> prompt;
    if (!read_prompt_tokens(env, prompt_tokens, prompt)) {
        return nullptr;
    }

    auto * mtmd_ctx = reinterpret_cast<mtmd_context *>(mtmd_ctx_handle);
    auto * chunks = reinterpret_cast<mtmd_input_chunks *>(chunks_handle);
    return run_generation(
        env,
        state,
        prompt,
        mtmd_ctx,
        chunks,
        max_tokens,
        temperature,
        top_k,
        top_p,
        min_p,
        repeat_penalty,
        repeat_last_n,
        seed
    );
}

extern "C" JNIEXPORT jstring JNICALL
Java_io_t6x_dust_llm_LlamaJNI_nativeGenerateStreaming(
    JNIEnv * env,
    jclass,
    jlong handle,
    jintArray prompt_tokens,
    jint max_tokens,
    jfloat temperature,
    jint top_k,
    jfloat top_p,
    jfloat min_p,
    jfloat repeat_penalty,
    jint repeat_last_n,
    jint seed,
    jobject callback
) {
    if (handle == 0 || prompt_tokens == nullptr || callback == nullptr) {
        return nullptr;
    }

    auto * state = reinterpret_cast<LlamaState *>(handle);
    if (state->model == nullptr || state->context == nullptr) {
        return nullptr;
    }

    std::vector<llama_token> prompt;
    if (!read_prompt_tokens(env, prompt_tokens, prompt)) {
        return nullptr;
    }

    return run_generation_streaming(
        env,
        state,
        prompt,
        nullptr,
        nullptr,
        max_tokens,
        temperature,
        top_k,
        top_p,
        min_p,
        repeat_penalty,
        repeat_last_n,
        seed,
        callback
    );
}

extern "C" JNIEXPORT jstring JNICALL
Java_io_t6x_dust_llm_LlamaJNI_nativeGenerateStreamingWithVision(
    JNIEnv * env,
    jclass,
    jlong handle,
    jintArray prompt_tokens,
    jlong mtmd_ctx_handle,
    jlong chunks_handle,
    jint max_tokens,
    jfloat temperature,
    jint top_k,
    jfloat top_p,
    jfloat min_p,
    jfloat repeat_penalty,
    jint repeat_last_n,
    jint seed,
    jobject callback
) {
    if (handle == 0 || prompt_tokens == nullptr || mtmd_ctx_handle == 0 || chunks_handle == 0 || callback == nullptr) {
        return nullptr;
    }

    auto * state = reinterpret_cast<LlamaState *>(handle);
    if (state->model == nullptr || state->context == nullptr) {
        return nullptr;
    }

    std::vector<llama_token> prompt;
    if (!read_prompt_tokens(env, prompt_tokens, prompt)) {
        return nullptr;
    }

    auto * mtmd_ctx = reinterpret_cast<mtmd_context *>(mtmd_ctx_handle);
    auto * chunks = reinterpret_cast<mtmd_input_chunks *>(chunks_handle);
    return run_generation_streaming(
        env,
        state,
        prompt,
        mtmd_ctx,
        chunks,
        max_tokens,
        temperature,
        top_k,
        top_p,
        min_p,
        repeat_penalty,
        repeat_last_n,
        seed,
        callback
    );
}

extern "C" JNIEXPORT jlong JNICALL
Java_io_t6x_dust_llm_LlamaJNI_nativeMtmdLoad(
    JNIEnv * env,
    jclass,
    jstring mmproj_path,
    jlong llama_handle
) {
    if (mmproj_path == nullptr || llama_handle == 0) {
        return 0;
    }

    ensure_backend_init();

    auto * state = reinterpret_cast<LlamaState *>(llama_handle);
    if (state->model == nullptr) {
        return 0;
    }

    const char * raw_path = env->GetStringUTFChars(mmproj_path, nullptr);
    if (raw_path == nullptr) {
        return 0;
    }

    mtmd_context_params params = mtmd_context_params_default();
    params.use_gpu = true;

    mtmd_context * ctx = mtmd_init_from_file(raw_path, state->model, params);
    env->ReleaseStringUTFChars(mmproj_path, raw_path);
    return reinterpret_cast<jlong>(ctx);
}

extern "C" JNIEXPORT void JNICALL
Java_io_t6x_dust_llm_LlamaJNI_nativeMtmdFree(
    JNIEnv *,
    jclass,
    jlong handle
) {
    if (handle == 0) {
        return;
    }

    mtmd_free(reinterpret_cast<mtmd_context *>(handle));
}

extern "C" JNIEXPORT jlong JNICALL
Java_io_t6x_dust_llm_LlamaJNI_nativeMtmdEncodeImage(
    JNIEnv * env,
    jclass,
    jlong mtmd_handle,
    jbyteArray image_bytes
) {
    if (mtmd_handle == 0 || image_bytes == nullptr) {
        return 0;
    }

    auto * mtmd_ctx = reinterpret_cast<mtmd_context *>(mtmd_handle);

    const jsize byte_count = env->GetArrayLength(image_bytes);
    if (byte_count <= 0) {
        return 0;
    }

    std::vector<unsigned char> bytes(static_cast<size_t>(byte_count));
    env->GetByteArrayRegion(
        image_bytes,
        0,
        byte_count,
        reinterpret_cast<jbyte *>(bytes.data())
    );

    mtmd_bitmap * bitmap = mtmd_helper_bitmap_init_from_buf(
        mtmd_ctx, bytes.data(), static_cast<size_t>(byte_count));
    if (bitmap == nullptr) {
        return 0;
    }

    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    if (chunks == nullptr) {
        mtmd_bitmap_free(bitmap);
        return 0;
    }

    const char * marker = mtmd_default_marker();
    mtmd_input_text input_text;
    input_text.text = marker;
    input_text.add_special = true;
    input_text.parse_special = true;

    const mtmd_bitmap * bitmap_ptr = bitmap;
    const int32_t result = mtmd_tokenize(mtmd_ctx, chunks, &input_text, &bitmap_ptr, 1);
    mtmd_bitmap_free(bitmap);

    if (result != 0) {
        mtmd_input_chunks_free(chunks);
        return 0;
    }

    return reinterpret_cast<jlong>(chunks);
}

extern "C" JNIEXPORT jint JNICALL
Java_io_t6x_dust_llm_LlamaJNI_nativeMtmdGetTokenCount(
    JNIEnv *,
    jclass,
    jlong chunks_handle
) {
    if (chunks_handle == 0) {
        return 0;
    }

    auto * chunks = reinterpret_cast<mtmd_input_chunks *>(chunks_handle);
    return static_cast<jint>(mtmd_helper_get_n_tokens(chunks));
}

extern "C" JNIEXPORT void JNICALL
Java_io_t6x_dust_llm_LlamaJNI_nativeMtmdFreeChunks(
    JNIEnv *,
    jclass,
    jlong chunks_handle
) {
    if (chunks_handle == 0) {
        return;
    }

    mtmd_input_chunks_free(reinterpret_cast<mtmd_input_chunks *>(chunks_handle));
}
