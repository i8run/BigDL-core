#define _POSIX_C_SOURCE 200112L
#include "utils.h"
#include "com_intel_analytics_bigdl_mkl_Memory.h"
#include <stdlib.h>
#include <string.h>
#include <mkl.h>
#include <omp.h>

#ifdef __cplusplus
extern "C" {
#endif


void fast_copy(float* dst, float *src, const size_t n)
{
  int threshold = omp_get_max_threads();
  const int run_parallel = (n >= threshold) && (omp_in_parallel() == 0);

  if (run_parallel) {
    const int block_mem_size = 256 * 1024;
    const int block_size = block_mem_size / sizeof(float);
    #pragma omp parallel for
    for (size_t i = 0; i < n; i += block_size)
      memcpy(dst + i, src + i,
              (i + block_size > n) ? (n - i) * sizeof(float) : block_mem_size);

    return;
  }

  memcpy(dst, src, sizeof(float) * n);
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_Memory
 * Method:    SetDataHandle
 * Signature: (JJI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_mkl_Memory_SetDataHandle
  (JNIEnv *env, jclass cls, jlong primitive, jlong data, jint offset)
  {
    float *j_data = (float*)data;
    CHECK(
      mkldnn_memory_set_data_handle(
        (mkldnn_primitive_t)primitive,
        j_data + offset));

    return (long)j_data;
  }

/*
 * Class:     com_intel_analytics_bigdl_mkl_Memory
 * Method:    Zero
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_mkl_Memory_Zero
  (JNIEnv *env, jclass cls, jlong data, jint length, jint element_size)
  {
    memset((float*)data, 0, length * element_size);
    return 0;
  }

/*
 * Class:     com_intel_analytics_bigdl_mkl_Memory
 * Method:    CopyPtr2Ptr
 * Signature: (JIJIII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_mkl_Memory_CopyPtr2Ptr
  (JNIEnv *env, jclass cls, jlong src, jint srcOffset,
   jlong dst, jint dstOffset, jint length, jint element_size)
  {
    fast_copy((float*)dst + dstOffset, (float*)src + srcOffset, length);
    return 0;
  }

/*
 * Class:     com_intel_analytics_bigdl_mkl_Memory
 * Method:    CopyArray2Ptr
 * Signature: ([FIJIII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_mkl_Memory_CopyArray2Ptr
  (JNIEnv *env, jclass cls, jfloatArray src, jint srcOffset,
   jlong dst, jint dstOffset, jint length, jint element_size)
  {
    float *j_src = (*env)->GetPrimitiveArrayCritical(env, src, JNI_FALSE);
    float *j_dst = (float*)dst;
    fast_copy(j_dst + dstOffset, j_src + srcOffset, length);
    (*env)->ReleasePrimitiveArrayCritical(env, src, j_src, 0);
    return 0;
  }

/*
 * Class:     com_intel_analytics_bigdl_mkl_Memory
 * Method:    CopyPtr2Array
 * Signature: (JI[FIII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_mkl_Memory_CopyPtr2Array
  (JNIEnv *env, jclass cls, jlong src, jint srcOffset,
   jfloatArray dst, jint dstOffset, jint length, jint element_size)
  {
    float *j_dst = (*env)->GetPrimitiveArrayCritical(env, dst, JNI_FALSE);
    // float *j_src = (float *)src;
    // int i = 0;
    // for (i = 0; i < 10; i++) {
    //   printf("%f\n", j_src[i + srcOffset]);
    // }
    // fflush(stdout);
    fast_copy(j_dst + dstOffset, (float *)src + srcOffset, length);
    (*env)->ReleasePrimitiveArrayCritical(env, dst, j_dst, 0);
    return 0;
  }

/*
 * Class:     com_intel_analytics_bigdl_mkl_Memory
 * Method:    AlignedMalloc
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_mkl_Memory_AlignedMalloc
  (JNIEnv *env, jclass cls, jint capacity, jint align)
  {
    void *p;
    // int ret = posix_memalign(&p, align, capacity);
    p = mkl_malloc(capacity, align);
    if (p != NULL) {
      return (long)p;
    } else {
      return (long)0;
    }
  }

/*
 * Class:     com_intel_analytics_bigdl_mkl_Memory
 * Method:    AlignedFree
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_Memory_AlignedFree
  (JNIEnv *env, jclass cls, jlong ptr)
  {
    // free((void*)ptr);
    mkl_free((void*)ptr);
  }

/*
 * Class:     com_intel_analytics_bigdl_mkl_Memory
 * Method:    SAdd
 * Signature: (IJIIIJI)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_Memory_SAdd
  (JNIEnv *env, jclass cls, jint n, jlong aPtr, jint aOffset,
   jlong bPtr, jint bOffset, jlong yPtr, jint yOffset)
  {
    vsAdd( n, (float*)aPtr + aOffset, (float*)bPtr + bOffset,
           (float*)yPtr + yOffset);
  }

/*
 * Class:     com_intel_analytics_bigdl_mkl_Memory
 * Method:    scale
 * Signature: (IFJJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_Memory_scale
  (JNIEnv *env, jclass cls, jint num, jfloat scale, jlong x, jlong y)
  {
    cblas_scopy(num, (float *)x, 1, (float *)y, 1);
    cblas_sscal(num, scale, (float *)y, 1);
  }

/*
 * Class:     com_intel_analytics_bigdl_mkl_Memory
 * Method:    axpby
 * Signature: (IFJFJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_Memory_axpby
  (JNIEnv *env, jclass cls, jint n, jfloat alpha, jlong x, jfloat beta, jlong y)
  {
    cblas_saxpby(n, alpha, (float *)x, 1, beta, (float *)y, 1);
  }

/*
 * Class:     com_intel_analytics_bigdl_mkl_Memory
 * Method:    set
 * Signature: (JFII)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_Memory_set
  (JNIEnv *env, jclass cls, jlong ptr, jfloat value, jint length, jint elementSize)
  {
    jfloat *data = (jfloat*)ptr;
    for (int i = 0; i < length; i++) {
      data[i] = value;
    }
  }

#ifdef __cplusplus
}
#endif
