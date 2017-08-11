/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class com_intel_analytics_bigdl_quantization_Quantization */

#ifndef _Included_com_intel_analytics_bigdl_quantization_Quantization
#define _Included_com_intel_analytics_bigdl_quantization_Quantization
#ifdef __cplusplus
extern "C" {
#endif
#undef com_intel_analytics_bigdl_quantization_Quantization_NCHW
#define com_intel_analytics_bigdl_quantization_Quantization_NCHW 0L
#undef com_intel_analytics_bigdl_quantization_Quantization_NHWC
#define com_intel_analytics_bigdl_quantization_Quantization_NHWC 1L
/*
 * Class:     com_intel_analytics_bigdl_quantization_Quantization
 * Method:    printHello
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_quantization_Quantization_printHello
  (JNIEnv *, jclass);

/*
 * Class:     com_intel_analytics_bigdl_quantization_Quantization
 * Method:    FixConvKernelDescInit
 * Signature: (IIII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_quantization_Quantization_FixConvKernelDescInit
  (JNIEnv *, jclass, jint, jint, jint, jint);

/*
 * Class:     com_intel_analytics_bigdl_quantization_Quantization
 * Method:    FixConvKernelInit
 * Signature: (J[FIIIIIFI)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_quantization_Quantization_FixConvKernelInit
  (JNIEnv *, jclass, jlong, jfloatArray, jint, jint, jint, jint, jint, jfloat, jint);

/*
 * Class:     com_intel_analytics_bigdl_quantization_Quantization
 * Method:    FixConvKernelLoadFromModel
 * Signature: (J[BI[F[FIIIIFI)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_quantization_Quantization_FixConvKernelLoadFromModel
  (JNIEnv *, jclass, jlong, jbyteArray, jint, jfloatArray, jfloatArray, jint, jint, jint, jint, jfloat, jint);

/*
 * Class:     com_intel_analytics_bigdl_quantization_Quantization
 * Method:    FixConvDataDescInit
 * Signature: (IIIIIIIIIIII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_quantization_Quantization_FixConvDataDescInit
  (JNIEnv *, jclass, jint, jint, jint, jint, jint, jint, jint, jint, jint, jint, jint, jint);

/*
 * Class:     com_intel_analytics_bigdl_quantization_Quantization
 * Method:    FixConvDataInit
 * Signature: (J[FIIIIIIIIIIIIIFI)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_quantization_Quantization_FixConvDataInit
  (JNIEnv *, jclass, jlong, jfloatArray, jint, jint, jint, jint, jint, jint, jint, jint, jint, jint, jint, jint, jint, jfloat, jint);

/*
 * Class:     com_intel_analytics_bigdl_quantization_Quantization
 * Method:    FixConvKernelSumDescInit
 * Signature: (I)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_quantization_Quantization_FixConvKernelSumDescInit
  (JNIEnv *, jclass, jint);

/*
 * Class:     com_intel_analytics_bigdl_quantization_Quantization
 * Method:    FixConvKernelSumInit
 * Signature: (J[FIIIII)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_quantization_Quantization_FixConvKernelSumInit
  (JNIEnv *, jclass, jlong, jfloatArray, jint, jint, jint, jint, jint);

/*
 * Class:     com_intel_analytics_bigdl_quantization_Quantization
 * Method:    InternalMixPrecisionConvolutionGEMM
 * Signature: (IJJ[FI[FI[FIIIIIF)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_quantization_Quantization_InternalMixPrecisionConvolutionGEMM
  (JNIEnv *, jclass, jint, jlong, jlong, jfloatArray, jint, jfloatArray, jint, jfloatArray, jint, jint, jint, jint, jint, jfloat);

/*
 * Class:     com_intel_analytics_bigdl_quantization_Quantization
 * Method:    FreeMemory
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_quantization_Quantization_FreeMemory
  (JNIEnv *, jclass, jlong);

/*
 * Class:     com_intel_analytics_bigdl_quantization_Quantization
 * Method:    FixFCKernelDescInit
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_quantization_Quantization_FixFCKernelDescInit
  (JNIEnv *, jclass, jint, jint);

/*
 * Class:     com_intel_analytics_bigdl_quantization_Quantization
 * Method:    FixFCKernelLoadFromModel
 * Signature: (J[B[F[FIIFI)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_quantization_Quantization_FixFCKernelLoadFromModel
  (JNIEnv *, jclass, jlong, jbyteArray, jfloatArray, jfloatArray, jint, jint, jfloat, jint);

/*
 * Class:     com_intel_analytics_bigdl_quantization_Quantization
 * Method:    FixFCDataDescInit
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_quantization_Quantization_FixFCDataDescInit
  (JNIEnv *, jclass, jint, jint);

/*
 * Class:     com_intel_analytics_bigdl_quantization_Quantization
 * Method:    FixFCDataInit
 * Signature: (J[FIIIFI)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_quantization_Quantization_FixFCDataInit
  (JNIEnv *, jclass, jlong, jfloatArray, jint, jint, jint, jfloat, jint);

#ifdef __cplusplus
}
#endif
#endif
