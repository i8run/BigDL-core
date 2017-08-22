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
 * Method:    ConvKernelDescInit
 * Signature: (IIII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_quantization_Quantization_ConvKernelDescInit
  (JNIEnv *, jclass, jint, jint, jint, jint);

/*
 * Class:     com_intel_analytics_bigdl_quantization_Quantization
 * Method:    ConvKernelInit
 * Signature: (J[FIIIIIFI)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_quantization_Quantization_ConvKernelInit
  (JNIEnv *, jclass, jlong, jfloatArray, jint, jint, jint, jint, jint, jfloat, jint);

/*
 * Class:     com_intel_analytics_bigdl_quantization_Quantization
 * Method:    ConvKernelLoadFromModel
 * Signature: (J[BI[F[FIIIIFI)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_quantization_Quantization_ConvKernelLoadFromModel
  (JNIEnv *, jclass, jlong, jbyteArray, jint, jfloatArray, jfloatArray, jint, jint, jint, jint, jfloat, jint);

/*
 * Class:     com_intel_analytics_bigdl_quantization_Quantization
 * Method:    ConvDataDescInit
 * Signature: (IIIIIIIIIIII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_quantization_Quantization_ConvDataDescInit
  (JNIEnv *, jclass, jint, jint, jint, jint, jint, jint, jint, jint, jint, jint, jint, jint);

/*
 * Class:     com_intel_analytics_bigdl_quantization_Quantization
 * Method:    ConvDataInit
 * Signature: (J[FIIIIIIIIIIIIIFI)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_quantization_Quantization_ConvDataInit
  (JNIEnv *, jclass, jlong, jfloatArray, jint, jint, jint, jint, jint, jint, jint, jint, jint, jint, jint, jint, jint, jfloat, jint);

/*
 * Class:     com_intel_analytics_bigdl_quantization_Quantization
 * Method:    ConvKernelSumDescInit
 * Signature: (I)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_quantization_Quantization_ConvKernelSumDescInit
  (JNIEnv *, jclass, jint);

/*
 * Class:     com_intel_analytics_bigdl_quantization_Quantization
 * Method:    ConvKernelSumInit
 * Signature: (J[FIIIII)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_quantization_Quantization_ConvKernelSumInit
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
 * Method:    FCKernelDescInit
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_quantization_Quantization_FCKernelDescInit
  (JNIEnv *, jclass, jint, jint);

/*
 * Class:     com_intel_analytics_bigdl_quantization_Quantization
 * Method:    FCKernelLoadFromModel
 * Signature: (J[B[F[FIIFI)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_quantization_Quantization_FCKernelLoadFromModel
  (JNIEnv *, jclass, jlong, jbyteArray, jfloatArray, jfloatArray, jint, jint, jfloat, jint);

/*
 * Class:     com_intel_analytics_bigdl_quantization_Quantization
 * Method:    FCDataDescInit
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_quantization_Quantization_FCDataDescInit
  (JNIEnv *, jclass, jint, jint);

/*
 * Class:     com_intel_analytics_bigdl_quantization_Quantization
 * Method:    FCDataInit
 * Signature: (J[FIIIFI)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_quantization_Quantization_FCDataInit
  (JNIEnv *, jclass, jlong, jfloatArray, jint, jint, jint, jfloat, jint);

#ifdef __cplusplus
}
#endif
#endif
