#include <omp.h>
#include <mkl.h>
#include "com_intel_analytics_bigdl_mkl_MklDnn.h"

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT
  void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_setNumThreads
(JNIEnv *env, jclass cls, jint num)
{
  omp_set_num_threads(num);
  mkl_set_dynamic(0);
  mkl_set_num_threads(num);
  // mkl_disable_fast_mm();
}

#ifdef __cplusplus
}
#endif
