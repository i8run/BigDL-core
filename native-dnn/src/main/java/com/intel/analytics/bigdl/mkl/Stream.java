/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.mkl;

public class Stream {
    public static class Kind {
        public static final int Any = 0;
        public static final int Eager = 1;
        public static final int Lazy = 2;
    }

    public native static long Create(int streamKind);
    public native static void Submit(long stream, int length, long[] primitives);

    public native static long Wait(long loc, int block);
    public native static long Rerun(long stream);
    public native static void Destroy(long loc);
}
