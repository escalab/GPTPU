# Instructions to add edgetpu plugin to support GPTPU via providing shared libraries such as libdense.so

# 1. copy dense.cc, dense.h, Mymodel_utils.cc, Mymodel_utils.h to /edgetpu/src/cpp/example/

# 2. add the following three sections to /edgetpu/src/cpp/example/BUILD file:
(notice that each cc section requires one empty line between consecutive ones.)

## 2-1. for Mymodel_utils
```

cc_library(
     name = "Mymodel_utils",
     srcs = ["Mymodel_utils.cc"],
     hdrs = ["Mymodel_utils.h"],
     deps = [
         "@com_google_absl//absl/strings",
         "@libedgetpu//:header",
         "@org_tensorflow//tensorflow/lite:builtin_op_data",
         "@org_tensorflow//tensorflow/lite:framework",
         "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
     ],
 )

```

## 2-2. for libdense.so
```

cc_binary(
     name = "libdense.so",
     testonly = 1,
     srcs = ["dense.cc", "dense.h"],
         linkopts = ["-shared", "-fPIC", "-O3"],
         linkshared=1,
         copts = ["-I./src/cpp/example"],
     deps = [
         ":Mymodel_utils",
         "@libedgetpu//:header",
         "@org_tensorflow//tensorflow/lite:framework",
     ],
 )

```

## 2-3. for libdense_arm.so
```

cc_binary(
    name = "libdense_arm.so",
    testonly = 1,
    srcs = ["dense.cc", "dense.h"],
        linkopts = ["-shared", "-fPIC", "-O3"],
        linkshared=1,
        copts = ["-I./src/cpp/example"],
    deps = [
        ":model_utils",
        "@libedgetpu//:header",
        "@org_tensorflow//tensorflow/lite:framework",
    ],
)

```

# 3. modify edgetpu/Makefile

## 3-1. Add two more lines under ''examples:'' tag.
To build .so files
```
                                          //src/cpp/examples:libdense.so \
                                          //src/cpp/examples:libdense_arm.so \
```

## 3-2. Add the following section at the end of examples: tag section. 
1. Copy these two files to be shared in system (GPTPU/Makefile will do it later)
2. Make sure that ''CPU'' exists in this Makefile, check the beginning.

```
ifeq ("$(CPU)", "k8")
        sudo cp ./bazel-out/k8-opt/bin/src/cpp/examples/libdense.so ./../ #/usr/loc    al/lib
else
ifeq ("$(CPU)", "aarch64")
        sudo cp ./bazel-out/aarch64-opt/bin/src/cpp/examples/libdense_arm.so ./../     #/usr/local/lib
else
        @echo "WARNING: edgetpu Makefile: unexpected architecture."
endif
endif
        sudo cp ./src/cpp/examples/dense.h ./../ #/usr/local/include
``` 





