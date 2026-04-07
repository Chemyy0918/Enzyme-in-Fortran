Translated by DeepSeek
# Introduction
For all scientific workers, differentiation is ubiquitous. For functions with a definite analytical expression, the derivative can be computed manually. However, most of the time the function is too complex or is a black box, making it difficult to write the derivative by hand. In the past, numerical differentiation was commonly used to obtain approximate derivatives by repeatedly sampling around a point.
When the function is very complex or when many numerical derivatives are required – for example, in gradient descent optimization – numerical differentiation becomes inefficient, and its accuracy is directly tied to the numerical difference precision. One cannot help but ask: is there a technique that allows the computer to write down the derivative expression like a human does, thus achieving automatic differentiation of programs?
Although research on automatic differentiation (AD) started early, the recent explosion of deep learning has finally brought this tool into the view of scientific workers. Many AD techniques exist today, but most of them rely on Python and cannot be used directly for other languages.
Enzyme is a high‑performance framework that performs automatic differentiation on LLVM IR. As long as the source code can be translated to LLVM IR, the Enzyme framework can differentiate any given function. It has already shown great success in the Julia scientific computing ecosystem, but support for legacy code is limited. This article takes the classic scientific computing language Fortran as an example to explain how to introduce elegant automatic differentiation into your HPC programs.
# Preparation

This section is written based on the Linux system. The procedures for Windows and macOS are unknown.
## LLVM
We need to ensure that Fortran source code can be translated to LLVM IR, so we require the LLVM‑based Fortran compiler `flang`. Based on my incomplete research, there is basically no pre‑built, ready‑to‑use `flang` available online. Even if you can download `flang` from some package managers, it may lack runtime libraries. Therefore, we need to [download](https://github.com/llvm/llvm-project) and [build](https://llvm.org/docs/GettingStarted.html#getting-the-source-code-and-building-llvm) the compiler from GitHub. The build configuration uses `cmake`; it is recommended to use `ninja` as the generator. The `LLVM_ENABLE_PROJECTS` variable must include the `flang` field, because the default behaviour only includes the C/C++ compiler `clang`.
The specific build process is not described in detail here. The entire LLVM package is very large. `clang` itself is mandatory, and together with `flang` and other runtime support, over 7000 files need to be compiled. If your CPU's single‑core performance is not strong, the whole process will be very long.
After compilation, add flang to your environment variables and run:
```bash
clang --version  
flang --version
```
If the correct version numbers are printed, the LLVM compiler has been installed successfully.
## Enzyme
Next we need to build the Enzyme plugin. The Enzyme obtained from a package manager may very likely be incompatible with your local LLVM version. Therefore, it is recommended to build from source. [Download](https://github.com/EnzymeAD/Enzyme) it from this location. The installation instructions can be found [here](https://enzyme.mit.edu/Installation/).
After building and installing, you will see `include` and `lib` directories in the installation location. In the `lib` directory you should see a file like
```
LLVMEnzyme-<VERSION>.so
```
This is the core dynamic library for Enzyme automatic differentiation.
# Implementation Workflow
## A Simple Example to Form a First Impression
The interface of Enzyme in C is very simple. Here is the simplest C code implementing Enzyme:

example1.c
```c
#include <stdio.h>
double square(double x)
{
    return x * x;
}
double __enzyme_autodiff(void*,double);
int main()
{
    double x = 3.14;
    double grad_x = __enzyme_autodiff((void*)square, x);
    printf("square'(%f)=%f\n", x, grad_x);
    return 0;
}
```
Enzyme's logic for performing automatic differentiation is: declare a mysterious function `__enzyme_autodiff` in the code. Its first argument is a pointer to the function to be differentiated, the second argument is the value of the independent variable, and the function is expected to return the derivative at that point. Wherever the derivative is needed, `__enzyme_autodiff` is called directly.
**The next step is the most mysterious: after compilation to LLVM IR, this mysterious function name will be preserved. Then the LLVM IR optimizer replaces the mysterious function with the derivative function, and at compile time the Enzyme dynamic library is automatically linked to differentiate the given function.**
In Fortran, we need to use the `ISO_C_BINDING` module to call this mysterious function just like any external C library function, noting that the syntax for interacting with C is slightly different from native Fortran.

example2.f95
```fortran
MODULE FUNC_TEST
    CONTAINS
    FUNCTION SQUARE(X) BIND(C) RESULT(Y)
        USE,INTRINSIC::ISO_C_BINDING
        IMPLICIT NONE
        REAL(C_DOUBLE),INTENT(IN),VALUE::X
        REAL(C_DOUBLE)::Y
        Y=X**2
    END FUNCTION
END MODULE
MODULE ENZYME
    INTERFACE
        FUNCTION GRAD(FUNC_PTR,X) BIND(C, NAME="__enzyme_autodiff") RESULT(GRAD_X)
            USE,INTRINSIC::ISO_C_BINDING
            IMPLICIT NONE
            TYPE(C_FUNPTR),INTENT(IN),VALUE::FUNC_PTR
            REAL(C_DOUBLE),INTENT(IN),VALUE::X
            REAL(C_DOUBLE)::GRAD_X
        END FUNCTION
    END INTERFACE
END MODULE
PROGRAM MAIN
    USE,INTRINSIC::ISO_C_BINDING
    USE FUNC_TEST
    USE ENZYME
    IMPLICIT NONE
    REAL(C_DOUBLE)::X=3.14D0,GRAD_X
    GRAD_X=GRAD(C_FUNLOC(SQUARE),X)
    PRINT *,"square'(",X,")=",GRAD_X
END PROGRAM
```
Execute:
```bash
flang example2.f95 -S -emit-llvm
```
This will generate `example2.ll` and two module files. Then execute:
```bash
opt example2.ll --load-pass-plugin=/to/your/LLVMEnzyme-<VERSION>.so --passes enzyme -S -o example2_opt.ll
```
This generates `example2_opt.ll`.
Now you can run `llvm-diff` to see the difference between the two files:
```bash
llvm-diff example2.ll example2_opt.ll
```
Output:
```
function @diffesquare exists only in right module
in function _QQmain:
  in block %0 / %0:
    >   %8 = call fast { double } @diffesquare(double %7, double 1.000000e+00)
    >   %9 = extractvalue { double } %8, 0
    >   store double %9, ptr %2, align 8
    <   %8 = call contract double @__enzyme_autodiff(ptr %6, double %7)
    <   store double %8, ptr %2, align 8
```
You can see that the original call to `__enzyme_autodiff` has been replaced by `diffesquare`.
Finally, compile `example2_opt.ll` to machine code and run it:
```bash
flang example2_opt.ll -o example2
./example2
```
Output:
```
square'( 3.14 )= 6.28
```
# TODO LIST
For multivariate functions with parameters, the parameters should be treated as constants, and partial derivatives with respect to different independent variables are needed. Enzyme does have relevant mechanisms, but they are more complex and will be discussed in the future.
