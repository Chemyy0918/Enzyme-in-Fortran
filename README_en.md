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
## Gradient of Arbitrary Multivariate Parametric Functions
Once multivariate parametric functions are involved, a pure C interface is no longer sufficient. It becomes necessary to write a C++ wrapper and expose a C-compatible interface so that Fortran can use automatic differentiation via standard C interoperability syntax.
Since Enzyme operates at the LLVM level, it is not possible to simply compile C++ code into a static or shared library (.a or .so) and link it during Fortran compilation. Instead, one possible approach is to compile each component separately into LLVM IR, then use llvm-link to merge them into a single mixed IR module. This combined IR can then be processed by opt to apply Enzyme’s automatic differentiation passes at the IR level. Finally, flang can be used to compile the optimized IR into an executable.
First, we briefly introduce the C++ interface syntax of Enzyme:
```c++
double* x, dx;
double y, dy;
double a;
int enzyme_dup, enzyme_out, enzyme_const;

dy = __enzyme_autodiff(
    (void*)func,
    enzyme_dup, x, dx,
    enzyme_out, y,
    enzyme_const, a
);
```
This syntax originates from Enzyme’s classification of all arguments (except the function pointer) into three categories:

1. Duplicated variables.These are input variables for which derivatives are computed. Two arrays are provided: x and dx. x specifies the primal input, while dx stores the gradients and must be initialized to zero. After autodiff, dx contains the derivative of the function evaluated at x.
2. Output variables .These are scalar outputs such as y. Their derivatives (dy) are returned as results of the autodiff call.
3. Inactive (constant) variables.These inputs, such as array a, are treated as constants and are not differentiated.
The argument types are specified using `enzyme_dup`, `enzyme_out`, and `enzyme_const` respectively.
A **significant** issue arises here: although Fortran can pass function pointers to C, such pointers are opaque to Enzyme. As a result, Enzyme cannot inspect the internal implementation of the function and therefore cannot perform automatic differentiation.
There are two possible solutions:
The first is to implement the function entirely inside a C++ wrapper. This makes the function fully visible to Enzyme. However, the original goal of this project is to add automatic differentiation capabilities to existing legacy Fortran code with minimal modification, rather than rewriting everything in C++. Moreover, if this were the approach, one might just as well use Julia directly, since Enzyme.jl already provides a much more convenient workflow.
The second approach is to bind the Fortran function using bind(C) and expose it under a C-visible symbol name. This symbol is then passed directly into `__enzyme_autodiff`. In the merged LLVM IR, Enzyme is able to see the full function definition and perform differentiation.
In this work, the second approach is preferred. A potential limitation is that when multiple functions in Fortran require differentiation, each `__enzyme_autodiff` call can only target a single function. As a result, multiple wrapper functions must be written, each corresponding to a different differentiated function.
One possible implementation of the C++ wrapper is shown below:
enzyme_wrap.cpp
```c++
extern "C"
{
    void __enzyme_autodiff(void*, ...);
    int enzyme_dup;
    int enzyme_const;

    double fn(double*, double*, int, int);

    void grad_fn(double* X, double* GRAD_X, double* P, int SIZE_X, int SIZE_P)
    {
        __enzyme_autodiff((void*)fn,
                          enzyme_dup, X, GRAD_X,
                          enzyme_const, P,
                          enzyme_const, SIZE_X,
                          enzyme_const, SIZE_P);
    }
}
```
Here, double fn(double*, double*, int, int) declares the function to be differentiated. Its implementation resides in Fortran. It has five inputs: the first is the variable array X, the second is the parameter array P, the third is the size of X (SIZE_X), and the fourth is the size of P (SIZE_P).
grad_fn is a wrapper around `__enzyme_autodiff`, taking five inputs. Compared to fn, it additionally includes GRAD_X, which stores the computed derivatives.
Considering the argument types, X and GRAD_X are marked as `enzyme_dup`, while all other arguments are marked as enzyme_const. `enzyme_out` is not used here because **Fortran cannot directly receive array return values from C functions; arrays can only be handled via subroutines (SUBROUTINE).**
A simple multivariate parametric Fortran example is:

$$f(x,y) = a x y + b / y$$

The gradient is:

$$
(\frac{\partial f}{\partial x},\frac{\partial f}{\partial y})=(ay,ax-\frac{b}{y^2})
$$

example3.f95
```fortran
MODULE FUNC_TEST
    USE ISO_C_BINDING
    IMPLICIT NONE
    INTERFACE
        SUBROUTINE GRAD(X,GRAD_X,P,SIZE_X,SIZE_P) BIND(C,NAME="grad_fn")
            USE ISO_C_BINDING
            IMPLICIT NONE
            INTEGER(C_INT),INTENT(IN),VALUE::SIZE_X,SIZE_P
            REAL(C_DOUBLE),INTENT(IN)::X(*),P(*)
            REAL(C_DOUBLE),INTENT(OUT)::GRAD_X(*)
        END SUBROUTINE
    END INTERFACE
    CONTAINS
        FUNCTION FUNC(X,P,SIZE_X,SIZE_P) BIND(C,NAME="fn") RESULT(Y)
            USE ISO_C_BINDING
            IMPLICIT NONE
            INTEGER(C_INT),INTENT(IN),VALUE::SIZE_X,SIZE_P
            REAL(C_DOUBLE),INTENT(IN)::X(*),P(*)
            REAL(C_DOUBLE)::Y
            Y = P(1)*X(1)*X(2) + P(2)/X(2)
        END FUNCTION
END MODULE FUNC_TEST

PROGRAM MAIN
    USE ISO_C_BINDING
    USE FUNC_TEST
    IMPLICIT NONE
    REAL(C_DOUBLE),ALLOCATABLE::X(:),P(:),GRAD_X(:)
    REAL(C_DOUBLE)::Y
    INTEGER(C_INT)::SIZE_X,SIZE_P

    ALLOCATE(X(2),GRAD_X(2),P(2))

    X = [2.,3.]
    P = [1.,1.]
    GRAD_X = 0.0D0

    SIZE_X = SIZE(X)
    SIZE_P = SIZE(P)

    Y = FUNC(X,P,SIZE_X,SIZE_P)
    CALL GRAD(X,GRAD_X,P,SIZE_X,SIZE_P)

    PRINT *,"f(",X(1),",",X(2),")=",Y
    PRINT *,"gradf(",X(1),",",X(2),")=(",GRAD_X(1),",",GRAD_X(2),")"

    DEALLOCATE(X,GRAD_X,P)
END PROGRAM MAIN
```
in this example, 

$$
a=1,b=1,x=2,y=3
$$

giving:

$$
f(x,y)=xy+\frac{1}{y}
$$

$$
(\frac{\partial f}{\partial x},\frac{\partial f}{\partial y})=(y,x-\frac{1}{y^2})
$$

The compilation and linking steps are as follows:
```bash
flang example3.f03 -emit-llvm -S
clang++ enzyme_wrap.cpp -emit-llvm -S
llvm-link example3.ll enzyme_wrap.ll -S -o mixed.ll
opt mixed.ll --load-pass-plugin=/to/your/enzyme.so --passes enzyme -S -o mixed_opt.ll
flang mixed_opt.ll -o a.out
```
Execution yields:
```
f( 2. , 3. )= 6.333333333333333
gradf( 2. , 3. )=( 3. , 1.8888888888888888 )
```
It can be seen that the gradient is correctly computed. By modifying X and P, gradients under different conditions can be obtained.
## TODO LIST
### Simple Application 1: Quasi-Newton Optimizer
### Simple Application 2: Symplectic Dynamics
