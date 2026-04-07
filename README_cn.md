# 引入
对于所有科学工作者来说，求导无处不在。对于有确定解析式的函数，可以手动计算导函数。但多数时候，函数过于复杂或干脆是一个黑箱，以致手写导数变得困难。过去通常采用数值微分，在某个点附近重复采样，计算近似导数。
而当函数过于复杂或需要相当多次数的数值微分，比如梯度下降优化过程，数值微分会变得低效，并且直接与数值差分精度相关。人们不禁想问，是否能够找到一种技术，让电脑像人求导一样写出导函数的表达式，实现程序自动微分？
尽管对自动微分的研究很早就有，近年来深度学习的爆火才将这一工具引入了科学工作者的视野。现在已经有了很多自动微分技术，但它们大多依赖于Python，不能直接用于其他语言。
Enzyme是一个可对LLVM IR做自动微分的高性能框架，只要源代码能够翻译为LLVM IR，Enzyme框架就能对任何给定的函数求导。目前已在Julia科学计算生态中大放异彩，但对老代码的支持有限。本文将以老牌科学计算语言Fortran为例，讲解如何将优雅的自动微分引入你的HPC程序里。
# 准备
本段基于Linux系统编写，Windows和MacOS上做法不详。
## LLVM
我们需要确保Fortran源码能够被翻译成LLVM IR，因此我们需要Fortran的LLVM编译器`flang`。根据我的不完全调研，目前在网上基本上没有可开箱即用的预编译`flang`，就算能在某些包管理器中下载到`flang`，也可能缺少运行库。因此，我们需要从github上[下载](https://github.com/llvm/llvm-project)并[编译](https://llvm.org/docs/GettingStarted.html#getting-the-source-code-and-building-llvm)编译器。编译配置采用`cmake`，生成器建议使用`ninja`，变量`LLVM_ENABLE_PROJECTS` 里需要加入`flang` 字段，默认行为中只有C/C++语言的编译器`clang` 。
这里不详述具体的编译过程，整个LLVM包体很大，`clang`本身是必须要有，加上`flang`以及其他运行时支持共有7000多个文件需要编译。如果你的CPU单核性能不够强劲，整个过程将会非常漫长。
编译完成后，将flang加入环境变量，运行
```
clang --version
flang --version
```
若能输出正确的版本号则证明LLVM编译器安装成功。
## Enzyme
接下来我们需要编译Enzyme插件，从包管理器下载到的Enzyme很可能与本地的LLVM版本不符，这里还是建议从源码编译，在这个地方[下载](https://github.com/EnzymeAD/Enzyme)，安装过程在[这里](https://enzyme.mit.edu/Installation/)。
编译并安装后，会在安装位置出现`include`和`lib`文件夹，且`lib`文件夹中出现
```
LLVMEnzyme-<VERSION>.so
```
这是Enzyme自动微分的核心动态库。
# 实现流程
## 构成第一印象的简单例子
Enzyme在C里的接口很简单，这是Enzyme在C中最简单的代码实现：
example1.c
```
#include <stdio.h>
double square(double x)
{
	return x*x;
}
double __enzyme_autodiff(void*,double);
int main()
{
    double x=3.14;
    double grad_x=__enzyme_autodiff((void*)square,x);
    printf("square'(%f)=%f\n",x,grad_x);
    return 0;
}
```
Enzyme实现自动微分的逻辑是，在代码中声明一个神秘函数`__enzyme_autodiff`，第一个输入是待求导函数的指针，第二个输入是自变量的值，函数预期输出该值处的导数。在需要计算导数的地方，直接把`__enzyme_autodiff`当作函数调用。
**接下来就是最神秘的一步，编译到LLVM IR后，这个神秘函数名将会保留下来，随后LLVM IR优化器会将神秘函数替换成导函数，编译时自动链接Enzyme动态库实现对给定函数求导。**
在Fortran中，需要使用`ISO_C_BINDING`模块像正常调用外部C库函数那样调用这个神秘函数，需注意与C交互时的语法和原生Fortran略有差别。
example2.f95
```
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
执行
```
flang example2.f95 -S -emit-llvm
```
会生成`example2.ll`和两个模块文件，再执行
```
opt example2.ll --load-pass-plugin=/to/your/LLVMEnzyme-<VERSION>.so --passes enzyme -S -o example2_opt.ll
```
会生成`example2_opt.ll`
这时可以执行`llvm-diff`，查看上述两个文件有什么区别
```
llvm example2.ll example2_opt.ll
```
输出
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
可以看到原本调用`__enzyme_autodiff`的位置变成了`diffsquare`。
最后将`example2_opt.ll`编译到机器码即可正常运行
```
flang example2_opt.ll -o example2
./example2
```
输出
```
square'(3.140000) = 6.280000
```
# TODO LIST
对于含参数的多元函数，参数应视作常数，对于不同自变量还需要求偏导，Enzyme确有相关机制处理，但是比较复杂，留待以后再讨论。
