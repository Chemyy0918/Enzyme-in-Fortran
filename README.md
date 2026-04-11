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
```bash
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
```c
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
执行
```bash
flang example2.f95 -S -emit-llvm
```
会生成`example2.ll`和两个模块文件，再执行
```bash
opt example2.ll --load-pass-plugin=/to/your/LLVMEnzyme-<VERSION>.so --passes enzyme -S -o example2_opt.ll
```
会生成`example2_opt.ll`
这时可以执行`llvm-diff`，查看上述两个文件有什么区别
```bash
llvm-diff example2.ll example2_opt.ll
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
```bash
flang example2_opt.ll -o example2
./example2
```
输出
```
square'( 3.14 )= 6.28
```
## 任意多元含参函数的梯度
一旦涉及到多元含参函数，C接口不再可行，必须写C++封装，并对外暴露C接口，让Fortran按调C库的语法使用自动微分。
由于Enzyme是在LLVM层面上的自动微分，不能把C++封成.a或.so然后编Fortran的时候链接它，可以考虑把它们各自编译到LLVM IR后采用`llvm-link`，构成一段混合代码，再过`opt`对LLVM IR函数自动微分，最后用`flang`编译，得到可执行文件。
首先简要介绍一下Enzyme的C++接口语法
```c++
double* x,dx;
double y,dy;
double a;
int enzyme_dup,enzyme_out,enzyme_const;
dy=__enzyme_autodiff(
	(void*)func,
	enzyme_dup,x,dx,
	enzyme_out,y
	enzyme_const,a
	);
```
这样的语法源自Enzyme把除了函数指针以外的参数分为三类，他们分别是：
1. Duplicated 复印变量 输入两个数组x和dx，x指明求导的位置，dx是存放梯度的数组(需要初始化为全0)，在运行autodiff之后，dx被赋值为在x点处的梯度值。
2. Output 输出变量 输入一个数值y，它的导数dy以返回值的形式给出。
3. Inactive 常量 输入一个数组a，它们作为参数存在，不需要求导。
依次使用`enzyme_dup`,`enzyme_out`和`enzyme_const`指明各个变量的类型。
接着，有一个**相当严峻**的问题，Fortran虽然可以向C传函数指针，但这种指针是不透明的，Enzyme看不到这个函数内部具体做了什么，也就不能实现自动微分。有两种解决方案：
其一是在C++封装内写函数，这样对Enzyme来说是透明的，但是我们这个项目的初衷是赋予老Fortran项目自动微分的新功能，理应对Fortran代码做足够小的改动，而不是在C++里重写。而且，为什么我不写Julia程序呢，Enzyme.jl可是非常好用的；
其二是在Fortran函数bind(C)的时候赋一个对C暴露的函数名，然后直接把这个函数名的指针写进autodiff，在混合的LLVM IR里，Enzyme能看到这个函数的具体实现。
我这边考虑采用第二种。这个过程可能包含的缺陷是，当Fortran函数里有多个函数需要自动微分，一个__enzyme_autodiff只能管一个函数，于是便需要写很多个__enzyme_autodiff，作为不同函数的导函数。
封装过的cpp代码的一种实现是：
enzyme_wrap.cpp
```c
extern "C" 
{
    void __enzyme_autodiff(void*,...);
    int enzyme_dup;
    int enzyme_const;
    double fn(double*,double*,int,int);
    void grad_fn(double* X,double* GRAD_X,double* P,int SIZE_X,int SIZE_P)
    {
        __enzyme_autodiff((void*)fn,
                          enzyme_dup,X,GRAD_X,
                          enzyme_const,P,
                          enzyme_const,SIZE_X,
                          enzyme_const,SIZE_P);
    }
}
```
在这里`double fn(double*,double*,int,int)`是待微分函数的声明，具体实现在Fortran里，有五个输入，第一个输入是自变量数组`X`，第二个输入是常量构成的数组`P`，第三个是自变量数组的尺寸`SIZE_X`，第四个是常量数组的尺寸`SIZE_P`。
`grad_fn`是一个包裹`__enzyme_autodiff`的函数，有五个输入，相比`fn`来说多了一个数组`GRAD_X`用于储存导数。
考虑到各个参量的类型，`X`和`GRAD_X`需要指定`enzyme_dup`，其他都是`enzyme_const`。在这里不使用`enzyme_out`的考量是，**Fortran不能直接通过C接口接收一个数组类型的函数返回值，只能通过子程序`SUBROUTINE`取得数组。**
一个简单的多元含参Fortran例子是
$$
f(x,y)=axy+\frac{b}{y}
$$
梯度是
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
            Y=P(1)*X(1)*X(2)+P(2)/X(2)
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
    X=[2.,3.]
    P=[1.,1.]
    GRAD_X=0.0D0
    SIZE_X=SIZE(X)
    SIZE_P=SIZE(P)
    Y=FUNC(X,P,SIZE_X,SIZE_P)
    CALL GRAD(X,GRAD_X,P,SIZE_X,SIZE_P)
    PRINT *,"f(",X(1),",",X(2),")=",Y
    PRINT *,"gradf(",X(1),",",X(2),")=(",GRAD_X(1),",",GRAD_X(2),")"
    DEALLOCATE(X,GRAD_X,P)
END PROGRAM MAIN
```
此例中，
$$
a=1,b=1,x=2,y=3
$$
那么
$$
f(x,y)=xy+\frac{1}{y}
$$
$$
(\frac{\partial f}{\partial x},\frac{\partial f}{\partial y})=(y,x-\frac{1}{y^2})
$$
按如下步骤链接与编译
```bash
flang example3.f03 -emit-llvm -S
clang++ enzyme_wrap.cpp -emit-llvm -S
llvm-link example3.ll enzyme_wrap.ll -S -o mixed.ll
opt mixed.ll --load-pass-plugin=/to/your/enzyme.so --passes enzyme -S -o mixed_opt.ll
flang mixed_opt.ll -o a.out
```
执行可得
``` 
f( 2. , 3. )= 6.333333333333333
gradf( 2. , 3. )=( 3. , 1.8888888888888888 )
```
可以看到正确地输出了梯度，修改`X`和`P`，可以得到各种情况下的梯度信息。

## TODO LIST
### 简单应用1：拟牛顿法优化器
### 简单应用2：辛动力学
