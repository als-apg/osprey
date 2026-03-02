## Appendix A Useful Mathematical Formulae

### Vector Algebra

Electric and magnetic fields are vectors which are defined by direction and magnitude in space \(\mathbf{E}(x,y,z)\) and \(\mathbf{B}(x,y,z)\), where we use a Cartesian coordinate system \((x,y,z)\).The distribution of such vectors is called a vector field in contrast to a scalar field such as the distribution of temperature \(T(x,y,z)\). In component form such vectors can be written as

\[\mathbf{E}=E_{x}\,\mathbf{x}+E_{y}\,\mathbf{y}+E_{z}\,\mathbf{z}\,.\] (A.1) \[\mathbf{E}+\mathbf{B}=\left(E_{x}+B_{x}\right)\,\mathbf{x}+\left(E_{y}+B_{y} \right)\,\mathbf{y}+\left(E_{z}+B_{z}\right)\,\mathbf{z}\] (A.2) \[\mathbf{E}\,\mathbf{B}=E_{x}B_{x}+E_{y}B_{y}+E_{z}B_{z}=|E|\,\,|B|\,\cos\theta\] (A.3)

where \(\theta\) is the angle between the vectors, and the

\[\mathbf{E}\times\mathbf{B}=\left(E_{y}B_{z}-E_{z}B_{y},E_{z}B_{x}-E_{x}B_{z},E_{x}B_{y }-E_{y}B_{x}\right)\,,\] (A.4)

\[|\mathbf{E}\times\mathbf{B}|=|E|\,\,|B|\,\sin\theta.\]

The resulting vector is orthogonal to both vectors \(\mathbf{E}\) and \(\mathbf{B}\) and the vectors \([\mathbf{E},\mathbf{B},\)\(\mathbf{E}\times\mathbf{B}]\) form a right handed orthogonal coordinate system.

#### Differential Vector Expressions

To describe the variation of scalar and vector fields a gradient for scalars if defined by

\[\nabla T=\operatorname{grad}T=\left(\frac{\partial T}{\partial x},\frac{\partial T }{\partial y},\frac{\partial T}{\partial z}\right), \tag{100}\]

which is a vector.

For vectors we define two differential expressions. The first is the divergence of the vector field:

\[\nabla\mathbf{E}=\operatorname{div}\mathbf{E}=\frac{\partial E_{x}}{\partial x}+\frac{ \partial E_{y}}{\partial y}+\frac{\partial E_{z}}{\partial z}\,, \tag{101}\]

which is a scalar. Geometrically, the divergence of a vector is the outward flux of this vector per unit volume. As an example consider a small cube with dimensions \(\operatorname{d}\!x\),\(\operatorname{d}\!y\).\(\operatorname{d}\!z\). Put this cube in a uniform vector field and you get zero divergence, because the flux into the cube is equal to the flux out. Now, put the cube into a field free area and place a positive charge into the cube. The flux of fields is all outwards and the divergence is nonzero.

The divergence can be evaluated by integrating over all volume and we get with Gauss's integral theorem (101)

\[\int_{V}\nabla\mathbf{E}\operatorname{d}\!V=\oint\mathbf{E}\mathbf{n}\operatorname{d}\!a\,, \tag{102}\]

where \(\mathbf{n}\) is a unit vector normal to the surface and \(\operatorname{d}\!a\) a surface element. The volume integral becomes an integral over the outer surface.

The second differential expression is the "curl" of a vector:

\[\nabla\times\mathbf{B}=\left(\frac{\partial B_{z}}{\partial y}-\frac{\partial B_{ y}}{\partial z},\frac{\partial B_{x}}{\partial z}-\frac{\partial B_{z}}{ \partial x},\frac{\partial B_{y}}{\partial x}-\frac{\partial B_{x}}{\partial y }\right). \tag{103}\]

The "curl" of a vector per unit area is the circulation about the direction of the vector.

#### Algebraic Relations

\[\mathbf{a}\left(\mathbf{b}\times\mathbf{c}\right) =\mathbf{b}\left(\mathbf{c}\times\mathbf{a}\right)=\mathbf{c}\left(\mathbf{a}\times \mathbf{b}\right) \tag{104}\] \[\mathbf{a}\times\left(\mathbf{b}\times\mathbf{c}\right) =\mathbf{b}\left(\mathbf{ac}\right)-\mathbf{c}\left(\mathbf{ab}\right)\] (105) \[\left(\mathbf{a}\times\mathbf{b}\right)\left(\mathbf{c}\times\mathbf{d}\right) =\left(\mathbf{ac}\right)\left(\mathbf{bd}\right)-\left(\mathbf{bc}\right) \left(\mathbf{ad}\right) \tag{106}\]\[\mathbf{a}\times\left(\mathbf{b}\times\mathbf{c}\right)+\mathbf{b}\times\left(\mathbf{c} \times\mathbf{a}\right)+\mathbf{c}\times\left(\mathbf{a}\times\mathbf{b}\right) =0\] (A.12) \[\left(\mathbf{a}\times\mathbf{b}\right)\times\left(\mathbf{c}\times\mathbf{d}\right) =\mathbf{c}\,\left[\left(\mathbf{a}\times\mathbf{b}\right)\mathbf{d}\right]-\mathbf{d }\,\left[\left(\mathbf{a}\times\mathbf{b}\right)\mathbf{c}\right]\] (A.13)

#### Differential Relations

\[\nabla\left(\mathbf{a}\varphi\right) =\varphi\nabla\mathbf{a}+\mathbf{a}\nabla\varphi\] (A.14) \[\nabla\times\left(\mathbf{a}\varphi\right) =\varphi\left(\nabla\times\mathbf{a}\right)-\mathbf{a}\times\nabla\varphi\] (A.15) \[\nabla\left(\mathbf{a}\times\mathbf{b}\right) =\mathbf{b}\,\left(\nabla\times\mathbf{a}\right)-\mathbf{a}\,\left(\nabla \times\mathbf{b}\right)\] (A.16) \[\nabla\times\left(\mathbf{a}\times\mathbf{b}\right) =\left(\mathbf{b}\nabla\right)\mathbf{a}-\left(\mathbf{a}\nabla\right)\mathbf{b}+ \mathbf{a}\left(\nabla\mathbf{b}\right)-\mathbf{b}\left(\nabla\mathbf{a}\right)\] (A.17) \[\nabla\left(\mathbf{ab}\right) =\left(\mathbf{b}\nabla\right)\mathbf{a}+\left(\mathbf{a}\nabla\right)\mathbf{b} +\mathbf{a}\times\left(\nabla\times\mathbf{b}\right)+\mathbf{b}\times\left(\nabla\times \mathbf{a}\right)\] (A.18) \[\nabla\times\left(\nabla\varphi\right) =0\] (A.19) \[\nabla\left(\nabla\times\mathbf{a}\right) =0\] (A.20) \[\nabla\times\left(\nabla\times\mathbf{a}\right) =\nabla\left(\nabla\mathbf{a}\right)-\Delta\mathbf{a}\] (A.21)

#### Partial Integration

Partial integration is defined by

\[\int_{a}^{b}uv^{\prime}dx =\left.uv\right|_{a}^{b}-\int_{a}^{b}vu^{\prime}dx,\quad\text{or}\] (A.22a) \[\int_{a}^{b}udv =\left.uv\right|_{a}^{b}-\int_{a}^{b}vdu\] (A.22b)

#### Trigonometric and Exponential Functions

\[\begin{array}{c}\mathrm{e}^{\mathrm{i}x}=\cos x+\mathrm{i}\sin x\\ \cos x=\frac{1}{2}\left(\mathrm{e}^{\mathrm{i}x}+\mathrm{e}^{-\mathrm{i}x} \right)\qquad\qquad\qquad\qquad\sin x=\frac{1}{\mathrm{i}2}\left(\mathrm{e}^{ \mathrm{i}x}-\mathrm{e}^{-\mathrm{i}x}\right)\\ \cos\left(a\pm b\right)=\cos a\cos b\mp\sin a\sin b\ \ \sin\left(a\pm b\right)=\sin a\cos b\pm\sin b\cos a \\ \tan\left(a\pm b\right)=\frac{\tan a\mp\tan b}{1\mp\tan a\tan b}\qquad\qquad \qquad\cot\left(a\pm b\right)=\frac{\cot a\cot b\mp 1}{\cot a\mp\cot b}\\ \frac{\mathrm{d}}{\mathrm{d}a}\tan a=\frac{1}{\cos^{2}a}=1+\tan^{2}a\qquad \qquad\qquad\frac{\mathrm{d}}{\mathrm{d}a}\arctan a=\frac{1}{1+a^{2}}\end{array}\] (A.23)

#### Integral Relations

\[\int\limits_{V}\nabla\varphi\;\mathrm{d}r =\oint\limits_{S}\varphi\hat{\boldsymbol{u}}\;\mathrm{d}\sigma\] (A.24) \[\int\limits_{V}\nabla\boldsymbol{a}\;\mathrm{d}r =\oint\limits_{S}a\hat{\boldsymbol{u}}\;\mathrm{d}\sigma\] Gauss' theorem (A.25) \[\int\limits_{S}\left(\nabla\boldsymbol{x}\boldsymbol{a}\right) \hat{\boldsymbol{u}}\;\mathrm{d}\sigma =\oint\limits_{S}a\;\mathrm{d}\boldsymbol{s}\] Stokes' theorem (A.26)

#### Dirac's Delta Function

\[\delta\left(x\right)=\left\{\begin{array}{l}\infty\;\mathrm{for}\;x=0\\ 0\;\mathrm{for}\;x\neq 0\end{array}\right.\qquad\qquad\int\limits_{-\infty}^{ \infty}\delta\left(x\right)\mathrm{d}x=1\] \[\left|\alpha\right|\int\limits_{-\infty}^{\infty}\delta\left( \alpha x\right)\mathrm{d}x=\int\limits_{-\infty}^{\infty}\delta\left(y\right) \mathrm{d}y=1\qquad\delta\left(\omega\right)=\int\limits_{-\infty}^{\infty} \delta\left(t\right)\mathrm{e}^{-\mathrm{i}2\pi\omega t}\mathrm{d}t=1\] \[\delta\left(x\right)=\frac{1}{2\pi}\sum\limits_{n=-\infty}^{ \infty}\mathrm{e}^{\mathrm{i}nx}\qquad\quad\frac{1}{2\pi}\sum\limits_{n=- \infty}^{\infty}\mathrm{e}^{\mathrm{i}nx}=\sum\limits_{m=-\infty}^{\infty} \delta\left(x-2\pi m\right)\] (A.27)

#### Bessel's Functions

Order \(n\) and first kind:

\[J_{n}\left(x\right)=\sum\limits_{p=0}^{\infty}\frac{\left(-1\right)^{p}\left( x/2\right)^{n+2p}}{p!\left(n+p\right)!}\] (A.28)

\[\begin{array}{l}n=0\qquad J_{0}\left(0\right)=1\\ n=1,2,3\ldots\;J_{n}\left(0\right)=0\end{array}\] (A.29)

derivatives and recursion formulas with \(J_{n}=J_{n}\left(x\right)\):

\[J_{n+1} =\frac{2n}{x}J_{n}-J_{n-1}\] (A.30) \[J_{n}^{\prime} =J_{n-1}-\frac{n}{x}J_{n}=\frac{n}{x}J_{n}-J_{n+1}=\frac{1}{2} \left(J_{n-1}-J_{n+1}\right)\] (A.31)first four roots of Bessel's functions of the first kind: \(J_{n}\left(x_{i}\right)=0\)

\[\begin{array}{ccccc}n&J_{n}\left(x_{1}\right)&J_{n}\left(x_{2}\right)&J_{n} \left(x_{3}\right)&J_{n}\left(x_{4}\right)\\ 0&2.4048&5.5200&8.6537&11.7954\\ 1&3.8317&7.0155&10.1743&\\ 2&5.1356&8.4172&11.6198&\\ 3&6.3801&9.7610&&\\ \end{array}\] (A.32)

#### Series Expansions

For \(\delta\ll 1\)

\[\mathrm{e}^{x}=!+x+\frac{1}{2!}x^{2}+\frac{1}{3!}x^{3}+\ldots\] (A.33) \[\ln\left(1-x\right) =-x-\frac{1}{2}x^{2}-\frac{1}{3}x^{3}-\ldots\ \ \mathrm{for}\ -1\leq x<1\] (A.34) \[\sin x =x-\frac{1}{3!}x^{3}+\frac{1}{5!}x^{5}-\ldots\] (A.35) \[\cos x =1-\frac{1}{2!}x^{2}+\frac{1}{4!}x^{4}-\ldots\] (A.36) \[\sqrt{1+\delta} =1+\frac{1}{2}\delta-\frac{1}{2^{3}}\delta^{2}+\frac{1}{2^{5}} \delta^{3}\ldots\] (A.37) \[\frac{1}{1+\delta} =1-\delta+\delta^{2}-\delta^{3}+\ldots.\] (A.38)

#### Fourier Series

A function \(f\left(t\right)\) is periodic if \(f\left(t\right)=f\left(t+T\right)=f\left(t+nT\right)\) where \(n\) is an integer and \(T\) the lowest value for which this statement is true. Such a function can be expressed with \(\tau=\frac{t}{T}\) by

\[f\left(\tau\right)=\frac{1}{2}a_{0}+\sum_{n=1}^{\infty}\left[a_{n}\cos\left(2 \pi n\tau+\vartheta_{n}\right)+b_{n}\sin\left(2\pi n\tau+\vartheta_{n}\right)\right]\] (A.39)or using exponentials

\[f\left(\tau\right)=\sum_{n=-\infty}^{\infty}c_{n}\mathrm{e}^{\mathrm{i}n\tau}\] (A.40)

where \(c_{n}=c_{-n}\) are complex with \(c_{n}=\overline{c_{-n}}\) and \(c_{0}=\left\langle f\left(\tau\right)\right\rangle.\) The coefficients are

\[c_{n}=\int_{0}^{1}f\left(\tau\right)\mathrm{e}^{-\mathrm{i}n\tau}\mathrm{d} \tau\quad\text{ where }\quad\tau=t/T\] (A.41)

Parseval's Theorem \[\int_{-\infty}^{\infty}F^{2}(t)\,\mathrm{d}t=\frac{1}{2\pi}\int_{-\infty}^{ \infty}F^{2}(\omega)\,\mathrm{d}\omega\;,\] (A.42)

where \(F(t)=\frac{1}{2\pi}\int F(\omega)\,\mathrm{e}^{-\mathrm{i}\omega t}\,\mathrm{ d}\omega\) and \(F(\omega)=\int F(t)\,\mathrm{e}^{\mathrm{i}\omega t}\,\mathrm{d}t\).

Fourier Transform

For non-periodic functions \(f\left(t\right)\) with \(\left(T\rightarrow\infty\right)\) the Fourier transform is

\[F(\omega)=\int_{-\infty}^{\infty}f(t)\,e^{-\mathrm{i}\omega t}\mathrm{d}t\] (A.43)

and

\[f(t)=\frac{1}{2\pi}\int_{-\infty}^{\infty}F(\omega)\,\mathrm{e}^{-\mathrm{i} \omega t}\mathrm{d}\omega\] (A.44)

#### Coordinate Transformations

Cartesian coordinates \[\mathrm{d}s^{2} =\mathrm{d}x^{2}+\mathrm{d}y^{2}+\mathrm{d}z^{2}\] \[\mathrm{d}V =\mathrm{d}x\,\mathrm{d}y\,\mathrm{d}z\] \[\nabla\psi =\left(\frac{\partial\psi}{\partial x},\frac{\partial\psi}{ \partial y},\frac{\partial\psi}{\partial z}\right)\] (A.45) \[\nabla\mathbf{a} =\frac{\partial a_{x}}{\partial x}+\frac{\partial a_{y}}{\partial y }+\frac{\partial a_{z}}{\partial z}\]

### Vector Algebra

\[\nabla\times\mathbf{a} = \left(\frac{\partial a_{z}}{\partial y}-\frac{\partial a_{y}}{ \partial z},\,\frac{\partial a_{x}}{\partial z}-\frac{\partial a_{z}}{\partial x },\,\frac{\partial a_{y}}{\partial x}-\frac{\partial a_{x}}{\partial y}\right)\] \[\Delta\psi = \frac{\partial^{2}\psi}{\partial x^{2}}+\frac{\partial^{2}\psi}{ \partial y^{2}}+\frac{\partial^{2}\psi}{\partial z^{2}}\]

### General Coordinate Transformation

Transformation into **new coordinates**\(\left(u,v,w\right),\) where \(x=x(u,v,w),\)

\(y=y(u,v,w)\) and \(z=z(u,v,w)\)

\[\mathrm{d}s^{2} = \frac{\mathrm{d}u^{2}}{U^{2}}+\frac{\mathrm{d}v^{2}}{V^{2}}+\frac {\mathrm{d}w^{2}}{W^{2}}\] \[\mathrm{d}V = \frac{\mathrm{d}u}{U}\frac{\mathrm{d}v}{V}\frac{\mathrm{d}w}{W}\] \[\nabla\psi = \left(U\frac{\partial\psi}{\partial u},V\frac{\partial\psi}{ \partial v},W\frac{\partial\psi}{\partial w}\right)\] (A.46) \[\nabla\mathbf{a} = UVW\left[\frac{\partial}{\partial u}\frac{a_{u}}{VW}+\frac{ \partial}{\partial v}\frac{a_{v}}{UW}+\frac{\partial}{\partial w}\frac{a_{w}}{ UV}\right]\] \[\nabla\times\mathbf{a} = \left\{\begin{array}{c}VW\left[\frac{\partial}{\partial v}\frac {a_{w}}{W}-\frac{\partial}{\partial w}\frac{a_{v}}{V}\right],\\ UV\left[\frac{\partial}{\partial w}\frac{a_{v}}{V}-\frac{\partial}{\partial v }\frac{a_{u}}{U}\right]\end{array}\right\}\] \[\Delta\psi = UVW\left[\frac{\partial}{\partial u}\left(\frac{U}{VW}\frac{ \partial\psi}{\partial u}\right)+\frac{\partial}{\partial v}\left(\frac{V}{UW }\frac{\partial\psi}{\partial v}\right)+\frac{\partial}{\partial w}\left( \frac{W}{UV}\frac{\partial\psi}{\partial w}\right)\right]\]

where

\[U^{-1} = \sqrt{\left(\frac{\partial x}{\partial u}\right)^{2}+\left( \frac{\partial y}{\partial u}\right)^{2}+\left(\frac{\partial z}{\partial u} \right)^{2}},\] \[V^{-1} = \sqrt{\left(\frac{\partial x}{\partial v}\right)^{2}+\left( \frac{\partial y}{\partial v}\right)^{2}+\left(\frac{\partial z}{\partial v} \right)^{2}},\] (A.47) \[W^{-1} = \sqrt{\left(\frac{\partial x}{\partial w}\right)^{2}+\left( \frac{\partial y}{\partial w}\right)^{2}+\left(\frac{\partial z}{\partial w} \right)^{2}},\]


### Vector Algebra

#### Curvilinear Coordinates

Transformation to **curvilinear coordinates** of beam dynamics

\[\mathrm{d}s^{2} = \mathrm{d}x^{2}+\mathrm{d}y^{2}+(1+\kappa_{x}x+\kappa_{y}y)^{2}\, \mathrm{d}z^{2}=\mathrm{d}x^{2}+\mathrm{d}y^{2}+h^{2}\mathrm{d}z^{2}\] \[\mathrm{d}V = \mathrm{d}x\,\mathrm{d}y\,h\,\mathrm{d}z\] \[\nabla\psi = \frac{\partial\psi}{\partial x}\mathbf{x}+\,\frac{\partial\psi}{ \partial y}\mathbf{y}+\,\frac{1}{h}\frac{\partial\psi}{\partial z}z,\] \[\nabla\mathbf{a} = \frac{1}{h}\left[\,\frac{\partial\left(ha_{x}\right)}{\partial x }+\,\frac{\partial\left(h\,a_{y}\right)}{\partial y}+\,\frac{\partial a_{z}}{ \partial z}\,\right],\] (A.50) \[\nabla\times\mathbf{a} = \frac{1}{h}\left[\,\frac{\partial\left(h\,a_{z}\right)}{\partial y }-\,\frac{\partial a_{y}}{\partial z}\right]\mathbf{x}+\frac{1}{h}\left[\,\frac{ \partial a_{x}}{\partial z}-\frac{\partial\left(h\,a_{z}\right)}{\partial x} \right]\mathbf{y}+\left[\,\frac{\partial a_{y}}{\partial x}-\frac{\partial a_{x}}{ \partial y}\right]\mathbf{z}\] \[\Delta\psi = \frac{1}{h}\left[\,\frac{\partial}{\partial x}\left(h\frac{ \partial\psi}{\partial x}\right)+\,\frac{\partial}{\partial y}\left(h\frac{ \partial\psi}{\partial y}\right)+\,\frac{\partial}{\partial z}\left(\frac{1}{ h}\,\frac{\partial\psi}{\partial z}\right)\right]\]

