## Part III

**Beam Dynamics**

## Chapter 7 Single Particle Dynamics

The general equations of motion, characterized by an abundance of perturbation terms on the right-hand side of, for example, (6.95), (6.96) have been derived in the previous chapter. If these perturbation terms were allowed to become significant in real beam transport systems, we would face almost insurmountable mathematical problems trying to describe the motion of charged particles in a general way. For practical mathematical reasons it is therefore important to design components for particle beam transport systems such that undesired terms appear only as small perturbations. With a careful design of beam guidance magnets and accurate alignment of these magnets we can indeed achieve this goal.

Most of the perturbation terms are valid solutions of the Laplace equation describing higher order fields components. Virtually all these terms can be minimized to the level of perturbations by proper design of beam transport magnets. Specifically, we will see that the basic goals of beam dynamics can be achieved by using only two types of magnets, bending magnets and quadrupoles which sometimes are combined into one magnet. Beam transport systems, based on only these two lowest order magnet types, are called linear systems and the resulting theory of particle dynamics in the presence of only such magnets is referred to as linear beam dynamics or linear beam optics.

In addition to the higher order magnetic field components, we also find purely kinematic terms in the equations of motion due to large amplitudes or due to the use of curvilinear coordinates. Some of these terms are generally very small for particle trajectories which stay close to the reference path such that divergences are small, \(x^{\prime}\ll 1\) and \(y^{\prime}\ll 1\). The lowest order kinematic effects resulting from the use of a curvilinear coordinate system, however, cannot generally be considered small perturbations. One important consequence of this choice for the coordinate system is that the natural bending magnet is a sector magnet which has very different beam dynamics properties than a rectangular magnet which would be the natural magnettype for a Cartesian coordinate system. While a specific choice of a coordinate system will not change the physics, we must expect that some features are expressed easier or more complicated in one or the other coordinate system. We have chosen to use the curvilinear system because it follows the ideal path of the particle beam and offers a simple direct way to express particle trajectories deviating from the ideal path. In a fixed Cartesian coordinate system we would have to deal with geometric expressions relating the points along the ideal path to an arbitrary reference point. The difference becomes evident for a simple trajectory like a circle of radius \(r\) and center at \((x_{0},y_{0})\) which in a fixed orthogonal coordinate system would be expressed by \((x-x_{0})^{2}+(y-y_{0})^{2}=r^{2}\). In the curvilinear coordinate system this equation reduces to the simple identity \(x(z)=0\).

### Linear Beam Transport Systems

The theory of beam dynamics based on quadrupole magnets for focusing is called strong focusing beam dynamics in contrast to the case of weak focusing, which utilizes the focusing of sector magnets in combination with a small gradient in the bending magnet profile. Such focusing is employed in circular accelerators like betatrons or some cyclotrons and the first generation of synchrotrons. The invention of strong focusing by Christofilos [1] and independently by Courant et al. [2] changed quickly the way focusing arrangements for large particle accelerators are determined. One of the main attraction for this kind of focusing was the ability to greatly reduce the magnet aperture needed for the particle beam since the stronger focusing confines the particles to a much smaller cross section compared to weak focusing. A wealth of publications and notes have been written during the fifties to determine and understand the intricacies of strong focusing, especially the rising problems of alignment and field tolerances as well as those of resonances. Particle stability conditions from a mathematical point of view have been investigated by Moser [3].

In this chapter, we will discuss the theory of linear charged particle beam dynamics and apply it to the development of beam transport systems, the characterization of particle beams, and to the derivation of beam stability criteria. The bending and focusing function may be performed either in separate magnets or be combined within a synchrotron magnet. The arrangement of magnets in a beam transport system, called the magnet lattice, is often referred to as a separated function or combined function lattice depending on whether the lattice makes use of separate dipole and quadrupole magnets or uses combined function magnets, respectively.

Linear equations of motion can be extracted from (6.95), (6.96) to treat beam dynamics in first or linear approximation. For simplicity and without restricting generality we assume the bending of the beam to occur only in one plane, the \(x\)-plane. The linear magnetic fields for bending and quadrupole magnets are expressed by

\[B_{x} = -g\,y, \tag{7.1a}\] \[B_{y} = B_{y0}+gx\,, \tag{7.1b}\]

where \(B_{y0}\) is the dipole field and \(g\) the gradient of the quadrupole field. With these field components we obtain from (6.95), (6.96) the equations of motion in the approximation of linear beam dynamics

\[x^{\prime\prime}+\left(k_{0}+\kappa_{0x}^{2}\right)x = 0\,, \tag{7.2a}\] \[y^{\prime\prime}-k_{0}y = 0\,. \tag{7.2b}\]

Both, the focusing from the bending magnet and that from a quadrupole may be combined into one parameter

\[K(z)=k_{0}(z)+\kappa_{0x}^{2}(z). \tag{7.3}\]

So far no distinction has been made between combined or separated function magnets and the formulation of the equations of motion based on the magnet strength parameter \(K\) as defined in (7.3), is valid for both types of magnets. For separated function magnets either \(k_{0}\) or \(\kappa_{0x}\) is set to zero while for combined function magnets both parameters are nonzero.

#### Nomenclature

Focusing along a beam transport line is performed by discrete quadrupoles placed to meet specific particle beam characteristics required at the end or some intermediate point of the beam line. The dependence of the magnet strength on \(z\) is, therefore, a priori indeterminate and is the subject of lattice design in accelerator physics. To describe focusing lattices simple symbols are used to point out location and sometimes relative strength of magnets. In this text we will use symbols from Fig. 7.1 for bending magnets, quadrupoles, and sextupoles or multipoles.

All magnets are symbolized by squares along the \(z\)-axis and the length of the symbol may represent the actual magnetic length. The symbol for pure dipole magnets is a square centered about the \(z\)-axis while bending magnets with a gradient are shifted vertically to indicate the sign of the focusing. Positive squares are used to indicate horizontal focusing and negative squares for horizontal defocusing quadrupoles.

Using such symbols, a typical beam transport line may have general patterns like that shown in Fig 7.1. The sequence of magnets and their strength seems random and is mostly determined by external conditions to be discussed later. More regular magnet arrangements occur for circular accelerators or very long beam transport lines composed of periodic sections.

### Matrix Formalism in Linear Beam Dynamics

The seemingly arbitrary distribution of focusing parameters in a beam transport system makes it impossible to formulate a general solution of the differential equations of motion (7.2). To describe particle trajectories analytically through a beam transport line composed of drift spaces, bending magnets, and quadrupoles, we will derive mathematical tools which consist of partial solutions and can be used to describe complete particle trajectories.

In this section we will derive and discuss the matrix formalism [4] as a method to describe particle trajectories. This method makes use of the fact that the magnet strength parameters are constant at least within each individual magnet. The equations of motion become very simple since the restoring force \(K\) is constant and the solutions have the form of trigonometric functions. The particle trajectories may now be described by analytical functions at least within each uniform element of a transport line including magnet free drift spaces.

These solutions can be applied to any arbitrary beam transport line, where the focusing parameter \(K\) changes in a step like function as shown in Fig. 7.1. Cutting this beam line into its smaller elements so that \(K=\) const. in each of these pieces

Figure 7.1: Symbols for magnets in lattice design and typical distributions of magnets along a beam transport line

we will be able to follow the particle trajectories analytically step by step through the whole transport system. This is the model generally used in particle beam optics and is called the hard edge model.

In reality, however, since nature does not allow sudden changes of physical quantities (natura non facit saltus) the hard edge model is only an approximation, although for practical purposes a rather good one. In a real magnet the field strength does not change suddenly from zero to full value but rather follows a smooth transition from zero to the maximum field. Sometimes, the effects due to this smooth field transition or fringe field are important and we will derive the appropriate corrections later in this section. For now, we continue using the hard edge model for beam transport magnets and keep in mind that in some cases a correction may be needed to take into account the effects of a smooth field transition at the magnet edges.

Using this approximation, where \(1/\rho_{0}\) and \(k\) are constants, and ignoring perturbations, the equation of motion is reduced to that of a harmonic oscillator,

\[u^{\prime\prime}+K_{u}u=0,\quad\text{where}\qquad K_{u}=k_{u0}+\kappa_{0u}^{2} =\text{const}\,. \tag{7.4}\]

The principal solutions have been derived in Sect. 5.5.1 and are expressed in matrix formulation by

\[\left(\begin{array}{c}u(z)\\ u^{\prime}(z)\end{array}\right)=\left(\begin{array}{cc}C_{u}(z)&S_{u}(z)\\ C_{u}^{\prime}(z)&S_{u}^{\prime}(z)\end{array}\right)\left(\begin{array}{c}u _{0}\\ u_{0}^{\prime}\end{array}\right)\,, \tag{7.5}\]

where \(u\) may be used for either \(x\) or \(y\). We have deliberately separated the motion in both planes since we do not consider coupling. Formally, we could combine the two \(2\times 2\) transformation matrices for each plane into one \(4\times 4\) matrix describing the transformation of all four coordinates

\[\left(\begin{array}{c}x(z)\\ x^{\prime}(z)\\ y(z)\\ y^{\prime}(z)\end{array}\right)=\left(\begin{array}{ccc}C_{x}(z)&S_{x}(z)&0 &0\\ C_{x}^{\prime}(z)&S_{x}^{\prime}(z)&0&0\\ 0&0&C_{y}(z)&S_{y}(z)\\ 0&0&C_{y}^{\prime}(z)&S_{y}^{\prime}(z)\end{array}\right)\left(\begin{array}[] {c}x_{0}\\ x_{0}^{\prime}\\ y_{0}\\ y_{0}^{\prime}\end{array}\right). \tag{7.6}\]

Obviously the transformations are still completely decoupled but in this form we could include coupling effects, where, for example, the \(x\)-motion depends also on the \(y\)-motion and vice versa. This can be further generalized to include any particle parameter like the longitudinal position of a particle with respect to a reference particle, or the energy of a particle, the spin vector, or any particle coordinate that may depend on other coordinates. In the following paragraphs we will restrict the discussion to linear (\(2\times 2\)) transformation matrices for a variety of beam line elements.

#### Driftspace

In a driftspace of length \(\ell\) or in a weak bending magnet, where \(\kappa_{0x}^{2}\ll 1\) and \(k_{0}=0\), the focusing parameter \(K=0\) and the solution of (7.4) in matrix formulation can be expressed by

\[\left(\begin{array}{c}u(z)\\ u^{\prime}(z)\end{array}\right)=\left(\begin{array}{cc}1&\ell\\ 0&1\end{array}\right)\left(\begin{array}{c}u_{0}\\ u^{\prime}_{0}\end{array}\right). \tag{7.7}\]

A more precise derivation of the transformation matrices for bending magnets of arbitrary strength will be described later in this chapter. Any drift space of length \(\ell=z-z_{0}\), therefore, is represented by the simple transformation matrix

\[\mathcal{M}_{\mathrm{d}}(\ell|0)=\left(\begin{array}{cc}1&\ell\\ 0&1\end{array}\right). \tag{7.8}\]

We recognize the expected features of a particle trajectory in a field free drift space. The amplitude \(u\) changes only if the trajectory has an original non vanishing slope \(u^{\prime}_{0}\neq 0\) while the slope itself does not change at all.

#### Quadrupole Magnet

For a pure quadrupole the bending term \(\kappa_{0x}=0\) and the field gradient or quadrupole strength \(k(z)\neq 0\) can be positive as well as negative. With these assumptions we solve again (7.4) and determine the integration constants by initial conditions. For \(k>0\) we get the transformation for a focusing quadrupole

\[\left(\begin{array}{c}u(z)\\ u^{\prime}(z)\end{array}\right)=\left(\begin{array}{cc}\cos\psi&\frac{1}{ \sqrt{k}}\sin\psi\\ -\sqrt{k}\sin\psi&\cos\psi\end{array}\right)\left(\begin{array}{c}u(z_{0}) \\ u^{\prime}(z_{0})\end{array}\right)\,, \tag{7.9}\]

where \(\psi=\sqrt{k}(z-z_{0}\). This equation is true for any section within the quadrupole as long as both points \(z_{0}\) and \(z\) are within the active length of the quadrupole.

For a full quadrupole of length \(\ell\) and strength \(k\) we set \(\varphi=\sqrt{k\ell}\) and the transformation matrix for a full quadrupole in the focusing plane is

\[\mathcal{M}_{\mathrm{QF}}\left(\ell\left|0\right.\right)=\left(\begin{array} []{cc}\cos\varphi&\frac{1}{\sqrt{k}}\sin\varphi\\ -\sqrt{k}\sin\varphi&\cos\varphi\end{array}\right). \tag{7.10}\]Similarly, we get in the other plane with \(k<0\) the solution for a defocusing quadrupole

\[\left(\begin{array}{c}u(z)\\ u^{\prime}(z)\end{array}\right)=\left(\begin{array}{cc}\cosh\psi&\frac{1}{ \sqrt{|k|}}\sinh\psi\\ \sqrt{|k|}\sinh\psi&\cosh\psi\end{array}\right)\left(\begin{array}{c}u(z_{0} )\\ u^{\prime}(z_{0})\end{array}\right), \tag{7.11}\]

where \(\psi=\sqrt{|k|}\left(z-z_{0}\right)\). The transformation matrix in the defocusing plane through a complete quadrupole of length \(\ell\) with \(\varphi=\sqrt{|k|}\ell\) is therefore

\[\mathcal{M}_{\mathrm{QD}}\left(\ell\left|0\right.\right)=\left(\begin{array}[ ]{cc}\cosh\varphi&\frac{1}{\sqrt{|k|}}\sinh\varphi\\ \sqrt{|k|}\sinh\varphi&\cosh\varphi\end{array}\right). \tag{7.12}\]

These transformation matrices make it straight forward to follow a particle through a transport line. Any arbitrary sequence of drift spaces, bending magnets and quadrupole magnets can be represented by a series of transformation matrices \(\mathcal{M}_{i}\). The transformation matrix for the whole composite beam line is then just equal to the product of the individual matrices. For example, by multiplying all matrices along the path in Fig. 7.2 the total transformation matrix \(\mathcal{M}\) for the eight magnetic elements of this example is determined by the product

\[\mathcal{M}=\mathcal{M}_{8}\ldots\mathcal{M}_{4}\mathcal{M}_{3}\mathcal{M}_{2 }\mathcal{M}_{1} \tag{7.13}\]

and the particle trajectory transforms through the whole composite transport line like

\[\left(\begin{array}{c}u(z)\\ u^{\prime}(z)\end{array}\right)=\mathcal{M}\left(z\left|z_{0}\right.\right) \,\left(\begin{array}{c}u(z_{0})\\ u^{\prime}(z_{0})\end{array}\right), \tag{7.14}\]

where the starting point \(z_{0}\) in this case is at the beginning of the drift space \(\mathcal{M}_{1}\) and the end point \(z\) is at the end of the magnet \(\mathcal{M}_{8}\).

Figure 7.2: Example of a beam transport line (schematic)

#### Thin Lens Approximation

As will become more apparent in the following sections, this matrix formalism is widely used to calculate trajectories for individual particle or for a virtual particle representing the central path of a whole beam. The repeated multiplication of matrices, although straightforward, is very tedious and therefore, most beam dynamics calculations are performed on digital computers. In some cases, however, it is desirable to analytically calculate the approximate properties of a small set of beam elements. For these cases it is sufficient to use what is called the thin lens approximation. In this approximation it is assumed that the length of a quadrupole magnet is small compared to its focal length (\(\ell\ll f\)) and we set

\[\ell\to 0\,, \tag{7.15}\]

while keeping the focal strength constant,

\[f^{-1}=+k\,l=\mathrm{const}\,. \tag{7.16}\]

This result is analogous to geometric light optics, where we assume the glass lenses to be infinitely thin. As a consequence \(\varphi=\sqrt{k}\,\ell\to 0\) and the transformation matrices (7.10,) (7.12) are the same in both planes except for the sign of the focal length

\[\left(\begin{array}{c}u(z)\\ u^{\prime}(z)\end{array}\right)=\left(\begin{array}{cc}1&\ell\\ -\frac{1}{f}&1\end{array}\right)\left(\begin{array}{c}u_{0}\\ u_{0}^{\prime}\end{array}\right)\,, \tag{7.17}\]

where

\[\begin{array}{l}f^{-1}=k\,\ell>0\ \ \ \text{in the focusing plane}\\ f^{-1}=k\,\ell<0\ \ \ \text{in the defocusing plane}.\end{array} \tag{7.18}\]

The transformation matrix has obviously become very simple and exhibits only the focusing property in form of the focal length. Quite generally one may regard for single as well as composite systems the matrix element \(M_{21}\) as the element that expresses the focal strength of the transformation.

In thin lens approximation it is rather easy to derive focusing properties of simple compositions of quadrupoles. A quadrupole doublet composed of two quadrupole magnets separated by a drift space of length \(L\) is described by the total transformation matrix

\[\mathcal{M}_{\mathrm{db}}\left(L\ |0\right) =\left(\begin{array}{cc}1&0\\ -\frac{1}{f_{2}}&1\end{array}\right)\left(\begin{array}{cc}1&L\\ 0&1\end{array}\right)\left(\begin{array}{cc}1&0\\ -\frac{1}{f_{1}}&1\end{array}\right) \tag{7.19}\] \[=\left(\begin{array}{cc}1-L/f_{1}&L\\ -1/f^{*}&1-L/f_{2}\end{array}\right),\]where we find the well known expression from geometric paraxial light optics

\[\frac{1}{f^{\ast}}=\frac{1}{f_{1}}+\frac{1}{f_{2}}-\frac{L}{f_{1}f_{2}}\,. \tag{7.20}\]

Such a doublet can be made focusing in both planes if, for example, the quadrupole strengths are set such that \(f_{1}=\neg f_{2}=f\). The total focal length then is \(f^{\ast}=+L/f^{2}>0\) in both the horizontal and the vertical plane.

This simple result, where the focal length is the same in both planes, is a valid solution only in thin lens approximation. For a doublet of finite length quadrupoles the focal length in the horizontal plane is always different from that in the vertical plane as can be verified by using the transformations (7.10), (7.12) to calculate the matrix \(\mathcal{M}_{\text{db}}\). Since individual matrices are not symmetric with respect to the sign of the quadrupole field, the transformation matrices for the horizontal plane \(\mathcal{M}_{\text{db,x}}\) and the vertical plane \(\mathcal{M}_{\text{db,y}}\) must be calculated separately and turn out to be different. In special composite cases, where the quadrupole distribution is symmetric as shown in Fig. 7.3, the matrices for both of the two symmetric half sections are related in a simple way. If the matrix for one half of the symmetric beam line is

\[\mathcal{M}=\left(\begin{array}{cc}a&b\\ c&d\end{array}\right) \tag{7.21}\]

then the reversed matrix for the second half of the beam line is

\[\mathcal{M}_{\text{r}}=\left(\begin{array}{cc}d&b\\ c&a\end{array}\right) \tag{7.22}\]

and the total symmetric beam line has the transformation matrix

\[\mathcal{M}_{\text{tot}}=\mathcal{M}_{\text{r}}\,\mathcal{M}=\left(\begin{array} []{cc}ad+bc&2bd\\ 2ac&ad+bc\end{array}\right). \tag{7.23}\]

Figure 7.3: Reversed lattice

We made no particular assumptions for the lattice shown in Fig. 7.3 except for symmetry and the relations (7.21), (7.22) are true for any arbitrary but symmetric beam line.

The result for the reversed matrix is not to be confused with the inverse matrix, where the direction of the particle path is also reversed. The inverses matrix of (7.21) is

\[{\cal M}_{\rm i}=\left(\begin{array}{cc}d&-b\\ -c&a\end{array}\right). \tag{7.24}\]

Going through an arbitrary section of a beam line and then back to the origin again results in a total transformation matrix equal to the unity matrix

\[{\cal M}_{\rm tot}={\cal M}_{\rm i}\,{\cal M}=\left(\begin{array}{cc}1&0\\ 0&1\end{array}\right). \tag{7.25}\]

These results allow us now to calculate the transformation matrix \({\cal M}_{\rm tr}\) for a symmetric quadrupole triplet. With (7.19), (7.24) the transformation matrix of a quadrupole triplet as shown in Fig. 7.4 is

\[{\cal M}_{\rm tr}={\cal M}_{\rm r}\,{\cal M}=\left(\begin{array}{cc}1-2L^{2} /f^{\;2}&2L\left(1+L/f\right)\\ -1/f^{*}&1-2L^{2}/f^{\;2}\end{array}\right), \tag{7.26}\]

where \(f^{*}\) is defined by (7.20) with \(f_{1}=-f_{2}=f\).

Such a triplet is focusing in both planes as long as \(f>L\). Symmetric triplets as shown in Fig. 7.4 have become very important design elements of long beam transport lines or circular accelerators since such a triplet can be made focusing in both planes and can be repeated arbitrarily often to provide a periodic focusing structure called a FODO-channel. The acronym is derived from the sequence of focusing (F) and defocusing (D) quadrupoles separated by non-focusing elements (O) like a drift s

Figure 7.4: Symmetric quadrupole triplet

#### Quadrupole End Field Effects

In defining the transformation through a quadrupole we have assumed the strength parameter \(k(z)\) to be a step function with a constant nonzero value within the quadrupole and zero outside. Such a hard edge field distribution is only approximately true for a real quadrupole. The strength parameter in a real quadrupole magnet varies in a gentle way from zero outside the quadrupole to a maximum value in the middle of the quadrupole. In Fig. 7.5 the measured gradient of a real quadrupole along the axis is shown.

The field extends well beyond the length of the iron core and the effective magnetic length, defined by

\[\ell_{\mathrm{eff}}=\frac{\int g\,\mathrm{d}z}{g_{0}}, \tag{7.27}\]

where \(g_{0}\) is the field gradient in the middle of the quadrupole, is longer than the iron length by about the radius of the bore aperture

\[\ell_{\mathrm{eff}}\approx\ell_{\mathrm{iron}}+R\,. \tag{7.28}\]

This is the effective or hard edge magnet length \(\ell_{0}\) with strength \(k\). The real field distribution can be approximated by a trapezoid such that \(\int g\,\mathrm{d}z\) is the same in both cases (see Fig. 7.5). To define the trapezoidal approximation we assume a fringe field extending over a length equal to the bore radius \(R\) as shown in Fig. 7.5. End field effects must therefore be expected specifically in quadrupoles with large bore radii and short iron cores. It is interesting to investigate as to what extend the transformation characteristics for a real quadrupole differ from the hard edge model. The real transformation matrix can be obtained by slicing the whole axial quadrupole field distribution in thin segments of varying strength. Treating these segments as short hard edge quadrupoles the full transformation matrix is the product of the matrices for all segments.

Figure 7.5: Field profile in a real quadrupole with a bore radius of \(R=3\,\mathrm{cm}\) and an iron length of \(\ell_{\mathrm{iron}}=15.9\,\mathrm{cm}\)While it is possible to obtain an accurate transformation matrix this way the variations of the matrix elements due to this smooth field distribution turn out to be mostly small and in practice, therefore, the hard edge model is used to develop beam transport lattices. Nonetheless after a satisfactory solution has been found, these real transformation matrices should be used to check the solution and possibly make small adjustment to the idealized hard edge model design.

In this section, we will discuss an analytical estimate of the correction to be expected for a real field distribution [5] by looking for the "effective" hard edge model parameters (\(k,\ell\)) which result in a transformation matrix equal to the transformation matrix for the corresponding real quadrupole. The transformation matrix for the real quadrupole be

\[{\cal M}_{{}_{\rm Q}}=\left(\begin{array}{cc}C&S\\ C^{\prime}&S^{\prime}\end{array}\right), \tag{7.29}\]

where the matrix elements are the result of multiplying all "slice" matrices for the quadrupole segments as shown in Fig. 7.6 over the length \(L\).

We assume now that this real quadrupole can be represented by a hard edge model quadrupole of length \(\ell\) with adjacent drift spaces \(\lambda\) as indicated in Fig. 7.6. The transformation through this system for a focusing quadrupole is given by [5]

\[\left(\begin{array}{cc}1&\lambda\\ 0&1\end{array}\right)\left(\begin{array}{cc}\cos\varphi&\frac{1}{\sqrt{k}} \sin\varphi\\ -\sqrt{k}\sin\varphi&\cos\varphi\end{array}\right)\left(\begin{array}{cc}1& \lambda\\ 0&1\end{array}\right)\]

\[=\left(\begin{array}{cc}\cos\varphi-\sqrt{k}\lambda\sin\varphi&2\lambda\cos \varphi-\frac{1}{\sqrt{k}}\sin\varphi\\ -\sqrt{k}\sin\varphi&\cos\varphi-\sqrt{k}\lambda\sin\varphi\end{array}\right) \tag{7.30}\]

Figure 7.6: Decomposition of an actual quadrupole field profile into segments of hard edge quadrupoles. (\(k_{0},\ell_{0}\) are for the hard edge model, \(k,\ell\) for the hard edge model with real fringe fields, \(\lambda\) and \(L\) are used for mathematical evaluation only)

with \(\varphi=\sqrt{k}\,\ell_{0}\). This hard edge transformation matrix must be the same as the actual matrix (7.29) and we will use this equality to determine the effective quadrupole parameters \(k,\ell\). First, we note that the choice of the total length \(L=\ell_{0}+2\lambda\) is arbitrary as long as it extends over the whole field profile, and both, the "slices" and hard edge matrices extend over the whole length \(L\) by employing drift spaces if necessary. Equating (7.29) and (7.30) we can compose two equations which allow us to determine the effective parameters \(k,\ell\) from known quantities

\[\begin{array}{l}C_{\rm f}-\frac{1}{2}LC_{\rm r}^{\prime}=\cos\varphi_{\rm f} +\frac{1}{2}\varphi_{\rm r}\sin\varphi_{\rm r}\,,\\ C_{\rm f}^{\prime}\,\ell_{\rm r}=-\varphi_{\rm f}\sin\varphi_{\rm r}\,.\end{array} \tag{7.31}\]

Here we have added the index \({}_{\rm f}\) to indicate a focusing quadrupole. The first of these equations can be solved for \(\varphi_{\rm f}\) since the quantities \(C_{\rm r},C_{\rm f}^{\prime}\), and \(L\) are known. The second equation then is solved for \(\ell_{\rm r}\) and \(k_{\rm f}=\varphi_{\rm r}^{2}/\ell_{\rm r}\). Two parameters are sufficient to equate the \(2\times 2\) matrices (7.29), (7.30) since two of the four equations are redundant for symmetry reasons, \(M_{11}=M_{22}=C=S^{\prime}\), and the determinant of the matrices on both sides must be unity. Similarly, we get for a defocusing quadrupole

\[\begin{array}{l}C_{\rm d}-\frac{1}{2}LC_{\rm d}^{\prime}=\cosh\varphi_{\rm d }-\frac{1}{2}\varphi_{\rm d}\sinh\varphi_{\rm d}\,,\\ C_{\rm d}^{\prime}\ell_{\rm d}=-\varphi_{\rm d}\sinh\varphi_{\rm d}\,.\end{array} \tag{7.32}\]

Equations (7.31) and (7.32) define a hard edge representation of a real quadrupole. However, we note that the effective quadrupole length \(\ell\) and strength \(k\) are different from the customary definition, where \(k_{0}\) is the actual magnet strength in the middle of the quadrupole and the magnet length is defined by \(\ell_{0}=\frac{1}{k_{0}}\int k(z)\,{\rm d}z\). We also observe that the effective values \(\ell\) and \(k\) are different for the focusing and defocusing plane. Since the endfields are not the same for all quadrupoles but depend on the design parameters of the magnet we cannot determine the corrections in general. In practical cases, however, it turns out that the corrections \(\Delta k=k-k_{0}\) and \(\Delta\ell=\ell-\ell_{0}\) are small for quadrupoles which are long compared to the aperture and are larger for short quadrupoles with a large aperture. In fact the differences \(\Delta k\) and \(\Delta\ell\) turn out to have opposite polarity and the thin lens focal length error \(\Delta k\,\Delta\ell\) is generally very small.

As an example, we use the quadrupole of Fig. 7.5 and calculate the corrections due to end field effects. We calculate the total transformation matrix for the real field profile as discussed above by approximating the actual field distribution by a series of hard edge "slice" matrices in both planes as a function of the focusing strength \(k_{0}\) and solve (7.31), (7.32) for the effective parameters \((k_{\rm f},\ell_{\rm f})\) and \((k_{\rm d},\ell_{\rm d})\), respectively. In Fig. 7.7 these relative fringe field corrections to the quadrupole strength \(\Delta k/k_{0}\) and to the quadrupole length \(\Delta\ell/\ell_{0}\) are shown as functions of the strength \(k_{0}\). The effective quadrupole length is longer and the effective quadrupole strength is lower than the pure hard edge values. In addition the corrections are different in both planes. Depending on the sensitivity of the beam transport system these corrections may have to be included in the final optimization.

### Focusing in Bending Magnets

Bending magnets have been treated so far just like drift spaces as far as focusing properties are concerned. This is a good approximation for weak bending magnets which bend the beam only by a small angle. In cases of larger deflection angles, however, we observe focusing effects which are due to the particular type of magnet and its end fields. In Chap. 6 we discussed the geometric focusing term \(\kappa^{2}\) which appears in sector magnets only. Other focusing terms are associated with bending magnets and we will discuss in this section these effects in a systematic way. Specifically, the focusing of charged particles crossing end fields at oblique angles will be discussed.

The linear theory of particle beam dynamics uses a curvilinear coordinate system following the path of the reference particle and it is assumed that all magnetic fields are symmetric about this path. The "natural" bending magnet in this system is one, where the ideal path of the particles enters and exits normal to the magnet pole faces. Such a magnet is called a sector magnet as shown in Fig. 7.8. The total deflection of a particle depends on the distance of the particle path from the ideal path in the deflecting plane which, for simplicity, we assume to be in the horizontal \(x\)-plane. Particles following a path at a larger distance from the center of curvature than the ideal path travel a longer distance through this magnet and, therefore, are deflected by a larger angle than a particle on the ideal path. Correspondingly, a particle passing through the magnet closer to the center of curvature is deflected less.

This asymmetry leads to a focusing effect which is purely geometric in nature. On the other hand, we may choose to use a magnet such that the ideal path of the particle beam does not enter the magnet normal to the pole face but rather at an angle. Such a configuration has an asymmetric field distribution about the beam axis and therefore leads to focusing effects. We will discuss the effects of fringe fields in more detail in Sect. 7.3.2.

Figure 7.7: Fringe field correction for the quadrupole of Fig. 7.5 with a bore radius of \(R=3.0\,\mathrm{cm}\) and a steel length of \(\ell_{\mathrm{iron}}=15.9\,\mathrm{cm}\)

#### Sector Magnets

The degree of focusing in a sector magnet can be evaluated in any infinitesimal sector of such a magnet by calculating the deflection angle as a function of the particle position \(x\). With the notation from Fig. 7.8 we get for the deflection angle while keeping only linear terms in \(x\)

\[\mathrm{d}\theta\,=\kappa_{0}\mathrm{d}\sigma\,=\kappa_{0}\left(1+\kappa_{0}x \right)\mathrm{d}z\,. \tag{7.33}\]

The first term on the r.h.s. merely defines the ideal path, while the second \(x\)-dependent term of the deflection angle in (7.33) describes the particle motion in the vicinity of the ideal path. With respect to the curvilinear coordinate system following the ideal path we get the additional deflection

\[\delta\theta\,=\kappa_{0}^{2}\,x\mathrm{d}z\,. \tag{7.34}\]

This correction is to be included in the differential equation of motion as an additional focusing term

\[\Delta x^{\prime\prime}=-\frac{\delta\theta}{\mathrm{d}z}=-\kappa_{0}^{2}\,x \tag{7.35}\]

to the straight quadrupole focusing leading to the equation of motion

\[x^{\prime\prime}+\left(k+\kappa_{0}^{2}\right)\,x=0\,, \tag{7.36}\]

which is identical to the result obtained in Sect. 5.3.

The differential equation (7.36) has the same form as that for a quadrupole and therefore the solutions must be of the same form. Using this similarity we replaceby \((k+\kappa_{0}^{2})\) and obtain immediately the transformation matrices for a general sector magnet. For \(K=k+\kappa_{0}^{2}>0\) and

\[\Theta=\sqrt{K}\ell \tag{7.37}\]

we get from (7.10) the transformation matrix

\[\mathcal{M}_{\mathrm{sy,f}}(\ell\,|\,0)=\left(\begin{array}{cc}\cos\Theta& \frac{1}{\sqrt{K}}\sin\Theta\\ -\sqrt{K}\sin\Theta&\cos\Theta\end{array}\right), \tag{7.38}\]

where \(\ell\) is the arc length of the sector magnet and where both the focusing term \(k\) and the bending term \(\kappa_{0}\) may be nonzero. Such a magnet is called a synchrotron magnet since this magnet type was first used for lattices of synchrotrons.

For the defocusing case, where \(K=k+\kappa_{0}^{2}<0\) and \(\Theta=\sqrt{|K|}\ell\), we get from (7.12)

\[\mathcal{M}_{\mathrm{sy,d}}(\ell|0)=\left(\begin{array}{cc}\cosh\Theta& \frac{1}{\sqrt{|K|}}\sinh\Theta\\ \sqrt{|K|}\sinh\Theta&\cosh\Theta\end{array}\right). \tag{7.39}\]

Note that the argument \(\Theta\) is equal to the deflection angle \(\theta\) only in the limit \(k\to 0\) because these transformation matrices include bending as well as focusing in the same magnet. Obviously, in the nondeflecting plane \(\kappa_{0}=0\) and such a magnet acts just like a quadrupole with strength \(k\) and length \(\ell\).

A subset of general sector magnets are pure dipole sector magnets, where we eliminate the focusing by setting \(k=0\) and get the pure dipole strength \(K=\kappa_{0}^{2}>0\). The transformation matrix for a pure sector magnet of length \(\ell\) and bending angle \(\theta\ =\ \kappa_{0}\ell\) in the deflecting plane becomes from (7.38)

\[\mathcal{M}_{\mathrm{s,\rho}}(\ell|0)=\left(\begin{array}{cc}\cos\theta& \rho_{0}\sin\theta\\ -\kappa_{0}\sin\Theta&\cos\Theta\end{array}\right). \tag{7.40}\]

If we also let \(\kappa_{0}\to 0\) we arrive at the transformation matrix of a sector magnet in the nondeflecting plane

\[\mathcal{M}_{\mathrm{s,0}}(\ell|0)=\left(\begin{array}{cc}1&\ell\\ 0&1\end{array}\right), \tag{7.41}\]

which has the form of a drift space. A pure dipole sector magnet therefore behaves in the non-deflecting plane just like a drift space of length \(\ell\). Note that \(\ell\) is the arc length of the magnet while the engineering magnet length might be given as the straight length between entry and exit point.

#### Fringe Field Effects

The results obtained above are those for a hard edge model and do not reflect modifications caused by the finite extend of the fringe fields. The hard edge model is again an idealization and for a real dipole we consider the gradual transition of the field from the maximum value to zero outside the magnet. The extend of the dipole fringe field is typically about equal to the gap height or distance between the magnet poles.

We assume magnet poles which are very wide compared to the gap height and therefore transverse field components in the deflecting plane, here \(B_{x}\), can be neglected. At the entrance into a magnet the vertical field component \(B_{y}\) increases gradually from the field free region to the maximum value in the middle of the magnet (Fig. 7.9). We will discuss the effects on the particle dynamics caused by this fringe field and compare it with the results for a hard edge model magnet.

For the following discussion we consider both a fixed orthogonal Cartesian coordinate system \((u,v,w)\), used in the fringe area, as well as a moving curvilinear system \((x,y,z)\). The origin of the fixed coordinate system is placed at the point \(P_{0}\) where the field starts to rise (Fig. 7.9). At this point both coordinate systems coincide. The horizontal field component vanishes for reasons of symmetry

\[B_{u}=0 \tag{7.42}\]

and the vertical field component in the fringe region may be described by

\[B_{v}=F(w)\;. \tag{7.43}\]

With Maxwell's curl equation \(\partial B_{w}/\partial v-\partial B_{v}/\partial w=0\) we get after integration the longitudinal field component \(B_{w}=\int(\partial B_{v}/\partial w)\,\mathrm{d}v\) or

\[B_{w}=y\frac{\partial F(w)}{\partial w}\;, \tag{7.44}\]

Figure 7.9: End field profile in a dipole magnet and fringe field focusingwhere \(y=v\) and where a linear fringe field (see Fig. 7.9) was assumed with \(\partial F(w)/\partial w=\text{const}\). These field components must be expressed in the curvilinear coordinate system \((x,y,z)\). Within the fringe field \(B_{w}\,(z)\) can be split into \(B_{x}\) and \(B_{z}\) as shown in Fig. 7.9. The horizontal field component is then \(B_{x}=B_{w}\sin\delta\) where \(\delta\) is the deflection angle at the point \(z\) defined by

\[\delta=\frac{e}{p_{0}}\int_{0}^{z}F(\bar{z})\,\mathrm{d}\bar{z}\,. \tag{7.45}\]

With

\[B_{w}=y\,\frac{\partial F(w)}{\partial w}=y\,\frac{\partial F(w)}{\partial z} \,\frac{\,\mathrm{d}z}{\,\mathrm{d}w}\,\approx y\,\frac{\partial F(z)}{ \partial z}\,\frac{1}{\cos\delta} \tag{7.46}\]

we get

\[B_{x}(z)=y\,F^{\prime}(z)\,\tan\delta\,, \tag{7.47}\]

where \(F^{\prime}(z)=\mathrm{d}F/\mathrm{d}z\). The vertical fringe field component is with \(\partial B_{x}/\partial y-\partial B_{y}/\partial x=0\) and integration

\[B_{y}(z)=B_{y0}+x\,F^{\prime}(z)\,\tan\delta\,. \tag{7.48}\]

The longitudinal field component is from (7.46) and with \(B_{z}\,=\,B_{w}\,\cos\delta\)

\[B_{z}(z)=y\,F^{\prime}(z)\,. \tag{7.49}\]

The field components of the fringe field depend linearly on the transverse coordinates and therefore fringe field focusing [6] must be expected. With the definition of the focal length from (7.3) we get

\[\frac{1}{f}=\int_{0}^{z_{\mathrm{f}}}K(\bar{z})\,\mathrm{d}\bar{z}\,, \tag{7.50}\]

where \(K(z)\) is the focusing strength parameter defined in (). In the deflecting plane the fringe field focusing is with \(k(z)=(e/p_{0})\,\partial B_{y}(z)/\partial x\) and (7.48)

\[\frac{1}{f_{x}}=\int_{0}^{z_{\mathrm{f}}}(\kappa^{\prime}\tan\delta+\kappa^{2} )\,\mathrm{d}\bar{z}\,, \tag{7.51}\]

where we have set \(\kappa(z)=(e/p_{0})\,F(z)\). For small deflection angles \(\delta\) in the fringe field \(\tan\delta\approx\delta=\int_{0}^{z_{\mathrm{f}}}\kappa\,\mathrm{d}\bar{z}\) and after integration of (7.48) by parts through the full fringe field we get the focal length while neglecting higher order terms in \(\delta_{\mathrm{f}}\)

\[\frac{1}{f_{x}}=\kappa_{0}\,\delta_{\mathrm{f}}\,, \tag{7.52}\]where \(\kappa_{0}=1/\rho_{0}\) is the curvature in the central part of the magnet and \(\delta_{\rm f}\) is the total deflection angle in the fringe field region.

This result does not deviate from that of the hard edge model, where for a small deflection angle \(\delta\) we have from (7.40) \(1/f_{x}\approx\kappa_{0}\,\delta\) agreeing with (7.52). We obtain therefore the convenient result that in the deflecting plane of a sector magnet there is no need to correct the focusing because of the finite extend of the fringe field.

#### Finite Pole Gap

In the vertical plane this situation is different since we expect vertical focusing from (7.47) while there is no focusing in the approximation of a hard edge model. Using the definition (7.50) of the focal length in the vertical plane gives with \(K(z)=-k(z)\) and (7.47)

\[\frac{1}{f_{y}}=-\int_{0}^{z_{\rm f}}\kappa^{\prime}\tan\delta\,{\rm d}\bar{z} \approx-\int_{0}^{z_{\rm f}}\kappa^{\prime}(\bar{z})\,\delta(\bar{z})\,{\rm d} \bar{z}\,. \tag{7.53}\]

The fringe field of a sector magnet therefore leads to a defocusing effect which depends on the particular field profile. We may approximate the fringe field by a linear fit over a distance approximately equal to the pole gap \(2G\) which is a good approximation for most real dipole magnets. We neglect the nonlinear part of the fringe field and approximate the slope of the field strength by \(\kappa^{\prime}=\kappa_{0}/2G=\)const. The focal length for the full fringe field of length \(z_{\rm f}=2G\) is therefore with \(\kappa(z)=\kappa^{\prime}z,\ 0\leq z\leq z_{\rm f}\) and

\[\delta(z)=\int_{0}^{z}\kappa^{\prime}\bar{z}\,{\rm d}\bar{z}=\frac{\kappa_{0} }{4G}z^{2} \tag{7.54}\]

given by

\[\frac{1}{f_{y}}=-\int_{0}^{2G}\kappa^{\prime}\delta(\bar{z})\,{\rm d}\bar{z}= -\tfrac{1}{3}\kappa_{0}^{2}\,G=-\tfrac{1}{3}\kappa_{0}\,\delta_{\rm f}\,, \tag{7.55}\]

where

\[\delta_{\rm f}=\kappa_{0}\,G\,. \tag{7.56}\]

This is the focusing due to the fringe field at the entrance of a sector magnet. At the exit we have the same effect since the sign change of \(\kappa^{\prime}\) is compensated by the need to integrate now from full field to the field free region which is just opposite to the case in the entrance fringe field. Both end fields of a sector magnet provide a small vertical defocusing. We note that this defocusing is quadratic in nature, since \(\delta_{\rm f}\propto\kappa_{0}\) and therefore independent of the sign of the deflection.

With these results we may now derive a corrected transformation matrix for a sector magnet by multiplying the hard edge matrix (7.41)on either side with thin length fringe field focusing

\[\left(\begin{array}{cc}1&0\\ -\frac{1}{\hat{f}_{y}}&1\end{array}\right)\,\left(\begin{array}{cc}1&\ell\\ 0&1\end{array}\right)\,\left(\begin{array}{cc}1&0\\ -\frac{1}{\hat{f}_{y}}&1\end{array}\right) \tag{7.57}\]

and get with (7.55) and \(\theta=\ell/\rho_{0}\) for the transformation matrix in the vertical, non-deflecting plane of a sector magnet instead of (7.41)

\[\mathcal{M}_{\text{s},0}(\ell\,|\,0)=\left(\begin{array}{cc}1+\frac{1}{3} \theta\,\delta_{\text{f}}&\ell\\ \frac{2}{3}\frac{\delta_{\text{f}}}{\rho_{0}}-\frac{1}{9}\frac{\delta_{\text{ f}}^{2}}{\rho_{0}^{2}}\ell&1+\frac{1}{3}\theta\,\delta_{\text{f}}\end{array}\right). \tag{7.58}\]

The second order term in the \(M_{21}\)-matrix element can be ignored for practical purposes but is essential to keep the determinant equal to unity.

#### Wedge Magnets

In a more general case compared to a sector magnet we will allow the reference path of the particle beam to enter and exit the magnet at an arbitrary angle with the pole face. Figure 7.10 shows such a wedge magnets and we will derive its transformation matrices. First, we note that the fringe field effect is not different from the previous case of a sector magnet except that now the angle \(\delta(z)\) must be replaced by a new angle \(\eta+\delta(z)\) where the pole rotation angle \(\eta\) and the sign convention is defined in Fig. 7.10.

Figure 7.10: Fringe field focusing in wedge magnets

Different from the case of a sector magnet, we cannot replace the tangent in (7.51) by its argument since the angle \(\eta\) may be large to prohibit such an approximation. As a further consequence of a large value of \(\eta\), we must take into account the actual path length in the fringe field. To calculate the focal length\(f_{x}\), we have instead of (7.51)

\[\frac{1}{f_{x}}=\int_{0}^{z_{\rm f}}\left[\kappa^{\prime}\tan\left(\eta+\delta \right)+\kappa^{2}\right]\,{\rm d}\bar{z} \tag{7.59}\]

Expanding for small angles \(\delta\ll 1\) we get \(\tan\left(\eta+\delta\right)\approx\tan\eta+\delta\). This approximation is true only as long as \(\delta\tan\eta\ll 1\) or for entrance angles \(\eta\) not too close to \(90^{\circ}\) and the argument in the integral (7.59) becomes \(\kappa^{\prime}\tan\eta+\kappa^{\prime}\delta+\kappa^{2}\). In addition to the terms for a sector magnet, a new term (\(\kappa^{\prime}\tan\eta\)) appears and the focal length of the fringe field is

\[\frac{1}{f_{x}}=\int_{0}^{z_{\rm f}}\kappa^{\prime}\tan\eta\,{\rm d}\bar{z}+ \kappa_{0}\delta_{\rm f}=\kappa_{0}\tan\eta+\kappa_{0}\delta_{\rm f}\,, \tag{7.60}\]

where the integral extends over the whole fringe field. Since to first order the path length through the fringe field is

\[z_{\rm f}=\frac{2G}{\cos\eta}\,, \tag{7.61}\]

where \(2G\) is the pole gap height, we have

\[\delta_{\rm f}=\int_{0}^{2G/\cos\eta}\kappa\,{\rm d}\bar{z}\,. \tag{7.62}\]

The term \(\kappa_{0}\delta_{\rm f}\) describes again the well-known focusing of a sector magnet in the deflecting plane while the term \(\kappa_{0}\tan\eta\) provides the correction necessary for non-normal entry of the beam path into the magnet. For the case shown in Fig. 7.10, where \(\eta>0\), we obtain beam focusing in the deflecting plane from the fringe field. Similarly, we get a focusing or defocusing effect at the exit fringe field depending on the sign of the pole rotation. The complete transformation matrix of a wedge magnet in the horizontal deflecting plane is obtained by multiplying the matrix of a sector magnet with thin lens matrices to take account of edge focusing. For generality, however, we must assume that the entrance and the exit angle may be different. We will therefore distinguish between the edge focusing for the entrance angle \(\eta=\eta_{0}\) and that for the exit angle \(\eta=\eta_{\rm e}\) and get for the transformation matrix in the deflecting plane

\[\mathcal{M}_{\rm w,\rho}\left(\ell,0\right)=\left[\begin{array}{cc}1&0\\ -\frac{1}{\rho_{0}}\tan\eta_{\rm e}&1\end{array}\right]\left[\begin{array}{ cc}\cos\theta&\rho_{0}\sin\theta\\ -\frac{1}{\rho_{0}}\sin\theta&\cos\theta\end{array}\right]\left[\begin{array}[] {cc}1&0\\ -\frac{1}{\rho_{0}}\tan\eta_{0}&1\end{array}\right]. \tag{7.63}\]In the vertical plane the focal length is similar to (7.53) and for not too large angles \(\eta\)

\[\frac{1}{f_{y}}=-\int_{0}^{zt}\kappa^{\prime}\tan\left(\eta+\delta\right)\,\mathrm{ d}\bar{z}\approx-\kappa_{0}\tan\eta-\int_{0}^{zt}\kappa^{\prime}\delta\, \mathrm{d}\bar{z}\,. \tag{7.64}\]

Again we have the additional focusing term which is now focusing in the vertical plane for \(\eta<0\). For a linear fringe field the focal length is in analogy to (7.55)

\[\frac{1}{f_{y}}=-\kappa_{0}\tan\eta+\tfrac{1}{3}\kappa_{0}\delta_{\mathrm{f}}\,, \tag{7.65}\]

where

\[\delta_{\mathrm{f}}=\int_{0}^{2G/\cos\eta}\kappa\,\mathrm{d}\bar{z}=\kappa^{ \prime}\frac{2G^{2}}{\cos^{3}\eta}=\frac{\kappa_{0}G}{\cos^{2}\eta}\,, \tag{7.66}\]

since \(\kappa\left(z\right)\approx\kappa^{\prime}z\) and \(\kappa^{\prime}=\kappa_{0}/\left(G/\cos\eta\right)\). The complete transformation matrix in the vertical plane for a horizontally deflecting wedge magnet becomes then

\[\mathcal{M}_{\mathrm{w,0}}\left(\ell,0\right)=\left[\begin{array}{cc}1&0\\ -\frac{1}{\rho_{0}}\left(\tan\eta_{\mathrm{e}}+\tfrac{1}{3}\delta_{\mathrm{f} _{\mathrm{e}}}\right)\,1\end{array}\right]\left[\begin{array}{cc}1&\ell\\ 0&1\end{array}\right]\left[\begin{array}{cc}1&0\\ -\frac{1}{\rho_{0}}\left(\tan\eta_{0}+\tfrac{1}{3}\delta_{\mathrm{f}_{0}} \right)\,1\end{array}\right]. \tag{7.67}\]

Equations (7.63) and (7.67) are for bending magnets with arbitrary entrance and exit angles \(\eta_{0}\) and \(\eta_{\mathrm{e}}\). We note specifically that the transformation in the nondeflecting plane becomes different from a simple drift space and find a focusing effect due to the magnet fringe fields which depends on the entrance and exit angles between particle trajectory and pole face.

This general derivation of the focusing properties of a wedge magnet must be taken with caution where the pole face rotations are very large. In spite of the finite pole rotation angles we have assumed that the particles enter the fringe field at the same location \(z\) along the beam line independent of the transverse particle amplitude \(x\). Similarly, the path length of the trajectory in such a wedge magnet depends on the particle amplitude \(x\) and slope \(x^{\prime}\). Obviously these are second order effects but may become significant in special cases.

#### Rectangular Magnet

A particular case of a wedge magnet is the rectangular magnet which has parallel end faces. If we install this magnet symmetrically about the intended particle trajectory the entrance and exit angles equal to half the bending angle as shown in Fig. 7.11.

For a deflection angle \(\theta\), \(\eta_{0}=\eta_{\rm e}=-\theta/2\) and the transformation matrix in the deflecting plane is from (7.63)

\[M_{\rm r,\rho}\left(\ell\mid 0\right) = \left(\begin{array}{cc}1&0\\ -\frac{\tan\eta_{\rm e}}{\rho_{0}}&1\end{array}\right)\left(\begin{array}{ cc}\cos\theta&\rho_{0}\sin\theta\\ -\frac{\sin\theta}{\rho_{0}}&\cos\theta\end{array}\right)\left(\begin{array}[] {cc}1&0\\ -\frac{\tan\eta_{0}}{\rho_{0}}&1\end{array}\right)\] \[= \left(\begin{array}{cc}1&\rho_{0}\sin\theta\\ 0&1\end{array}\right).\]

A rectangular dipole magnet transforms in the deflecting plane like a drift space of length \(\rho_{0}\sin\theta\) and does not focus the beam. Note, that the "magnet length" \(\ell\) defined by the deflection angle \(\theta=\ell/\rho_{0}\) is the arc length and is related to the straight magnet length \(L\) by

\[L=2\rho_{0}\sin\frac{\theta}{2}=2\rho_{0}\sin\frac{\ell}{2\rho_{0}}. \tag{7.69}\]

In the vertical plane we observe a focusing with the focal length

\[\frac{1}{f_{y}}=+\frac{1}{\rho_{0}}\left(\tan\frac{\theta}{2}-\frac{\delta_{ \theta/2}}{3}\right). \tag{7.70}\]

From (7.66) \(\delta_{\theta/2}=G/[\rho_{0}\,\cos(\theta/2)]\) and with (7.69) \(\delta_{\theta/2}=2G\,\tan(\theta/2)/L\). Inserting this in (7.70), we obtain for the transformation matrix of a rectangular bending magnet in the nondeflecting plane

\[{\cal M}_{\rm r,0}(\ell|0) = \left(\begin{array}{cc}1&0\\ -\frac{1}{f_{y}}&1\end{array}\right)\left(\begin{array}{cc}1&\ell\\ 0&1\end{array}\right)\left(\begin{array}{cc}1&0\\ -\frac{1}{f_{y}}&1\end{array}\right)\ =\left(\begin{array}{cc}1-\frac{\ell}{f_{y}}& \ell\\ -\frac{2}{f_{y}}+\frac{\ell}{f_{y}^{2}}&1-\frac{\ell}{f_{y}}\end{array}\right), \tag{7.71}\]

Figure 7.11: Rectangular magnet

where

\[\frac{1}{f_{y}}=\frac{1}{\rho_{0}}\left(1-\frac{2G}{3L\cos\left(\theta/2\right)} \right)\tan\left(\frac{\theta}{2}\right)\,. \tag{7.72}\]

In a rectangular dipole magnet we find just the opposite edge focusing properties compared to a sector magnet. The focusing in the deflecting plane of a sector magnet has shifted to the vertical plane in a rectangular magnet and focusing is completely eliminated in the deflecting plane. Because of the finite extend of the fringe field, however, the focusing strength is reduced by the fraction \(2G/\left[3L\cos\left(\theta/2\right)\right]\) where \(2G\) is the gap height and \(L\) the straight magnet length.

#### Focusing in a Wiggler Magnet

The derivation of fringe field focusing in ordinary dipole magnets as discussed in previous sections can be directly applied to wiggler magnets. The beam path in a wiggler magnet is generally not parallel to the reference trajectory \(z\) because of the transverse deflection in the wiggler field and follows a periodic sinusoidal form along the reference path. For this reason the field component \(B_{z}\) appears to the particle partially as a transverse field \(B_{\xi}=B_{z}\tan\vartheta\,\approx\,B_{z}\,\vartheta\), where we use for a moment \(\xi\) as an auxiliary transverse coordinate normal to and in the plane of the actual wiggling beam path. We also assume that the wiggler deflection angle is small, \(\vartheta\,\ll 1\). The field component \(B_{\xi}\) can be expressed with (6.116), (6.117) more explicitly by

\[\frac{e}{p}B_{\xi}=-\left[\frac{1}{\rho_{0}}\sin\left(k_{\rm p}z\right)\right] ^{2}\frac{\sinh\left(k_{\rm p}y\right)\cosh\left(k_{\rm p}y\right)}{k_{\rm p}} \tag{7.73}\]

where \(1/\rho_{0}=\frac{e}{p}B_{0}\) is the inverse bending radius in the center of a wiggler pole at which point the field reaches the maximum value \(B_{0}\). With the expansions (6.119) we have finally

\[\frac{e}{p}B_{\xi}=-\left[\frac{1}{\rho_{0}}\,\sin\left(k_{\rm p}z\right) \right]^{2}\left(y+\tfrac{2}{3}k_{\rm p}^{2}y^{3}+\dots\right). \tag{7.74}\]

The linear \(y\)-dependence is similar to that found to produce vertical focusing in wedge magnets. Since the wiggler field appears quadratically in (7.73) \(B_{\xi}(z)=B_{\xi}(-z)\) and \(B_{\xi}(B_{0})=B_{\xi}(-B_{0})\). In other words, the transverse field has the same sign along all wiggler poles independent of the polarity of the vertical main wiggler field. The integrated field gradient per wiggler half pole is from (7.74)

\[k_{y}\ell=-\frac{1}{\rho_{0}^{2}}\int_{0}^{\lambda_{\rm p}/4}\sin^{2}k_{\rm p }z\,\mathrm{d}z=-\tfrac{1}{8}\frac{\lambda_{\rm p}}{\rho_{0}^{2}} \tag{7.75}\]where \(\ell\) is the effective length of the focusing element and \(k_{\rm p}=\frac{2\pi}{\lambda_{\rm p}}\). The integrated equivalent quadrupole strength or inverse focal length for each half pole is

\[k_{y}\ell-\frac{1}{f_{y}}=-\frac{1}{8}\left(\frac{eB_{0}}{p_{0}}\right)^{2} \lambda_{\rm p}=-\frac{\lambda_{\rm p}}{8\rho_{0}^{2}}. \tag{7.76}\]

For \(N\) wiggler poles we have \(2N\) times the focusing strength and the focal length of the total wiggler magnet of length \(L_{\rm w}=\frac{1}{2}N\lambda_{\rm p}\) expressed in units of the wiggler strength parameter \(K\) becomes

\[\frac{1}{f_{y}}=\frac{K^{2}}{2\gamma^{2}}k_{\rm p}^{2}L_{\rm w}. \tag{7.77}\]

Tacitly, a rectangular form of the wiggler poles has been assumed (Fig. 7.12) and consistent with our sign convention, we find that wiggler fringe fields cause focusing in the nondeflecting plane. Within the approximation used there is no corresponding focusing effect in the deflecting plane. This is the situation for most wiggler magnets or poles except for the first and last half pole where the beam enters the magnetic field normal to the pole face.

A reason to possibly use wiggler magnets with rotated pole faces like wedge magnets originates from the fact that the wiggler focusing is asymmetric and not part of the lattice focusing and may therefore need to be compensated. For moderately strong wiggler fields the asymmetric focusing in both planes can mostly be compensated by small adjustments of lattice quadrupoles. The focusing effect of strong wiggler magnets, however, may generate a significant perturbation of the lattice focusing structure or create a situation where no stable solution for betatron functions exists anymore. The severity of this problem can be reduced by designing the wiggler poles as wedge magnets in such a way as to split the focusing equally between both the horizontal and vertical plane. In this case local correction can be applied efficiently in nearby lattice quadrupoles.

We will therefore discuss the focusing and transformation matrix through a wiggler pole in the case of arbitrary entry and exit angles. To derive the complete and general transformation matrices, we note that the whole wiggler field can be

Figure 7.12: Wiggler magnet with parallel pole end faces

treated in the same way as the fringe field of ordinary magnets. The focal length of one half pole in the horizontal deflecting plane is from (7.60)

\[\frac{1}{f_{x}}=\int_{0}^{\lambda_{\rm p}/4}\kappa_{x}^{\prime}\eta\,{\rm d}z+ \kappa_{x0}\,\delta_{\rm f}\,, \tag{7.78}\]

where the pole face rotation angle \(\eta\) has been assumed to be small and of the order of the wiggler deflection angle per pole (Fig. 7.13). With \(\kappa_{x}=\kappa_{x0}\cos k_{\rm p}z\) the field slope is

\[\kappa_{x}^{\prime}=\kappa_{x0}k_{\rm p}\sin k_{\rm p}z \tag{7.79}\]

and after integration of (7.78), the focal length for the focusing of a wiggler half pole is

\[\frac{1}{f_{x}}=\kappa_{x0}\left(\delta_{\rm f}+\eta\right)\,, \tag{7.80}\]

where \(\delta_{\rm f}\) is given by (7.56) and in the case of a wiggler magnet is equal to the deflection angle of a half pole. In the case of a rectangular wiggler pole \(\eta=-\delta_{\rm f}\) and the focusing in the deflecting plane vanishes as we would expect. In the nondeflecting plane (7.53) applies and the focal length is for small angles \(\eta\) and \(\delta\)

\[\frac{1}{f_{y}}=-\int_{0}^{\lambda_{\rm p}/4}\kappa_{x}^{\prime}[\eta+\delta( \overline{z})]\,{\rm d}\overline{z}\,. \tag{7.81}\]

The focal length per wiggler half pole is after integration

\[\frac{1}{f_{y}}=-\kappa_{x0}(\eta+\delta_{\rm f})-\frac{\pi}{4}\kappa_{x0}\, \delta_{\rm f}\,. \tag{7.82}\]

Here again setting \(\eta=-\delta_{\rm f}\) restores the result obtained in (7.77).

The focusing in each single wiggler pole is rather weak and we may apply thin lens approximation to derive the transformation matrices. For this we consider the focusing to occur in the middle of each wiggler pole with drift spaces of length \(\lambda_{\rm p}/4\)

Figure 7.13: Wiggler magnet with wedge shaped poles

on each side. With \(2/f\) being the focal length of a full pole in either the horizontal plane (7.80) or vertical plane (7.82) the transformation matrix for each wiggler pole is finally

\[\mathcal{M}_{\text{pole}}= \left(\begin{array}{cc}1&\lambda_{\text{p}}/4\\ 0&1\end{array}\right)\left(\begin{array}{cc}1&0\\ -2/f&1\end{array}\right)\left(\begin{array}{cc}1&\lambda_{\text{p}}/4\\ 0&1\end{array}\right) \tag{7.83}\] \[= \left(\begin{array}{cc}1&-\frac{\lambda_{\text{p}}}{2f}&\frac{ \lambda_{\text{p}}}{f}\left(1-\frac{\lambda_{\text{p}}}{4f}\right)\\ -\frac{2}{f}&1-\frac{\lambda_{\text{p}}}{2f}\end{array}\right)\approx\left( \begin{array}{cc}1&\frac{1}{2}\lambda_{\text{p}}\\ -\frac{2}{f}&1\end{array}\right)\,,\]

where the approximation \(\lambda_{\text{p}}\ll f\) was used. For a wiggler magnet of length \(L_{\text{w}}=\frac{1}{2}N\lambda_{\text{p}}\), we have \(N\) poles and the total transformation matrix is

\[\mathcal{M}_{\text{wiggler}}=\mathcal{M}_{\text{pole}}^{N}\,. \tag{7.84}\]

This transformation matrix can be applied to each plane and any pole rotation angle \(\eta\). Specifically, we set \(\eta=-K/\gamma\) for a rectangular pole shape and \(\eta=0\) for pole rotations orthogonal to the path like in sector magnets.

#### Hard-Edge Model of Wiggler Magnets

Although the magnetic properties of wiggler magnets are well understood and easy to apply it is nonetheless often desirable to describe the effects of wiggler magnets in the form of hard-edge models. This is particularly true when numerical programs are to be used which do not include the feature of properly modeling a sinusoidal wiggler field. On the other hand accurate modeling is important since frequently strong wiggler magnets are to be inserted into a beam transport lattice.

For the proper modeling of linear wiggler magnet properties we choose three conditions to be fulfilled. The deflection angle for each pole should be the same as that for the equivalent hard-edge model. Similarly the edge focusing must be the same. Finally, like any other bending magnet in an electron circular accelerator, a wiggler magnet also contributes to quantum excitation and damping of the beam emittance and beam energy spread. The quantum excitation is in first approximation proportional to the third power of the curvature while the damping scales like the square of the curvature similar to focusing.

Considering now a wiggler field

\[B(z)=B_{0}\sin k_{\text{p}\mathbb{Z}}\,, \tag{7.85}\]

we try to model the field for a half pole with parallel endpoles by a hard-edge magnet. Three conditions should be met. The deflection angle of the hard-edge model of length \(\ell\) and field \(B\) must be the same as that for a wiggler half pole,or

\[\theta=\frac{\ell_{\rm h}}{\rho_{\rm h}}=\frac{e}{p_{0}}\int_{\rm halfpole}B_{y}(z )\,{\rm d}z=\frac{\lambda_{\rm p}}{2\pi\rho_{0}}. \tag{7.86}\]

Here we use \(\rho_{\rm h}\) for the bending radius of the equivalent hard-edge model and \(\rho_{0}\) for the bending radius at the peak wiggler field \(B_{0}\). The edge focusing condition can be expressed by

\[\frac{1}{f}=\frac{\ell_{\rm h}}{\rho_{\rm h}^{2}}=\frac{1}{\rho_{0}^{2}}\int_{ \rm halfpole}\sin^{2}k_{\rm p}z\,{\rm d}z=\frac{\lambda_{\rm p}}{8\rho_{0}^{2}}. \tag{7.87}\]

Modeling a wiggler field by a single hard-edge magnet requires in linear beam optics only two conditions to be met which can be done with the two parameters \(B(z)\) and \(\ell\) available. From (7.86), (7.87) we get therefore the hard-edge magnet parameters (Fig. 7.14)

\[\rho_{\rm h}=\frac{4}{\pi}\rho_{0}\qquad\mbox{ and }\qquad\ell_{\rm h}= \frac{2}{\pi^{2}}\lambda_{\rm p}. \tag{7.88}\]

For a perfect modeling of the equilibrium energy spread and emittance due to quantum excitation in electron storage rings we would also like the cubic term to be the same

\[\frac{\ell_{\rm h}}{\rho_{\rm h}^{3}}\stackrel{{?}}{{=}}\frac{1}{ \rho_{0}^{3}}\int_{\rm halfpole}\sin^{3}k_{\rm p}z\,{\rm d}z=\frac{\lambda_{ \rm p}}{3\pi\,\rho_{0}^{3}}. \tag{7.89}\]

Since we have no more free parameters available, we can at this point only estimate the mismatch. With (7.87), (7.88) we get from (7.89) the inequality

\[\frac{1}{3\pi}\neq\frac{\pi}{32} \tag{7.90}\]

which indicates that the quantum excitation from wiggler magnets is not correctly treated although the error is only about 8 %.

Figure 7.14: Hard edge model for a wiggler magnet period

Similarly, one could decide that the quadratic and cubic terms must be equal while the deflection angle is let free. This would be a reasonable assumption since the total deflection angle of a wiggler is compensated anyway. In this case the deflection angle would be underestimated by about 8 %. Where these mismatches are not significant, the simple hard-edge model (7.89) can be applied. For more accuracy the sinusoidal wiggler field must be segmented into smaller hard-edge magnets.

### Elements of Beam Dynamics

The most basic elements of a beam transport line are drift spaces, bending magnets and focusing magnets or quadrupoles. Obviously, in a drift space of length \(\ell\) the electric or magnetic field vanishes. Bending magnets act as beam guidance devices while quadrupoles will focus the beam. In the following section, we will discuss building blocks made up of bending magnets and quadrupoles, which exhibit features known from light optics thus justifying our extensive use of terminology from optics in particle beam dynamics.

#### Building Blocks for Beam Transport Lines

With special arrangements of bending and focusing magnets it is possible to construct lattice sections with particular properties. We may desire a lattice section with specific chromatic properties, achromatic or isochronous sections. In the next paragraphs we will discuss such lattice elements with special properties.

##### General Focusing Properties

The principal solutions and some elements of transformation matrices through an arbitrary beam transport line can reveal basic beam optical properties of this beam line. A close similarity to paraxial light optics is found in the matrix element \(C^{\prime}(z)\). As shown schematically in Fig. 7.15, parallel trajectories \(\left(u_{0}^{\prime}=0\right)\) are deflected by the focusing system through the matrix element \(C^{\prime}(z)\) and emerge with a slope \(u^{\prime}(z)=C^{\prime}(z)\,u_{0}\).

From light optics we know that \(-u_{0}/u^{\prime}(z)\) is defined as the focal length of the system. In analogy, we define therefore also a focal length \(f\) for a composite focusing system by setting

\[f^{-1}=C^{\prime}(z). \tag{7.91}\]

The focal point is defined by the condition \(u\left(z_{l}\right)=0\) and is, therefore, located where the cosine like solution becomes zero or \(C(z_{l})=0\).

More similarities with paraxial light optics can be identified. Point to point imaging, for example, is defined in particle beam optics by the sine like function \(S(z)\), starting at the object plane at \(z=z_{0}\). The image point is located where the sine-like function crosses again the reference axis or where \(S(z_{i}+z_{0})=0\) as shown in Fig. 7.16.

By definition such a section of a beam transport system has a betatron phase advance of \(180^{\circ}\). The beam size or object size \(H_{0}\) at \(z_{0}\) is transformed by the cosine like function to become at the image point \(H(z_{i})=|\,C(z_{i}+z_{0})|\,H_{0}\) and the magnification of the beam optical system is given by the absolute value of the cosine like function at the image point

\[M=|C(z_{i}+z_{0})|. \tag{7.92}\]

##### Chromatic Properties

Very basic features can be derived for the chromatic characteristics of a beam transport line. In (5.81), we have already derived the dispersion function

\[D(z)=S(z)\int_{0}^{z}\kappa\,(\overline{z})\,C(\overline{z})\,\mathrm{d} \overline{z}-C(z)\int_{0}^{z}\kappa\,(\overline{z})\,S(\overline{z})\,\mathrm{ d}\overline{z}\,. \tag{7.93}\]

Figure 7.16: Point to point imaging

Figure 7.15: Focusing in a quadrupole doublet

From this expression we conclude that there is dispersion only if at least one of the two integrals in (7.93) is nonzero. That means only dipole fields can cause a dispersion as a consequence of the linear chromatic perturbation term \(\kappa\delta\). All other perturbation terms in (6.95), (6.96) are of higher order in \(\delta\) or depend on the transverse particle coordinates and therefore contribute only to higher order corrections of the dispersion function.

Specifically, we find from (5.26) the lowest order chromatic quadrupole perturbation to be \(kx\delta\). Since any arbitrary particle trajectory is composed of an energy independent part \(x_{\beta}\) and an energy dependent part \(D\delta\), expressed by \(x=x_{\beta}\,+\,D\delta\), we find the lowest chromatic quadrupole perturbation to the dispersion function to be the second order term \(kD\delta^{2}\) which does not contribute to linear dispersion.

While some dispersion cannot be avoided in beam transport systems where dipole magnets are used, it is often desirable to remove this dispersion at least in some parts of the beam line. As a condition for that to happen at say \(z=z_{\rm d}\), we require that \(D(z_{\rm d})=0\). From ( 7.93) this can be achieved if

\[\frac{S(z_{\rm d})}{C(z_{\rm d})}\,=\,\frac{\int_{0}^{z_{\rm d}}\kappa\,\left( \overline{z}\right)S\left(\overline{z}\right)\,{\rm d}\overline{z}}{\int_{0}^ {z_{\rm d}}\kappa\,\left(\overline{z}\right)C\left(\overline{z}\right)\,{\rm d }\overline{z}}\,, \tag{7.94}\]

a condition that can be met by proper adjustments of the focusing structure.

#### Achromatic Lattices

A much more interesting case is the one, where we require both the dispersion and its derivative to vanish, \(D(z_{\rm d})\,=\,0\) and \(D^{\prime}(z_{\rm d})\,=\,0\). In this case we have no dispersion function downstream from the point \(z\,=\,z_{\rm d}\) up to the point, where the next dipole magnet creates a new dispersion function. The conditions for this to happen are

\[\begin{array}{l}D(z_{\rm d})=0=-S(z_{\rm d})\,I_{\rm c}+C(z_{\rm d})\,I_{\rm s }\,,\\ D^{\prime}(z_{\rm d})=0=-S^{\prime}(z_{\rm d})\,I_{\rm c}+\,C^{\prime}(z_{\rm d })\,I_{\rm s},\end{array} \tag{7.95}\]

where we have set \(I_{\rm c}=\int_{0}^{z_{\rm d}}\kappa\,C\,{\rm d}\overline{z}\) and \(I_{\rm s}=\int_{0}^{z_{\rm d}}\kappa\,S\,{\rm d}\overline{z}\). We can solve (7.95) for \(I_{\rm c}\) or \(I_{\rm s}\) and get

\[\begin{array}{l}\left[C(z_{\rm d})\,S^{\prime}(z_{\rm d})-S(z_{\rm d})\,C^{ \prime}(z_{\rm d})\right]\,I_{\rm c}=0\,,\\ \left[C(z_{\rm d})\,S^{\prime}(z_{\rm d})-S(z_{\rm d})\,C^{\prime}(z_{\rm d}) \right]\,I_{\rm s}=0\,.\end{array} \tag{7.96}\]

Since \(C(z_{\rm d})S^{\prime}(z_{\rm d})-S(z_{\rm d})C^{\prime}(z_{\rm d})\,=\,1\), the conditions for a vanishing dispersion function are

\[\begin{array}{l}I_{\rm c}=\int_{0}^{z_{\rm d}}\kappa\left(\overline{z}\right) C(\overline{z})\,{\rm d}\overline{z}=0\,,\\ I_{\rm s}=\int_{0}^{z_{\rm d}}\kappa\left(\overline{z}\right)S(\overline{z})\,{ \rm d}\overline{z}=0\,.\end{array} \tag{7.97}\]A beam line is called a first order achromat or short an achromat if and only if both conditions (7.97) are true. The physical characteristics of an achromatic beam line is that at the end of the beam line, the position and the slope of a particle trajectory is independent of the energy.

#### Isochronous Systems

For the accelerating process we will find that the knowledge of the path length is of great importance. The path length \(L\) of any arbitrary particle trajectory can be derived by integration to give

\[L=\int\mathrm{d}s=\int_{0}^{L_{0}}\frac{\mathrm{d}s}{\mathrm{d}\bar{z}}\, \mathrm{d}\bar{z}=\int_{0}^{L_{0}}\sqrt{x^{\prime\,2}+y^{\prime\,2}+\left(1+ \kappa_{x}x\right)^{2}}\,\mathrm{d}\bar{z}\,, \tag{7.98}\]

where \(L_{0}\) is the length of the beam line along the ideal reference path. For simplicity we have ignored a vertical deflection of the beam. The path length variation due to a vertical bend would be similar to that for a horizontal bend and can therefore be easily derived form this result. Since \(x^{\prime},y^{\prime}\) and \(\kappa_{x}x\) are all small compared to unity, we may expand the square root and get in keeping only second order terms

\[L=\int_{0}^{L_{0}}[1\,+\kappa_{x}\,x+\ \tfrac{1}{2}(x^{\prime\,2}+y^{\prime\,2}+ \kappa_{x}^{2}\,x^{2})]\,\mathrm{d}\bar{z}+\mathcal{O}(3)\,. \tag{7.99}\]

Utilizing (5.83) we get from (7.99) for the path length difference

\[(L-L_{0})_{\mathrm{sector}} =x_{0}\int_{0}^{L_{0}}\kappa_{x}(\bar{z})C(\bar{z})\,\mathrm{d} \bar{z}+x_{0}^{\prime}\int_{0}^{L_{0}}\kappa_{x}(\bar{z})S(\bar{z})\,\mathrm{d} \bar{z} \tag{7.100}\] \[\qquad+\delta\int_{0}^{L_{0}}\kappa_{x}(\bar{z})D(\bar{z})\, \mathrm{d}\bar{z}\,.\]

The variation of the path length has two contributions. For \(\delta=0\) the path length varies due to the curvilinear coordinate system, where dipole fields exist. This is a direct consequence of the coordinate system which selects a sector magnet as its natural bending magnet. The ideal path enters and exits this type of dipole magnet normal to its pole face as shown in Fig. 7.17. It becomes obvious from Fig. 7.17 that the path length difference depends on the particle position with respect to the reference path and is in linear approximation

\[\mathrm{d}\ell=\ell-\ell_{0}=(\rho_{0}+x)\,\mathrm{d}\varphi-\rho_{0}\, \mathrm{d}\varphi\,. \tag{7.101}\]Figure 7.18 displays the general situation for a wedge magnet with arbitrary entrance and exit pole face angles. The path length differs from that in a sector magnet on either end of the magnet. The first integral in (7.100) therefore must be modified to take into account the path length elements in the fringe field. For a wedge magnet we have therefore instead of (7.100)

\[(L-L_{0})_{\text{wedge}} =x_{0}\int_{0}^{L_{0}}\kappa_{x}(\widetilde{z})\;C(\widetilde{z}) \;\text{d}\widetilde{z} \tag{7.102}\] \[\qquad+[C(z_{0})x_{0}+\rho_{0}]\eta_{0}+[C(z_{\text{e}})x_{0}+\rho _{0}]\eta_{e}\] \[\qquad-x_{0}C(z_{0})\tan\eta_{0}-x_{0}\;C(z_{\text{e}})\tan\eta_{ \text{e}}\] \[\qquad+x_{0}^{\prime}\int_{0}^{L_{0}}\kappa_{x}(\widetilde{z})S( \widetilde{z})\;\text{d}\widetilde{z}+\delta\;\int_{0}^{L_{0}}\kappa_{x}( \widetilde{z})D(\widetilde{z})\;\text{d}\widetilde{z}\] \[\approx(L-L_{0})_{\text{sector}}+\mathcal{O}(2)\;.\]

Here \([C(z)\,x_{0}+\rho_{0}]\;\eta\) is the arc length through the wedge-like deviations from a sector magnet which must be compensated by the decrease or increase \(C(z)\,x_{0}\tan\eta\) in the adjacent drift space. For small edge angles both terms compensate well and the total path length of a wedge magnet is similar to that of a sector magnet. In general we therefore ignore path length variations in wedge magnets with respect to

Figure 7.17: Path length in a sector magnet

Figure 7.18: Path length in a wedge magnet

sector magnets as well as those in the adjacent drift spaces. For large edge angles, however, this assumption should be reconsidered.

Equation (7.100) imposes quite severe restrictions on the focusing system if the path length is required to be independent of initial condition and the energy. Since the parameters \(x_{0},x_{0}^{\prime}\) and \(\delta\) are independent parameters for different particles, all three integrals in (7.100) must vanish separately. An isochronous beam transport line must therefore be a first order achromat (7.97) with the additional condition that \(\int\kappa_{x}\,D\,\mathrm{d}\bar{z}=0\).

For highly relativistic particles (\(\beta\approx 1\)) and this condition is equivalent to being an isochronous beam line. In general, any beam line becomes isochronous if we require the time of flight rather than the path length to be equal for all particles. In this case we have to take into account the velocity of the particles as well as its variation with energy. The variation of the particle velocity with energy introduces in (7.100) an additional chromatic correction and the variation of the time of flight becomes

\[\beta c(T-T_{0})=x_{0}\,I_{\mathrm{c}}+x_{0}^{\prime}\,I_{\mathrm{s}}+\delta( I_{\mathrm{d}}-\gamma^{-2})\,. \tag{7.103}\]

In straight beam lines, where no bending magnets are involved, (7.103) vanishes and higher than linear terms must be considered. From (7.99) it is obvious that the bending independent terms are quadratic in nature and therefore isochronicity cannot be achieved exactly since

\[\beta c\,\Delta T=\int_{0}^{L_{0}}({x^{\prime}}^{2}+\,{y^{\prime}}^{2})\mathrm{ d}\bar{z}>0\,. \tag{7.104}\]

This integral is positive for any particle oscillating with a finite betatron amplitude. A straight beam transport line is therefore an isochronous transport system only in first order.

## Problems

### (S)

Sketch a quadrupole doublet and draw the sine- and cosine-like trajectories through the quadrupole doublet to the focal point for the horizontal and vertical plane. Verify that (7.20) is indeed true. (hint: first define from where to where you need to measure the combined focal length \(f\) ).

### (S)

Consider a thin quadrupole doublet with a drift space of 1 m between them. The quadrupole strengths are to be adjusted to make a focal point in both planes at a point 5 m from the second quadrupole. Determine the quadrupole strengths and calculate the combined doublet focal length in both planes. Sketch the doublet focusing and define in this sketch the calculated combined focal lengths.

**7.3 (S).** Consider a quadrupole doublet made of thin lenses. a) Calculate the focal length of a quadrupole doublet with \(|f_{1}|=|f_{2}|=5\,\)m and a distance between the magnets of \(d=1\,\)m. Plot for this doublet the focal length as a function of particle momentum \(-5\%<\Delta p/p<5\%\). b) Use a parallel beam of radius \(r_{0}\) and maximum divergence \(r^{\prime}\) and calculate the beam radius \(r\) at the focal point of this doublet. c) Plot the magnification \(r/r_{0}\) as a function of momentum \(-5\,\%<\Delta p/p<5\,\%\). What is the chromatic aberration \(\left(r-r_{0}\right)/r_{0}\) of the spot size?

**7.4 (S).** Sector and rectangular magnets have opposite focusing properties. Determine the geometry of a wedge magnet with equal focusing in both planes (ignore the gap effect).

**7.5 (S).** In an arbitrary open beam transport line, we assume that at the point \(z_{0}\) the particle beam is kicked in the horizontal or vertical plane by the deflection angle \(\vartheta\). What is the betatron amplitude for the beam at any point \(z\) downstream from \(z_{0}\)? To maximize the betatron amplitude at \(z\) how should the lattice functions, betatron function and/or phase, be chosen at \(z_{0}\) and \(z\)?

**7.6 (S).** Design a beam bump within three cells of a symmetric FODO lattice \(\frac{1}{2}\)QF\({}_{1}\)-QD\({}_{1}\)-QF\({}_{2}\)-QD\({}_{2}\)-QF\({}_{3}\)-QD\({}_{3}\)-\(\frac{1}{2}\)QF\({}_{4}\) with a betatron phase advance \(\psi_{\rm F}=90^{\circ}\) per cell. Further assume there are special coils in the quadrupoles to produce dipole fields which can be used to deflect the beam. a) Construct a symmetric beam bump which starts at QF\({}_{1}\), ends at QF\({}_{4}\) and reaches an amplitude \(A_{\rm M}=2\,\)cm in the center of QD\({}_{2}\). How many trim coils need to be activated? b) Derive the relative kick angles required to construct the beam bump and calculate the beam displacement in each quadrupole. Is \(A_{\rm M}\) the maximum amplitude of the beam bump? Why? Why not? (hint: do not use betatron and phase functions, but use thin lens approximation)

**7.7.** a) Design a symmetric thin lens triplet with a focal point for both planes at the same point \(z=z_{\rm f}\). b) Calculate and plot the betatron function for the quadrupole triplet and some drift space extending beyond the focal point. The value for the betatron function be \(\beta=8\,\)m at the entrance to the triplet \(z=0\) where we also assume \(\alpha=0\) c) Derive the phase advance in one plane between \(z=0\) and \(z=z_{\rm f}\) both from the elements of the transformation matrix and by integrating the betatron function. Both method should give the same results. (note: do the integration roughly from the drawing of the betatron function with linear interpolation).

## Bibliography

* (1) N. Christofilos, US Patent No 2,736,766 (1950)
* (2) E.D. Courant, M.S. Livingston, H.S. Snyder, Phys. Rev. **88**, 1190 (1952)
* (3) J. Moser, Stabilitatsverhalten kanonischer differentialgleichungssysteme. Nachr. der Akad. der Wiss. Gottingen **IIa**(6), 87 (1951)
* (4) E.D. Courant, H.S. Snyder, Appl. Phys. **3**, 1 (1959)
* (5) K.G. Steffen, _High Energy Beam Optics_ (Wiley, New York, 1965)
* (6) R.F.K. Herzog, Acta Phys. Austriaca **4**, 431 (1951)


