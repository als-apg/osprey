## Chapter 14 Beam Emittance and Lattice Design

The task of lattice design for proton and ion beams can be concentrated to a pure particle beam optics problem. Transverse as well as longitudinal emittances of such beams are constants of motion and therefore do not depend on the particular design of the beam transport or ring lattice. This situation is completely different for electron and positron beams in circular accelerators where the emission of synchrotron radiation determines the particle distribution in six-dimensional phase space. The magnitude and characteristics of synchrotron radiation effects can, however, be manipulated and influenced by an appropriate choice of lattice parameters. We will discuss optimization and scaling laws for the transverse beam emittance of electron or positron beams in circular accelerators.

Originally electron storage rings have been designed, optimized and constructed for the sole use as colliding beam facilities for high energy physics. The era of electron storage rings for experimentation at the very highest particle energies has, however, reached a serious limitation due to excessive energy losses into synchrotron radiation. Of course, such a limitation does not exist for proton and ion beams with particle energies up to the order of some tens of TeV's and storage rings are therefore still the most powerful and productive research tool in high-energy physics. At lower and medium-energies electron storage rings with specially high luminosity still serve as an important research tool in high energy physics to study more subtle phenomena which could not be detected on earlier storage rings with lower luminosity like \(\tau\)- and \(B\)-factories.

To overcome the energy limitation in electron colliding beam facilities, the idea of linear colliders which avoids energy losses into synchrotron radiation [1, 2] becomes increasingly attractive to reach ever higher center of mass energies for high-energy physics. Even though electron storage rings are displaced by this development as the central part of a colliding beam facility they play an important role for linear colliders in the form of damping rings to prepare very small emittance particle beams.

The single purpose of electron storage rings for high-energy physics has been replaced by a multitude of applications of synchrotron radiation from such rings in a large variety of basic and applied research disciplines. It is therefore appropriate to discuss specific design and optimization criteria for electron storage rings.

Synchrotron radiation sources have undergone significant transitions and modifications over past years. Originally, most experiments with synchrotron radiation were performed parasitically on high energy physics colliding beam storage rings. Much larger photon fluxes could be obtained from such sources compared to any other source available. The community of synchrotron radiation users grew rapidly and so did the variety of applications and fields. By the time the usefulness of storage rings for high energy physics was exhausted some of these facilities were turned over to the synchrotron radiation community as fully dedicated radiation sources. Those are called first generation synchrotron radiation sources. They were not optimized for minimum beam emittance and maximum photon beam brightness. Actually, the optimization for high energy physics called for a maximum beam emittance to maximize collision rates for elementary particle events. The radiation sources were mostly bending magnets although the development and use of insertion devices started in these rings. Typically, the beam emittance is in the 100s of nanometer.

As the synchrotron radiation community further grew, funds became available to construct dedicated radiation facilities. Generally, these rings were designed as bending magnet sources but with reduced beam emittance (\(\leq 100\,\)nm) to increase photon brightness. The design emittances were much smaller than those in first generation rings but still large by present day standards. The use of insertion devices did not significantly affect the storage ring designs yet. These rings are called second generation rings.

Third generation synchrotron radiation sources were and are being designed, constructed and operated now. These rings are specifically designed for insertion device radiation with minimum beam emittances below \(20\,\)nm down to \(0.5\,\)nm for maximum photon beam brightness. As such, they exhibit a large number of magnet-free insertion straight sections.

Finally, fourth generation synchrotron radiation sources are the latest development for synchrotron radiation sources. Such sources are based on linear accelerator technology and the principle of single pass FELs where a high energy and high quality electron beam passing through a long undulator produces coherent undulator radiation in the X-ray regime.

Whatever the applications, in most cases it is the beam emittance which will ultimately determine the usefulness of the storage ring design for a particular application. We will derive and discuss physics and scaling laws for the equilibrium beam emittance in storage rings while using basic phenomena and processes of accelerator physics as derived in previous sections.

### 14.1 Equilibrium Beam Emittance in Storage Rings

The equilibrium beam emittance in electron storage rings is determined by the counteracting effects of quantum excitation and damping as has been discussed earlier. Significant synchrotron radiation occurs only in bending magnets and the radiation from each bending magnet contributes independently to both quantum excitation and damping. The contribution of each bending magnet to the equilibrium beam emittance can be determined by calculating the average values for \(\left\langle\left|\kappa^{3}\right|\mathcal{H}\right\rangle\) and \(\left\langle\kappa^{2}\right\rangle\) by

\[\left\langle\left|\kappa\right|\,^{3}\mathcal{H}\right\rangle_{z}=\frac{1}{C} \int_{0}^{C}\left|\kappa^{3}\left(z\right)\right|\mathcal{H}(z)\,\mathrm{d}z\,, \tag{14.1}\]

where \(\mathcal{H}\) is defined by (11.52) and \(C\) is the circumference of the storage ring. Obviously, this integral receives contributions only where there is a finite bending radius and therefore the total integral is just the sum of individual integrals over each bending magnet.

#### FODO Lattice

We consider here briefly the FODO lattice because of its simplicity and its ability to give us a quick feeling for the scaling of beam emittance with lattice parameters. The beam emittance can be manipulated at design time by adjusting \(\left\langle\mathcal{H}\right\rangle\) to the desired value. To calculate the average value \(\left\langle\mathcal{H}\right\rangle\) in a FODO lattice is somewhat elaborate. Here, we are interested primarily in the scaling of the beam emittance with FODO lattice parameters. Recollecting the results for the symmetric solutions of the lattice functions in a FODO lattice (10.3), (10.5), (10.74) we notice the following scaling laws

\[\beta \propto L\,, \tag{14.2}\] \[\beta^{\prime} \propto L^{0}\,,\] (14.3) \[\eta \propto L^{2}/\rho\,,\] (14.4) \[\eta^{\prime} \propto L/\rho\,, \tag{14.5}\]

where \(L\) is the distance between the centers of adjacent quadrupoles. All three terms in the function \(\mathcal{H}(z)=\gamma(z)\,\eta^{2}+2\alpha(z)\,\eta\eta^{\prime}+\beta(z)\,\eta ^{\prime 2}\) scale in a similar fashion like

\[\left\{\mathcal{H}(z)\right\}=\left\{\frac{1}{L}\frac{L^{4}}{\rho};\;L^{0} \frac{L^{2}}{\rho}\frac{L}{\rho};\;L\frac{L^{2}}{\rho}\right\}\propto\frac{L^ {3}}{\rho^{2}} \tag{14.6}\]and the equilibrium emittance for a FODO lattice scales then like

\[\epsilon_{x}=C_{\rm q}\gamma^{2}\frac{\langle{\cal H}/\rho^{3}\rangle}{\langle{1} /{\rho^{2}}\rangle}\propto\gamma^{2}\frac{L^{3}}{\rho^{3}}\propto\gamma^{2} \Theta^{3}\,, \tag{14.7}\]

where \(\Theta=\ell_{\rm b}/\rho\) is the deflection angle in each bending magnet. The proportionality factor depends on the beam focusing. A minimum can be reached for a focal length of \(|f\,|\,\approx\,1.06\,L\) in each half-quadrupole resulting in a minimum beam emittance achievable in a FODO lattice given in practical units by

\[\epsilon({\rm rad}\,{\rm m})\,\approx\,10^{-11}E^{2}({\rm GeV})\;\Theta^{3}({ \rm deg}^{3})\,, \tag{14.8}\]

where \(\varphi=2\pi/N_{\rm M}\), \(N_{\rm M}\) the number of bending magnets in the ring and \(N_{\rm M}\,/\,2\) the total number of FODO cells in the ring. This result is significant because it exhibits a general scaling law of the beam emittance proportional to the square of the beam energy and the cube of the deflecting angle in each bending magnet, which is valid for all lattice types. The coefficients, though, vary for different lattices. While the beam energy is primarily driven by the desired photon spectrum, we find that high brightness photon beams from low emittance electron beams require a storage ring design composed of many lattice cells with a small deflection angle per magnet. Of course, there are some limits on how far one can go with this concept due to other limitations, not the least being size and cost of the ring which both grow with the number of lattice cells.

#### Minimum Beam Emittance

While the cubic dependence of the beam emittance on the bending angle is a significant design criterion, we discuss here a more detailed optimization strategy. The emittance is determined by the beam energy, the bending radius and the \({\cal H}\)-function. Generally, we have no choice on the beam energy which is mostly determined by the desired critical photon energy of bending magnet and insertion device radiation or cost. Similarly, the bending radius is defined by the ring geometry, desired spectrum etc. Interestingly, it is not the bending radius but rather the bending angle which influences the equilibrium beam emittance. The main process to minimize the beam emittance is to adjust the focusing such that the lattice functions in the bending magnets generate a minimum value for \(\langle{\cal H}_{\rm b}\rangle_{z}\). The equilibrium beam emittance (13.18) depends only on the lattice function \({\cal H}_{\rm b}(z)\) inside bending magnets. Independent of any lattice type, we may therefore consider this function only within bending magnets. For the purpose of this discussion we assume a regular periodic lattice, where all bending magnets are the same and all lattice functions within each bending magnet are the same. That allows us to concentrate our discussion just on one bending magnet. The contribution of any individual bending magnet to the beam emittance can be determined by calculation of the average value for

\[\left\langle\mathcal{H}_{\mathrm{b}}\right\rangle_{z}=\frac{1}{\ell_{\mathrm{b}}} \int_{0}^{\ell_{\mathrm{b}}}\mathcal{H}_{\mathrm{b}}(z)\,\mathrm{d}z\,, \tag{14.9}\]

where \(\ell_{\mathrm{b}}\) is the length of the bending magnet and the bending radius is assumed to be constant within a magnet. From here on, we ignore the index \(x\) since we assume a flat storage ring in the horizontal plane. All lattice functions are therefore to be taken in the horizontal plane.

In evaluating the integral (14.1) we must include all contributions. The emission of photons depends only on the bending radius regardless of whether the bending occurs in the horizontal or vertical plane. Since for the calculation of equilibrium beam emittances only the energy loss due to the emission of photons is relevant it does not matter in which direction the beam is bent. The effect of the emission of a photon on the particle trajectory, however, is different for both planes because dispersion functions are different resulting in a different quantum excitation factor \(\mathcal{H}\). For a correct evaluation of the equilibrium beam emittances in the horizontal and vertical plane (14.1) should be evaluated for both planes by determining \(\mathcal{H}_{x}\) and \(\mathcal{H}_{y}\) separately but including in both calculations all bending magnets in the storage ring.

The integral in (14.1) can be evaluated for each magnet if the values of the lattice functions at the beginning of the bending magnet are known. With these initial values the lattice functions at any point within the bending magnet can be calculated assuming a pure dipole magnet. With the definitions of parameters from Fig. 14.1, we find the following expressions for the lattice functions in a bending magnet where \(z\) is the distance from the entrance of the magnet

\[\begin{array}{l}\beta(z)=\,\beta_{0}-2\alpha_{0}z+\,\gamma_{0}z^{2},\\ \alpha(z)=\alpha_{0}-\gamma_{0}z,\\ \gamma(z)=\gamma_{0},\\ \eta(z)=\,\eta_{0}+\,\eta_{0}^{\prime}z+\rho\left(1-\cos\theta\right),\\ \eta^{\prime}(z)=\,\eta_{0}^{\prime}+\,\sin\theta.\end{array} \tag{14.10}\]

Figure 14.1: Lattice functions in a bending magnet

Here the deflection angle is \(\theta=z/\rho\) and \(\beta_{0},\alpha_{0},\gamma_{0},\eta_{0},\eta_{0}^{\prime}\) are the values of the lattice functions at the beginning of the magnet. Before we use these equations we assume lattices where \(\eta_{0}=\eta_{0}^{\prime}=0\). The consequences of this assumption will be discussed later. Inserting (14.10) into (14.1) we get for small deflection angles after integration over one dipole magnet

\[\langle\mathcal{H}_{\mathrm{b}}\rangle_{\mathrm{z}}=\tfrac{1}{3}\Theta^{2} \beta_{0}-\alpha_{0}\rho\tfrac{1}{4}\Theta^{3}+\gamma_{0}\rho^{2}\tfrac{1}{20} \Theta^{4}+\mathcal{O}(\Theta^{5}), \tag{14.11}\]

where we have assumed the bending radius to be constant within the length \(\ell_{\mathrm{b}}\) of the magnet. In a storage ring with dipole magnets of different strength, contributions from all magnets must be added to give the average quantum excitation term for the whole ring of length \(C\)

\[\left\langle\left|\kappa^{3}\right|\mathcal{H}_{\mathrm{b}}\right\rangle_{ \mathrm{z}}=\frac{1}{C}\sum_{i}\left\langle\left|\kappa^{3}\right|\mathcal{H} _{\mathrm{b},i}\right\rangle_{\mathrm{z}}\ell_{\mathrm{b},i}\,, \tag{14.12}\]

where we sum over all magnets \(i\) with length \(\ell_{\mathrm{b},i}\). In an isomagnetic ring the factor \(\langle\left|\kappa^{3}\right|\mathcal{H}_{\mathrm{b}}/\langle\kappa^{2} \rangle\rangle_{\mathrm{z}}\) becomes simply \(\left|\kappa\right|\langle\mathcal{H}_{\mathrm{b}}\rangle_{\mathrm{z}}\) and the equilibrium beam emittance is

\[\epsilon_{\mathrm{iso}}=C_{\mathrm{q}}\frac{\gamma^{2}}{J_{x}}\left|\kappa \right|\langle\mathcal{H}_{\mathrm{b}}\rangle_{\mathrm{z}}\,. \tag{14.13}\]

Inserting (14.11) into (14.13) we get for the beam emittance in the lowest order of approximation

\[\epsilon_{\mathrm{iso}}=C_{\mathrm{q}}\gamma^{2}\Theta^{3}\left[\frac{1}{3} \frac{\beta_{0}}{\ell_{\mathrm{b}}}-\frac{1}{4}\alpha_{0}+\frac{1}{20}\gamma_{ 0}\,\ell_{\mathrm{b}}\right]+\mathcal{O}(\Theta^{4})\,, \tag{14.14}\]

where \(\gamma_{0}=\gamma(z_{0})\) is one of the lattice functions not to be confused with the particle energy \(\gamma\).

Here we have assumed a separate function lattice where the damping partition number \(J_{x}=1\). For strong bending magnets or sector magnets this assumption is not always justified due to focusing in the bending magnets and the damping partition number should be corrected accordingly.

The result (14.14) shows clearly a cubic dependence of the beam emittance on the deflection angle \(\Theta\) of the bending magnets which is a general lattice property since we have not yet made any assumption on the lattice type yet. Equation (14.14) exhibits minima with respect to both \(\alpha_{0}\) and \(\beta_{0}\). We solve the derivation \(\partial\langle\mathcal{H}\rangle/\partial\alpha_{0}=0\) for \(\alpha_{0}\) and the derivative \(\partial\langle\mathcal{H}\rangle/\partial\beta_{0}=0\) for \(\beta_{0}\) and get the optimum values for the Twiss functions at the entrance to the bending magnet

\[\beta_{0,\mathrm{opt}}=\sqrt{\frac{12}{5}}\ell_{\mathrm{b}}\,, \tag{14.15a}\] \[\alpha_{0,\mathrm{opt}}=\sqrt{15} \tag{14.15b}\]and the minimum value for \(\left\langle\mathcal{H}\right\rangle\) is

\[\left\langle\mathcal{H}\right\rangle_{\text{min}}=\frac{\Theta^{3}\rho}{4\sqrt{1 5}}\,. \tag{14.16}\]

With this, the minimum obtainable beam emittance in any lattice is from (13.18)

\[\epsilon_{\text{dba,min}}\approx C_{\text{q}}\gamma^{2}\frac{\left\langle \mathcal{H}_{\text{b}}(z)/\rho^{3}\right\rangle z}{\left\langle 1/\rho^{2} \right\rangle_{z}}\approx C_{\text{q}}\gamma^{2}\frac{\Theta^{3}}{4\sqrt{15}}\,. \tag{14.17}\]

The results are very simple for small deflection angles but for angles larger than about \(30^{\circ}\) per bending magnet the error for \(\left\langle\mathcal{H}\right\rangle_{\text{min}}\) exceeds \(10\,\%\) and higher order terms must be included.

For simplicity, we assumed that the dispersion functions \(\eta_{0}=0\) and \(\eta_{0}=0\). This a desirable feature, because it means that the dispersion function is also zero in the insertion devices (ID) of a synchrotron radiation source. A finite dispersion function in IDs can lead to an undesirable increase of the beam emittance.

In summary it has been demonstrated that for certain optimum lattice functions in the bending magnets the equilibrium beam emittance becomes a minimum. No assumption about a particular lattice has been made. Another observation is that the beam emittance is proportional to the third power of the magnet deflection angle and proportional to the square of the beam energy. Therefore many small deflection magnets interspersed within quadrupoles should be used to achieve a small beam emittance. Low emittance storage rings, therefore, are characterized by many short magnet lattice cells.

This approach has been used for a number of third generation synchrotron light sources. However, soon it was apparent that modification of the dispersion function could produce even smaller beam emittance in spite of the effect of IDs. Only, as it became possible in recent years to reach sub-nm beam emittances with sufficient dynamic aperture did the choice of finite dispersions in the IDs become undesirable again.

### 14.2 Absolute Minimum Emittance

In the previous section we found conditions which lead to a minimum beam emittance in an isomagnetic ring

\[\epsilon_{x}== C_{\text{q}}\,\gamma^{2}\frac{1}{\rho\ell}\int_{-\frac{1}{2}\ell}^{ \frac{1}{2}\ell}\mathcal{H}(z)\,\text{d}z \tag{14.18}\]

The \(\mathcal{H}\)-function in (14.1) is a nonlinear function of \(z\) and therefore any asymmetry of the Twiss functions lead to larger values of the \(\mathcal{H}\)-integral. We may 


With this optimum \(\int\mathcal{H}_{1}(z)\,\mathrm{d}z=\,\frac{1}{8\sqrt{15}}\frac{\ell^{4}}{\rho^{3}}\) and the minimum beam emittance is

\[\epsilon_{x}=\,C_{\mathrm{q}}\,\gamma^{2}\frac{\Theta^{3}}{J_{x}}\frac{1}{8 \sqrt{15}}. \tag{14.21}\]

This result has been derived by L. Teng in [3] and immediately rejected as "absolute minimum but useless". This judgement was based on the realization that the dispersion function at either end of the bending magnet is not zero and must therefore be of finite value at the insertion straight section too. This is not good as discussed above because insertion devices will enhance the emittance where \(\eta\neq 0\) and will also lead to an increased effective emittance for the synchrotron radiation users. This becomes a serious problem for very small beam emittances as can be obtained now about 30 years after his note. However there is a way out.

If we cut one bending magnet in a cell into two pieces and install them as the first and last bending magnet we get a zero dispersion function for all straight sections without change of the beam emittance. There may be an arbitrary number of such bending magnets between those half-magnets and there are enough quadrupoles between the last bending magnet and the center of the straight section to match the horizontal betatron function to any desired value while the dispersion function is now zero in the IDs. The vertical betatron function does not contribute to the emittance and may be matched any way possible within reason. Within the unit cell we expect a periodic matching section between magnets. Incidentally, the same result can be obtained if we set \(\alpha_{0}=0\) in (14.11) and look again for the optimum \(\beta_{0}\). However, we must replace the total deflection angle by its half.

Just to be complete in this discussion we assume for a moment that \(\eta_{0}\neq 0\)

\[\int\mathcal{H}_{2}(z)\,\mathrm{d}z=\,\frac{\eta_{0}^{2}}{\beta_{0}}\ell+\, \frac{\beta_{0}}{12\rho^{2}}\ell^{3}+\,\frac{1}{320\beta_{0}\rho^{2}}\ell^{5} -\frac{\eta_{0}}{12\beta_{0}\rho}\ell^{3}.\]

From \(\frac{\partial}{\partial\beta_{0}}\int\mathcal{H}_{2}(z)\,\mathrm{d}z=0\) the optimum betatron function is

\[\frac{\beta_{0}^{2}}{\ell^{2}}=\,\frac{3}{80}+\frac{12\eta_{0}^{2}\rho^{2}}{ \ell^{4}}-\frac{\eta_{0}\rho}{\ell^{2}}.\]

Furthermore there is also an optimum dispersion function and evaluating \(\frac{\partial}{\partial\eta_{0}}\int\mathcal{H}_{2}(z)\,\mathrm{d}z=0\) we get an optimum dispersion function in the middle of the bending magnet of

\[\eta_{0}=\,\frac{\ell^{2}}{24\rho}=\,\frac{\ell}{24}\Theta\]for which \(\int\mathcal{H}_{2}(z)\,\mathrm{d}z=\frac{1}{12\sqrt{15}}\frac{\ell^{4}}{\rho^{3}}\) and the minimum beam emittance

\[\epsilon_{x}=C_{\mathrm{q}}\,\gamma^{2}\frac{\Theta^{3}}{J_{x}}\,\frac{1}{12 \sqrt{15}} \tag{14.22}\]

which is even smaller.

The reduction in emittance by a factor \(3/2\) looks desirable but now we have again a finite although small dispersion function in the long straight section. The dispersion function scales like the square root of the betatron functions and for a betatron phase of \(90^{\circ}\), for example, the dispersion function \(\eta^{*}\) in the middle of the straight section is \(\eta^{*}=\sqrt{\frac{\beta^{*}}{\beta_{0}}}\eta_{0}=\frac{\sqrt{\beta^{*}\ell }}{124}\sqrt{\frac{5}{3}}\). For the users the effective emittance \(\epsilon_{\mathrm{eff}}=\epsilon_{0}\sqrt{1+\frac{\eta^{*2}\delta^{2}}{ \epsilon_{0}\beta^{*}}}\), where the relative energy spread is \(\delta^{2}=C_{\mathrm{q}}\gamma^{2}\frac{\Theta^{3}}{J_{x}\rho}=\epsilon_{0} \frac{12\sqrt{15}}{\rho\Theta^{3}}\). Finally, the effective beam emittance is

\[\frac{\epsilon_{\mathrm{eff}}}{\epsilon_{0}}=\sqrt{1+\frac{5}{12}\frac{1}{ \Theta^{3}}}.\]

To keep the effective beam emittance close to the natural emittance the deflection angle in the bending magnets must be large. In other words, the effective beam emittance for finite values of the dispersion function in insertion devices is much larger for modern low emittance storage rings with small deflection angles per bending magnet. for an emittance increase of a factor \(\sqrt{2}\) the deflection angle per bending magnet must be \(\Theta>0.75\) or \(42.8^{\circ}\).

### Beam Emittance in Periodic Lattices

To achieve a small particle beam emittance a number of different basic magnet storage ring lattice units are available and in principle most any periodic lattice unit can be used to achieve as small a beam emittance as desired. More practical considerations, however, will limit the minimum beam emittance achievable in a lattice. While all lattice types to be discussed have been used in existing storage rings and work well at medium to large beam emittances, differences in the characteristics of particular lattice types become more apparent as the desired equilibrium beam emittance is pushed to very small values.

Of the large variety of magnet lattices that have been used in existing storage rings the most commonly used ones are based on the double bend achromat (DBA) and derivatives thereof. In the DBA lattice the straight sections are separated by two bending magnets forming an achromat. In more recent years this approach has been modified into a multi-bend achromat where several bending magnets form an achromat between the straight sections. This trend was stimulated by the desire to minimize the beam emittance ever more while utilizing the \(\Theta^{3}\) scaling. However,at the same time the ring circumference would grow equally because of the higher number of long insertion straight sections unless there are several bending magnets between straight sections, thus limiting circumference and costs.

##### The Double Bend Achromat Lattice (DBA)

The double bend achromat or DBA lattice is designed to make full use of the minimization of beam emittance by the proper choice of lattice functions as discussed earlier. In Fig. 14.2 the basic layout of this lattice is shown.

A set of two or three quadrupoles provides the matching of the lattice functions into the bending magnet to achieve the minimum beam emittance. The central part of the lattice between the bending magnets may consist of one or more quadrupoles and its only function is to focus the dispersion function such that it is matched again to zero at the end of the next bending magnet resulting necessarily in a phase advance from bending magnet to bending magnet of close to 180\({}^{\circ}\). This lattice type has been proposed first by Panofsky [4] and later by Chasman and Green [5] as an optimized lattice for a synchrotron radiation source. In Fig. 14.3 an example of a synchrotron light source based on this type of lattice is shown representing the solution of the design study for the European Synchrotron Radiation Facility ESRF [6].

Figure 14.2: Double bend achromat (DBA) lattice (schematic) first proposed by Panofsky [4]

Figure 14.3: European synchrotron radiation facility, ESRF [6] (one half of 16 superperiods). The lattice is asymmetric to provide a mostly parallel beam in one insertion and a small beam cross section in the other

The ideal minimum beam emittance (14.17) in this lattice type for small bending angles and an isomagnetic ring with \(J_{x}=1\) is

\[\epsilon_{\mbox{\tiny DBA}}\,=\,\frac{C_{\mbox{\scriptsize q}}}{4\,\sqrt{15}} \gamma^{2}\Theta^{3} \tag{14.23}\]

or in more practical units

\[\epsilon_{\mbox{\tiny DBA}}(\mbox{rad m})\,=\,5.036\times 10^{-13}E^{2}(\mbox{ GeV}^{2})\,\Theta^{3}(\mbox{deg}^{3})\,. \tag{14.24}\]

This lattice type can be very useful for synchrotron light sources where many component and dispersion free straight sections are required for the installation of insertion devices. For damping rings this lattice type is not quite optimum since it is a rather "open" lattice with a low bending magnet fill factor and consequently a long damping time. Other more compact lattice types must be pursued to achieve in addition to a small beam emittance also a short damping time.

#### The FODO Lattice

The FODO lattice, shown schematically in Fig. 14.4 is the most commonly used and best understood lattice in storage rings optimized for high-energy physics colliding beam facilities where large beam emittances are desired. This choice is obvious considering that the highest beam energies can be achieved while maximizing the fill factor of the ring with bending magnets.

This lattice provides the most space for bending magnets compared to other lattices. The usefulness of the FODO lattice, however, is not only limited to high-energy large emittance storage rings. By using very short cells very low beam emittances can be achieved as has been demonstrated in the first low emittance storage ring designed [7] and constructed [8] as a damping ring for the linear collider SLC to reach an emittance of \(11\times 10^{-9}\,\)m at 1 GeV.

The lattice functions in a FODO structure have been derived and discussed in detail and are generally determined by the focusing parameters of the quadrupoles.

Figure 14.4: FODO lattice (schematic)

Since FODO cells are not achromatic the dispersion function is in general not zero at either end of the bending magnets.

The beam emittance can be derived analytically in thin lens approximation by integrating the quantum excitation factor along the bending magnets. The result is shown in Fig. 14.5 where the function \([\langle\mathcal{H}\rangle/(\rho\Theta^{3})]\) is plotted as a function of the betatron phase advance per FODO half cell which is determined by the focal length of the quadrupoles.

The beam emittance for an isomagnetic FODO lattice is given by [9]

\[\epsilon_{\text{\tiny FODO}}\,=\,C_{\text{q}}\gamma^{2}\Theta^{3}\frac{\ell_{ \text{b}}}{\ell_{\text{b},0}}\frac{\langle\mathcal{H}\rangle}{\rho\,\Theta^{3 }}\,, \tag{14.25}\]

where \(\ell_{\text{b},0}\) is the actual effective length of one bending magnet and \(2\ell_{\text{b}}\) the length of a FODO cell. From Fig. 14.5 it becomes apparent that the minimum beam emittance is reached for a betatron phase of about \(136.8^{\circ}\) per FODO cell. In this case \(\langle\mathcal{H}\rangle/(\rho\,\Theta^{3})\approx 1.25\) and the minimum beam emittance in such a FODO lattice in practical units is

\[\epsilon_{\text{\tiny FODO}}(\text{rad}\,\text{m})\,=\,97.53\times 10^{-13} \frac{\ell_{\text{b}}}{\ell_{\text{b},0}}E^{2}(\text{GeV}^{2})\,\Theta^{3}( \text{deg}^{3})\,. \tag{14.26}\]

Comparing the minimum beam emittance achievable in various lattice types the FODO lattice seems to be the least appropriate lattice to achieve small beam emittances. This, however, is only an analytical distinction. FODO cells can be made much shorter than the lattice units of other structures and for a given circumference many more FODO cells can be incorporated than for any other lattice. As a consequence, the deflection angles per FODO cell can be much smaller. For very

Figure 14.5: Electron beam emittance of a FODO lattice as a function of the betatron phase advance per half cell in the deflecting plane

low emittance storage ring, therefore, it is not a priori obvious that one lattice is better than another. However, additional requirements like number of desired insertion straight sections for a particular application must be included in the decision for the optimum storage ring lattice.

#### Optimum Emittance for Colliding Beam Storage Rings

The single most important parameter of colliding beam storage rings is the luminosity and most of the design effort for such facilities is aimed at maximizing the collision rate. As a consequence of the beam-beam effect, the beam emittance must be chosen to be as large as possible for maximum luminosity as will be discussed in Sect. 2.2.2. Since for most high energy storage rings a FODO lattice is employed it is clear that for maximum emittance the phase advance per cell should be kept low as indicated in Fig. 14.5. Of course, there is a practical limit given by increasing magnet apertures and associated costs.

In linear colliders there is no beam stability concern due to the beam-beam effect like in a storage ring and a much smaller beam cross section can be chosen. The limit here is the total beam-beam disruption due to the large electromagnetic fields at the surface of the colliding beams. Strong synchrotron radiation introduce, for example, significant energy losses which jeopardize the analysis of high energy physics events.

## Problems

**14.1 (S).** Derive an approximate expression of the beam emittance in an isomagnetic FODO lattice as a function of phase per cell and determine the minimum value of the emittance. Use a lattice which is symmetric in both planes and assume that the bending magnets are as long as the half cells (\(\ell_{\mathrm{b}}=L\)).

**14.2 (S).** Consider a storage ring made of FODO cells at an energy of your choice. How many bending magnets or half cells do you need to reach a beam emittance of no more than \(\epsilon_{x}=5\cdot 10^{-9}\,\mathrm{m}\)?

## Bibliography

* (1) M. Tigner. Nuovo Cimento **37**, 1228 (1965)
* (2) J.E. Augustin, N. Dikanski, Y. Derbenev, J. Rees, B. Richter, A. Skrinski, M. Tigner, H. Wiedemann, in _Proceedings of the Workshop on Possibilities and Limitations of Accelerators and Detectors_ (1979), p. 87
* (3) L. Teng, Technical Report, TM-1269, Fermi Nat. Lab., Chicago (1984)* [4] K.G. Steffen, _High Energy Beam Optics_ (Wiley, New York, 1965), p. 117
* [5] R. Chasman, K. Green, E. Rowe, IEEE Trans. Nucl. Sci. **22**, 1765 (1975)
* [6] Design report of the european synchrotron radiation facility (esrf). Technical report, Grenoble (1985)
* [7] H. Wiedemann, Scaling of damping rings for colliding linac beam systems. in _11th International Conference on High Energy Accelerators_ (CERN/Birkhauser, Geneva/Basel, 1980), p 693
* [8] G.E. Fischer, W. Davis-White, T. Figuth, H. Wiedemann, in _Proceedings of 12th International Conference High Energy Accelerators_ (Fermilab, Chicago, 1983)
* [9] R. Helm, H. Wiedemann, Technical Report, PEP-Note 303, Stanford Linear Accelerator Center, Stanford (1979)

