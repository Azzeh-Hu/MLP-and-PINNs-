

# Baseline 
1. FEM with least-squares calibration: the linear FEM is
fitted to the same sensor data by optimising $h_{eff}$ [?] 
2. direct interpolation:
   cubic-spline interpolation of the sensors along x at each time step, with no physics.
## FEM with phase aware calculations 
####  questions and notes
1. is NiTi isotropic?
2. ## 6.9 Case Study-XIV

**Phase–field modeling of multi-variant** **martensitic transformations**

**Objective**:

The objective of this case study is to demonstrate a FEM phase-field algorithm in which phase-field equations are fully coupled and solved simultaneously. Although, in this case, the fully coupled equations are elasticity and non-conserved order parameters; however, the approach serves as a template for fully coupled formulism of other phase-field models.

3. ## 6.7 Case Study-XII

**Phase-field modeling of** **grain growth** **with** **finite element method**

**Objectives**:

The objective of this case study is to develop a numerical implementation of the finite element method, FEM, for the solution of multicomponent non-conserved Allen–Cahn equations in phase-field modeling.

4. ## 6.6 Case Study-XI

**Simulation of spinodal decomposition of a binary alloy with finite element** **method**

**Objectives** :

The objective of this case study is to demonstrate a numerical implementation of the finite element method, FEM, for the solution of conserved Cahn–Hilliard equation in phase-field modeling.



# Data structure 
	# FEM file 
		lvl 0: linear 
		lvl 1: phase aware  - Apparentheat capacity c_eff(T),
		 equilibrium ξ(T)
		lvl 2: Coupled T–ξ with relaxation kinetics (above)
		 and separate heating and cooling branches
	export files per case per Current:
			1. Data csv - lvl
			2. Figures - lvl
			3. Sensor data with uniform placement for
			   for training
	#FenCsi code 
		 1.Serperate code script for each case
		 2.The exports should go to each file 
		 Each script should have a small intro of the main --
		 equation that it is being derived from and what is --
		 being simplified, introduced, removed, etc.. 

use the pre-existing scripts and data in the file to rebuild the new format in much more structured format 
