import numpy as np 
# import pyamg #TODO: later add it to optional dependencies,like sisl
# from pyamg.gallery import poisson
from dpnegf.utils.constants import elementary_charge
from dpnegf.utils.constants import Boltzmann, eV2J
from scipy.constants import epsilon_0 as eps0  #TODO:later add to untils.constants.py
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import logging
#eps0 = 8.854187817e-12 # in the unit of F/m
# As length in deeptb is in the unit of Angstrom, the unit of eps0 is F/Angstrom
eps0 = eps0*1e-10 # in the unit of F/Angstrom

log = logging.getLogger(__name__)

class Grid(object):
    """
    Represents a 3D grid structure for spatial discretization.
    Parameters
    ----------
    xg : array_like
        1D array of grid point coordinates along the x-axis.
    yg : array_like
        1D array of grid point coordinates along the y-axis.
    zg : array_like
        1D array of grid point coordinates along the z-axis.
    xa : array_like
        1D array of atom coordinates along the x-axis. Atoms must be within the grid bounds.
    ya : array_like
        1D array of atom coordinates along the y-axis. Atoms must be within the grid bounds.
    za : array_like
        1D array of atom coordinates along the z-axis. Atoms must be within the grid bounds.
    Attributes
    ----------
    xg, yg, zg : ndarray
        Grid coordinates along x, y, z axes.
    xall, yall, zall : ndarray
        Unique coordinates of all grid and atom positions along each axis.
    shape : tuple
        Shape of the grid as (len(xall), len(yall), len(zall)).
    grid_coord : ndarray
        Array of shape (Np, 3) containing the coordinates of all grid points, sorted lexicographically.
    Np : int
        Total number of grid points.
    Na : int
        Number of atoms.
    atom_index_dict : dict
        Dictionary mapping atom indices to their corresponding grid point indices.
    surface_grid : ndarray
        Array of shape (Np, 3) containing the surface area of each grid point along x, y, z axes.
    """
    # define the grid in 3D space
    def __init__(self,xg,yg,zg,xa,ya,za):
        # xg,yg,zg are the coordinates of the basic grid points
        self.xg = xg
        self.yg = yg
        self.zg = zg
        # xa,ya,za are the coordinates of the atoms
        # atom should be within the grid
        assert np.min(xa) >= np.min(xg) and np.max(xa) <= np.max(xg)
        assert np.min(ya) >= np.min(yg) and np.max(ya) <= np.max(yg)
        assert np.min(za) >= np.min(zg) and np.max(za) <= np.max(zg)

        self.Na = len(xa) # number of atoms
        uxa = np.unique(xa).round(decimals=6);uya = np.unique(ya).round(decimals=6);uza = np.unique(za).round(decimals=6)
        # x,y,z are the coordinates of the grid points
        self.xall = np.unique(np.concatenate((uxa,self.xg),0).round(decimals=3)) # unique results are sorted
        self.yall = np.unique(np.concatenate((uya,self.yg),0).round(decimals=3))
        self.zall = np.unique(np.concatenate((uza,self.zg),0).round(decimals=3))
        self.shape = (len(self.xall),len(self.yall),len(self.zall))

        

        # create meshgrid
        xmesh,ymesh,zmesh = np.meshgrid(self.xall,self.yall,self.zall)
        xmesh = xmesh.flatten()
        ymesh = ymesh.flatten()
        zmesh = zmesh.flatten()
        self.grid_coord = np.array([xmesh,ymesh,zmesh]).T #(Np,3)
        sorted_indices = np.lexsort((xmesh,ymesh,zmesh))
        self.grid_coord = self.grid_coord[sorted_indices] # sort the grid points firstly along x, then y, lastly z        
        ## check the number of grid points
        self.Np = int(len(self.xall)*len(self.yall)*len(self.zall))
        assert self.Np == len(xmesh)
        assert self.grid_coord.shape[0] == self.Np
        
        log.info(msg="Number of grid points: {:.1f}   Number of atoms: {:.1f}".format(float(self.Np),self.Na))
        # print('Number of grid points: ',self.Np,' grid shape: ',self.grid_coord.shape,' Number of atoms: ',self.Na)

        # find the index of the atoms in the grid
        self.atom_index_dict = self.get_atom_index(xa,ya,za)


        # create surface area for each grid point along x,y,z axis
        # each grid point corresponds to a Voronoi cell(box)
        surface_grid = np.zeros((self.Np,3))
        x_vorlen = self.cal_vorlen(self.xall);y_vorlen = self.cal_vorlen(self.yall);z_vorlen = self.cal_vorlen(self.zall)
        
        XD,YD = np.meshgrid(x_vorlen,y_vorlen)
        ## surface along x-axis (yz-plane)
        ax,bx = np.meshgrid(YD.flatten(),z_vorlen)
        surface_grid[:,0] = abs((ax*bx).flatten())
        ## surface along y-axis (xz-plane) 
        ay,by = np.meshgrid(XD.flatten(),z_vorlen)
        surface_grid[:,1] = abs((ay*by).flatten())
        ## surface along z-axis (xy-plane)
        az,_ = np.meshgrid((XD*YD).flatten(),self.zall)
        surface_grid[:,2] = abs(az.flatten())

        self.surface_grid = surface_grid  # grid points order are the same as that of  self.grid_coord
        

    def get_atom_index(self,xa,ya,za):
        """
        Finds the indices of atoms in the grid based on their coordinates.
        Parameters
        ----------
            xa (array-like): Array of x-coordinates of atoms.
            ya (array-like): Array of y-coordinates of atoms.
            za (array-like): Array of z-coordinates of atoms.
        Returns
        ---------
            dict: A dictionary where keys are atom indices and values are the corresponding
                  grid point indices in `self.grid_coord` that match the atom positions.
        """
        # find the index of the atoms in the grid
        swap = {}
        for atom_index in range(self.Na):
            for gp_index in range(self.Np):
                if abs(xa[atom_index]-self.grid_coord[gp_index][0])<1e-3 and \
                   abs(ya[atom_index]-self.grid_coord[gp_index][1])<1e-3 and \
                   abs(za[atom_index]-self.grid_coord[gp_index][2])<1e-3:
                    swap.update({atom_index:gp_index})
        return swap
    
    def cal_vorlen(self,x):
        """
        Compute the length of the Voronoi segment for each point in a one-dimensional array.

        For each point in the input array `x`, this function calculates the length of the Voronoi segment,
        which is defined as half the distance to the neighboring points. The endpoints are handled by taking
        half the distance to their single neighbor.

        Parameters
        ----------
        x : array_like
            One-dimensional array of points (must be indexable and support len()).

        Returns
        -------
        xd : numpy.ndarray
            Array of the same length as `x`, where each element represents the Voronoi segment length
            corresponding to each point in `x`.
        """
        # compute the length of the Voronoi segment of a one-dimensional array x
        xd = np.zeros(len(x))
        xd[0] = abs(x[0]-x[1])/2
        xd[-1] = abs(x[-1]-x[-2])/2
        for i in range(1,len(x)-1):
            xd[i] = (abs(x[i]-x[i-1])+abs(x[i]-x[i+1]))/2
        return xd


class region(object):
    """
    A class representing a 3D rectangular region defined by ranges along the x, y, and z axes.
    parameters
    ----------
    x_range : tuple or list
        A sequence of two values specifying the minimum and maximum of the x-axis.
    y_range : tuple or list
        A sequence of two values specifying the minimum and maximum of the y-axis.
    z_range : tuple or list
        A sequence of two values specifying the minimum and maximum of the z-axis.
    Attributes
    ----------
        xmin (float): Minimum value of the x-axis range.
        xmax (float): Maximum value of the x-axis range.
        ymin (float): Minimum value of the y-axis range.
        ymax (float): Maximum value of the y-axis range.
        zmin (float): Minimum value of the z-axis range.
        zmax (float): Maximum value of the z-axis range.
    """
    def __init__(self,x_range,y_range,z_range):
        self.xmin,self.xmax = float(x_range[0]),float(x_range[1])
        self.ymin,self.ymax = float(y_range[0]),float(y_range[1])
        self.zmin,self.zmax = float(z_range[0]),float(z_range[1])
    
class Dirichlet(region):
    """
    Dirichlet boundary condition class for defining regions with fixed potential.

    Inherits from the `region` class and defines a region in 3D space
    with given x, y, and z ranges. The Dirichlet boundary condition is specified
    by a Fermi level (`Ef`) which represents the potential at the boundary.
    Parameters
    ----------
    x_range : tuple or list
        The lower and upper bounds (min, max) for the x-coordinate range.
    y_range : tuple or list
        The lower and upper bounds (min, max) for the y-coordinate range.
    z_range : tuple or list
        The lower and upper bounds (min, max) for the z-coordinate range.
    Attributes
    ----------
    Ef : float
        Fermi level of the gate (in unit eV), representing the fixed potential at the boundary.
    """
    def __init__(self,x_range,y_range,z_range):
        # Dirichlet boundary condition
        super().__init__(x_range,y_range,z_range)
        # Fermi_level of gate (in unit eV)
        self.Ef = 0.0        


class Dielectric(region):
    """
    Represents a dielectric region with a specified permittivity.

    Inherits from the `region` class and defines a region in 3D space
    with given x, y, and z ranges. The dielectric permittivity (`eps`)
    is initialized to 1.0 by default.
    Parameters
    ----------
    x_range : tuple or list
        The lower and upper bounds (min, max) for the x-coordinate range.
    y_range : tuple or list
        The lower and upper bounds (min, max) for the y-coordinate range.
    z_range : tuple or list
        The lower and upper bounds (min, max) for the z-coordinate range.
    Attributes
    ----------
    eps : float
        Dielectric permittivity of the region, default is 1.0.
    """
    def __init__(self,x_range,y_range,z_range):
        # dielectric region
        super().__init__(x_range,y_range,z_range)
        # dielectric permittivity
        self.eps = 1.0




class Interface3D(object):
    """
    Interface3D(grid, Dirichlet_group, dielectric_group)
    A class to handle the initialization and solution of the 3D Poisson equation
    on a structured grid with support for Dirichlet and dielectric regions.
    Parameters
    ----------
    grid : Grid
        An instance of the Grid class defining the spatial discretization.
    Dirichlet_group : list of Dirichlet
        List of Dirichlet region objects specifying boundary conditions.
    dielectric_group : list of Dielectric
        List of Dielectric region objects specifying spatially varying permittivity.
    Attributes
    ----------
    Dirichlet_group : list
        List of Dirichlet region objects.
    dielectric_group : list
        List of Dielectric region objects.
    grid : Grid
        The grid object used for discretization.
    eps : np.ndarray
        Dielectric permittivity at each grid point.
    phi : np.ndarray
        Electrostatic potential at each grid point.
    phi_old : np.ndarray
        Previous iteration's electrostatic potential.
    free_charge : np.ndarray
        Free charge density at each grid point.
    fixed_charge : np.ndarray
        Fixed charge density at each grid point.
    Temperature : float
        Temperature in Kelvin.
    kBT : float
        Thermal energy in eV.
    boudnary_points : dict
        Dictionary mapping grid point indices to boundary type or "in" for internal.
    lead_gate_potential : np.ndarray
        Potential values for lead/gate Dirichlet regions.
    internal_NP : int
        Number of internal (non-boundary) grid points.
    """
    def __init__(self,grid,Dirichlet_group,dielectric_group,eps_average_mode:str='harmonic'):
        """
        Initializes the Poisson solver with the given grid, Dirichlet boundary regions, and dielectric regions.
        Parameters:
            grid (Grid): The computational grid object. Must be an instance of the Grid class.
            Dirichlet_group (list of Dirichlet): List of Dirichlet boundary region objects. Each must be an instance of the Dirichlet class.
            dielectric_group (list of Dielectric): List of dielectric region objects. Each must be an instance of the Dielectric class.
        Attributes:
            Dirichlet_group (list): Stores the Dirichlet boundary regions.
            dielectric_group (list): Stores the dielectric regions.
            grid (Grid): The computational grid.
            eps (np.ndarray): Dielectric permittivity array, initialized to ones.
            phi (np.ndarray): Potential array, initialized to zeros.
            phi_old (np.ndarray): Previous potential array, initialized to zeros.
            free_charge (np.ndarray): Free charge density array, initialized to zeros.
            fixed_charge (np.ndarray): Fixed charge density array, initialized to zeros.
            Temperature (float): Temperature in Kelvin, default is 300.0.
            kBT (float): Thermal energy in eV.
            boudnary_points (dict): Dictionary mapping grid point indices to boundary status ("in" or boundary type).
            lead_gate_potential (np.ndarray): Lead or gate potential array, initialized to zeros.
        """
        assert grid.__class__.__name__ == 'Grid'

        
        for i in range(0,len(Dirichlet_group)):
            if not Dirichlet_group[i].__class__.__name__ == 'Dirichlet':
                raise ValueError('Unknown region type in Gate list: ',Dirichlet_group[i].__class__.__name__)
        for i in range(0,len(dielectric_group)):
            if not dielectric_group[i].__class__.__name__ == 'Dielectric':
                raise ValueError('Unknown region type in Dielectric list: ',dielectric_group[i].__class__.__name__)
            
        self.Dirichlet_group = Dirichlet_group
        self.dielectric_group = dielectric_group
        self.grid = grid
        self.eps = np.ones(grid.Np) # dielectric permittivity
        self.phi,self.phi_old = np.zeros(grid.Np),np.zeros(grid.Np) # potential
        self.free_charge,self.fixed_charge  = np.zeros(grid.Np),np.zeros(grid.Np)  # free charge density and fixed charge density 

        self.Temperature = 300.0 # temperature in unit of Kelvin
        self.kBT = Boltzmann*self.Temperature/eV2J # thermal energy in unit of eV

        # store the boundary information: xmin,xmax,ymin,ymax,zmin,zmax,gate
        self.boudnary_points = {i:"in" for i in range(self.grid.Np)} # initially set all points as internal
        self.get_boundary_points()

        self.lead_gate_potential = np.zeros(grid.Np) # no lead or gate potential initially, all grid points are set to zero
        self.average_mode = eps_average_mode
        

    def get_fixed_charge(self,x_range,y_range,z_range,molar_fraction,atom_gridpoint_index):
        """
        Sets the fixed charge density for grid points within the specified spatial ranges and atom indices.

        Parameters:
            x_range (tuple or list): The lower and upper bounds (min, max) for the x-coordinate range.
            y_range (tuple or list): The lower and upper bounds (min, max) for the y-coordinate range.
            z_range (tuple or list): The lower and upper bounds (min, max) for the z-coordinate range.
            molar_fraction (float): The value to assign as the fixed charge density for the selected grid points.
            atom_gridpoint_index (array-like): Indices of grid points corresponding to atom positions.

        Modifies:
            self.fixed_charge (np.ndarray): Updates the fixed charge density at the selected grid points.
        """
        # set the fixed charge density
        mask = (
            (float(x_range[0]) <= self.grid.grid_coord[:, 0]) &
            (float(x_range[1]) >= self.grid.grid_coord[:, 0]) &
            (float(y_range[0]) <= self.grid.grid_coord[:, 1]) &
            (float(y_range[1]) >= self.grid.grid_coord[:, 1]) &
            (float(z_range[0]) <= self.grid.grid_coord[:, 2]) &
            (float(z_range[1]) >= self.grid.grid_coord[:, 2])
        )
        index = np.nonzero(mask)[0]
        valid_indices = index[np.isin(index, atom_gridpoint_index)]
        self.fixed_charge[valid_indices] = molar_fraction



    def get_boundary_points(self):
        """
        Identifies and labels the boundary points of the grid in the x, y, and z directions.
        For each point in the grid, this method checks if the point lies on the minimum or maximum
        boundary along the x, y, or z axes.
        It assigns a corresponding label ("xmin", "xmax", "ymin", "ymax", "zmin", or "zmax") to 
        the `self.boudnary_points` array for boundary points. Points that do not lie on any boundary 
        are counted as internal points.
        Updates:
            - self.boudnary_points: Array with boundary labels for each grid point.
            - self.internal_NP: Number of internal (non-boundary) grid points.
        """

        # set the boundary points
        xmin,xmax = np.min(self.grid.xall),np.max(self.grid.xall)
        ymin,ymax = np.min(self.grid.yall),np.max(self.grid.yall)
        zmin,zmax = np.min(self.grid.zall),np.max(self.grid.zall)
        internal_NP = 0
        for i in range(self.grid.Np):
            if self.grid.grid_coord[i,0] == xmin: self.boudnary_points[i] = "xmin"
            elif self.grid.grid_coord[i,0] == xmax: self.boudnary_points[i] = "xmax"
            elif self.grid.grid_coord[i,1] == ymin: self.boudnary_points[i] = "ymin"   
            elif self.grid.grid_coord[i,1] == ymax: self.boudnary_points[i] = "ymax" 
            elif self.grid.grid_coord[i,2] == zmin: self.boudnary_points[i] = "zmin"  
            elif self.grid.grid_coord[i,2] == zmax: self.boudnary_points[i] = "zmax"
            else: internal_NP +=1
                
        self.internal_NP = internal_NP
    

    def get_potential_eps(self, region_list):
        """
        Assigns potential values and dielectric permittivity to grid points based on the provided region list.
        For each region in `region_list`, this method:
            - Identifies the grid points that fall within the spatial boundaries of the region.
            - If the region is of type 'Dirichlet', assigns the corresponding potential to those grid points and marks them as Dirichlet boundary points.
            - If the region is of type 'Dielectric', assigns the region's dielectric permittivity to those grid points.
            - Raises a ValueError if the region type is unknown.
        Parameters
        ----------
        region_list : list
            List of region objects, each with attributes defining spatial boundaries (xmin, xmax, ymin, ymax, zmin, zmax)
            and either a potential (Ef) for Dirichlet regions or permittivity (eps) for Dielectric regions.
        Raises
        ------
        ValueError
            If a region in the list has an unknown type.
        Logs
        ----
        The number of Dirichlet points assigned.
        """
        # assign the potential of Dirichlet region and dielectric permittivity to the grid points
        Dirichlet_point = 0
        for i in range(len(region_list)):    
            # find gate region in grid
            index=np.nonzero((region_list[i].xmin<=self.grid.grid_coord[:,0])&
                             (region_list[i].xmax>=self.grid.grid_coord[:,0])&
                        (region_list[i].ymin<=self.grid.grid_coord[:,1])&
                        (region_list[i].ymax>=self.grid.grid_coord[:,1])&
                        (region_list[i].zmin<=self.grid.grid_coord[:,2])&
                        (region_list[i].zmax>=self.grid.grid_coord[:,2]))[0]
            if region_list[i].__class__.__name__ == 'Dirichlet': 
                #attribute potential of Dirichlet region(lead and gate) to the corresponding grid points
                self.boudnary_points.update({index[i]: "Dirichlet" for i in range(len(index))})
                self.lead_gate_potential[index] = -1*region_list[i].Ef 
                Dirichlet_point += len(index)
            elif region_list[i].__class__.__name__ == 'Dielectric':
                # attribute dielectric permittivity to the corresponding grid points
                self.eps[index] = region_list[i].eps
            else:
                raise ValueError('Unknown region type: ',region_list[i].__class__.__name__)
        
        log.info(msg="Number of Dirichlet points: {:.1f}".format(float(Dirichlet_point)))
        
        
    def to_pyamg_Jac_B(self,dtype=np.float64):
        """
        Converts the current object's data into a Jacobian matrix and right-hand side vector suitable for use with PyAMG solvers.

        Parameters:
            dtype (data-type, optional): The desired data-type for the arrays. Default is numpy.float64.

        Returns:
            tuple:
                - Jacobian (scipy.sparse.csr_matrix): The constructed Jacobian matrix in CSR (Compressed Sparse Row) format.
                - B (numpy.ndarray): The right-hand side vector corresponding to the Jacobian matrix.

        Notes:
            This method initializes a zero Jacobian matrix and right-hand side vector, constructs their values using
            the `NR_construct_Jac_B` method, and returns them in formats compatible with PyAMG.
        """
        # convert to amg format A,b matrix
        # A = poisson(self.grid.shape,format='csr',dtype=dtype)
        Jacobian = csr_matrix(np.zeros((self.grid.Np,self.grid.Np),dtype=dtype))
        B = np.zeros(Jacobian.shape[0],dtype=Jacobian.dtype)

        Jacobian_lil = Jacobian.tolil()
        self.NR_construct_Jac_B(Jacobian_lil,B)
        Jacobian = Jacobian_lil.tocsr() 
        return Jacobian,B
    
    
    def to_scipy_Jac_B(self,dtype=np.float64):
        """
        Constructs the Jacobian matrix and right-hand side vector (B) for the Poisson equation in SciPy sparse format.
        The method relies on the `NR_construct_Jac_B` method to fill in the Jacobian and B.
        Parameters
        ----------
        dtype : data-type, optional
            The desired data-type for the Jacobian and B arrays (default is np.float64).
        Returns
        -------
        Jacobian : scipy.sparse.csr_matrix
            The constructed Jacobian matrix in CSR sparse format.
        B : numpy.ndarray
            The right-hand side vector for the Poisson equation.
        """
        # create the Jacobian and B for the Poisson equation in scipy sparse format
        
        # Jacobian = csr_matrix(np.zeros((self.grid.Np,self.grid.Np),dtype=dtype))
        Jacobian = csr_matrix((self.grid.Np,self.grid.Np),dtype=dtype)
        B = np.zeros(Jacobian.shape[0],dtype=Jacobian.dtype)

        Jacobian_lil = Jacobian.tolil()
        self.NR_construct_Jac_B(Jacobian_lil,B)
        Jacobian = Jacobian_lil.tocsr() 
        # self.construct_poisson(A,b)
        return Jacobian,B
    


    def solve_poisson_NRcycle(self,method='pyamg',tolerance=1e-7,dtype:str='float64'):
        """
        Solve the Poisson equation using the Newton-Raphson (NR) iterative method.
        This NR method is inspired by NanoTCAD ViDES (http://vides.nanotcad.com/vides/),iteratively solving 
        the nonlinear Poisson equation by updating the potential (`self.phi`). 
        At each iteration, it constructs the Jacobian and right-hand side (B),
        solves the resulting linear system using either a direct solver ('scipy') or an algebraic multigrid solver ('pyamg'),
        and updates the potential. The process continues until the correction norm falls below a threshold or a maximum
        number of iterations is reached. The method also includes a control mechanism to prevent divergence by monitoring
        the norm of B after each update.
        Parameters
        ----------
        method : str, optional
            The linear solver to use for the Poisson equation. Options are:
            - 'pyamg': Use algebraic multigrid solver (default).
            - 'scipy': Use direct solver from scipy.
        tolerance : float, optional
            The tolerance for the linear solver (default: 1e-7).
        dtype : str, optional
            Data type for the computation, either 'float64' (default, recommended for stability) or 'float32'.
        Returns
        -------
        max_diff : float
            The maximum absolute difference between the updated potential (`self.phi`) and the previous potential (`self.phi_old`)
            after the NR cycle.
        Raises
        ------
        ValueError
            If an unknown data type or Poisson solver method is specified.
        Notes
        -----
        - The method logs progress and warnings during the NR cycle.
        - Includes a control mechanism to avoid increasing the norm of B after an NR update.
        - The NR cycle stops if the correction norm is below 1e-3 or after 100 iterations.
        """
        # solve the Poisson equation with Newton-Raphson method
        # delta_phi: the correction on the potential
        # It has been tested that dtype='float64' is a more stable SCF choice.
      
        
        norm_delta_phi = 1.0 #  Euclidean norm of delta_phi in each step
        NR_cycle_step = 0

        if dtype == 'float64':
            dtype = np.float64
        elif dtype == 'float32':
            dtype = np.float32
        else:
            raise ValueError('Unknown data type: ',dtype)

        while norm_delta_phi > 1e-3 and NR_cycle_step < 100:
            # obtain the Jacobian and B for the Poisson equation
            Jacobian,B = self.to_scipy_Jac_B(dtype=dtype)
            norm_B = np.linalg.norm(B)
           
            if method == 'scipy':   #TODO: rename to 'Direct
                if NR_cycle_step == 0:
                    log.info(msg="Solve Poisson equation by scipy")
                delta_phi = spsolve(Jacobian,B)
            elif method == 'pyamg': #TODO: rename to 'AMG'
                if NR_cycle_step == 0:
                    log.info(msg="Solve Poisson equation by pyamg")
                delta_phi = self.solver_pyamg(Jacobian,B,tolerance=1e-5)
            else:
                raise ValueError('Unknown Poisson solver: ',method)
                        
            max_delta_phi = np.max(abs(delta_phi))
            norm_delta_phi = np.linalg.norm(delta_phi)
            self.phi += delta_phi

            if norm_delta_phi > 1e-3:
                _,B = self.to_scipy_Jac_B()
                norm_B_new = np.linalg.norm(B)
                control_count = 1
                # control the norm of B to avoid larger norm_B after one NR cycle
                while norm_B_new > norm_B and control_count < 2:
                    if control_count==1: 
                        log.warning(msg="norm_B increase after this  NR cycle, contorler starts!")
                    self.phi -= delta_phi/np.power(2,control_count)
                    _,B = self.to_scipy_Jac_B()
                    norm_B_new = np.linalg.norm(B)
                    control_count += 1
                    log.info(msg="    control_count: {:.1f}   norm_B_new: {:.5f}".format(float(control_count),norm_B_new))    
                               
            NR_cycle_step += 1
            log.info(msg="  NR cycle step: {:d}   norm_delta_phi: {:.8f}   max_delta_phi: {:.8f}".format(int(NR_cycle_step),norm_delta_phi,max_delta_phi))
        
        max_diff = np.max(abs(self.phi-self.phi_old))
        return max_diff

    def solver_pyamg(self,A,b,tolerance=1e-7,accel=None):
        # solve the Poisson equation
        # log.info(msg="Solve Poisson equation by pyamg")
        try:
            import pyamg
        except:
            raise ImportError("pyamg is required for Poisson solver. Please install pyamg firstly! ")
        
        pyamg_solver = pyamg.aggregation.smoothed_aggregation_solver(A, max_levels=1000)
        del A
        # print('Poisson equation solver: ',pyamg_solver)
        residuals = []

        def callback(x):
        # residuals calculated in solver is a pre-conditioned residual
        # residuals.append(np.linalg.norm(b - A.dot(x)) ** 0.5)
            print(
                "    {:4d}  residual = {:.5e}   x0-residual = {:.5e}".format(
                    len(residuals) - 1, residuals[-1], residuals[-1] / residuals[0]
                )
            )

        x = pyamg_solver.solve(
            b,
            tol=tolerance,
            # callback=callback,
            residuals=residuals,
            accel=accel,
            cycle="W",
            maxiter=1e3,
        )
        return x
    
    def NR_construct_Jac_B(self,J,B):
        """
        Constructs the Jacobian matrix (J) and right-hand side vector (B) for the Newton-Raphson solution 
        of the Poisson equation on a 3D grid, accounting for both interior and boundary grid points.
        For interior points, the method computes flux contributions in the x, y, and z directions using 
        local permittivity, potential, and grid geometry.
        For boundary points, the method applies appropriate boundary conditions (Dirichlet or Neumann) by 
        modifying J and B accordingly, based on the type of boundary (xmin, xmax, ymin, ymax, zmin, zmax, or Dirichlet).
        After assembling the contributions, the sign of B is flipped for nonzero entries for the Newton-Raphson iteration.
        Parameters
        ----------
        J : numpy.ndarray
            The Jacobian matrix to be constructed/updated (shape: [Np, Np], where Np is the number of grid points).
        B : numpy.ndarray
            The right-hand side vector to be constructed/updated (shape: [Np]).
        Notes
        -----
        - Assumes that self.grid, self.eps, self.phi, self.phi_old, self.free_charge, self.fixed_charge, 
            self.kBT, self.boudnary_points, and self.lead_gate_potential are properly initialized.
        - Uses constants such as eps0 and elementary_charge, which must be defined in the scope.
        - The method modifies J and B in place.
        """
        # construct the Jacobian and B for the Poisson equation
        def average_eps(eps1, eps2, mode:str='harmonic'):
            if mode == 'arithmetic':
                return 0.5 * (eps1 + eps2)
            elif mode == 'harmonic':
                return 2.0 * eps1 * eps2 / (eps1 + eps2)
        average_mode = self.average_mode
        Nx = self.grid.shape[0];Ny = self.grid.shape[1];Nz = self.grid.shape[2]
        for gp_index in range(self.grid.Np):
            if self.boudnary_points[gp_index] == "in":
                flux_xm_J = self.grid.surface_grid[gp_index,0]*eps0*average_eps(self.eps[gp_index-1],self.eps[gp_index],mode = average_mode)\
                /abs(self.grid.grid_coord[gp_index,0]-self.grid.grid_coord[gp_index-1,0])
                flux_xm_B = flux_xm_J*(self.phi[gp_index-1]-self.phi[gp_index])

                flux_xp_J = self.grid.surface_grid[gp_index,0]*eps0*average_eps(self.eps[gp_index+1],self.eps[gp_index],mode = average_mode)\
                /abs(self.grid.grid_coord[gp_index+1,0]-self.grid.grid_coord[gp_index,0])
                flux_xp_B = flux_xp_J*(self.phi[gp_index+1]-self.phi[gp_index])
                
                flux_ym_J = self.grid.surface_grid[gp_index,1]*eps0*average_eps(self.eps[gp_index-Nx],self.eps[gp_index],mode = average_mode)\
                /abs(self.grid.grid_coord[gp_index-Nx,1]-self.grid.grid_coord[gp_index,1])
                flux_ym_B = flux_ym_J*(self.phi[gp_index-Nx]-self.phi[gp_index])

                flux_yp_J = self.grid.surface_grid[gp_index,1]*eps0*average_eps(self.eps[gp_index+Nx],self.eps[gp_index],mode = average_mode)\
                /abs(self.grid.grid_coord[gp_index+Nx,1]-self.grid.grid_coord[gp_index,1])
                flux_yp_B = flux_yp_J*(self.phi[gp_index+Nx]-self.phi[gp_index])

                flux_zm_J = self.grid.surface_grid[gp_index,2]*eps0*average_eps(self.eps[gp_index-Nx*Ny],self.eps[gp_index],mode = average_mode)\
                /abs(self.grid.grid_coord[gp_index-Nx*Ny,2]-self.grid.grid_coord[gp_index,2])
                flux_zm_B = flux_zm_J*(self.phi[gp_index-Nx*Ny]-self.phi[gp_index])

                flux_zp_J = self.grid.surface_grid[gp_index,2]*eps0*average_eps(self.eps[gp_index+Nx*Ny],self.eps[gp_index],mode = average_mode)\
                /abs(self.grid.grid_coord[gp_index+Nx*Ny,2]-self.grid.grid_coord[gp_index,2])
                flux_zp_B = flux_zp_J*(self.phi[gp_index+Nx*Ny]-self.phi[gp_index])

                # add flux term to matrix J
                J[gp_index,gp_index] = -(flux_xm_J+flux_xp_J+flux_ym_J+flux_yp_J+flux_zm_J+flux_zp_J)\
                    +elementary_charge*self.free_charge[gp_index]*(-np.sign(self.free_charge[gp_index]))/self.kBT*\
                    np.exp(-np.sign(self.free_charge[gp_index])*(self.phi[gp_index]-self.phi_old[gp_index])/self.kBT)
                J[gp_index,gp_index-1] = flux_xm_J
                J[gp_index,gp_index+1] = flux_xp_J
                J[gp_index,gp_index-Nx] = flux_ym_J
                J[gp_index,gp_index+Nx] = flux_yp_J
                J[gp_index,gp_index-Nx*Ny] = flux_zm_J
                J[gp_index,gp_index+Nx*Ny] = flux_zp_J


                # add flux term to matrix B
                B[gp_index] = (flux_xm_B+flux_xp_B+flux_ym_B+flux_yp_B+flux_zm_B+flux_zp_B)
                B[gp_index] += elementary_charge*self.free_charge[gp_index]*np.exp(-np.sign(self.free_charge[gp_index])\
                    *(self.phi[gp_index]-self.phi_old[gp_index])/self.kBT)+elementary_charge*self.fixed_charge[gp_index]

            else:# boundary points
                J[gp_index,gp_index] = 1.0 # correct for both Dirichlet and Neumann boundary condition
                
                if self.boudnary_points[gp_index] == "xmin":   
                    J[gp_index,gp_index+1] = -1.0
                    B[gp_index] = (self.phi[gp_index]-self.phi[gp_index+1])
                elif self.boudnary_points[gp_index] == "xmax":
                    J[gp_index,gp_index-1] = -1.0
                    B[gp_index] = (self.phi[gp_index]-self.phi[gp_index-1])
                elif self.boudnary_points[gp_index] == "ymin":
                    J[gp_index,gp_index+Nx] = -1.0
                    B[gp_index] = (self.phi[gp_index]-self.phi[gp_index+Nx])
                elif self.boudnary_points[gp_index] == "ymax":
                    J[gp_index,gp_index-Nx] = -1.0
                    B[gp_index] = (self.phi[gp_index]-self.phi[gp_index-Nx])
                elif self.boudnary_points[gp_index] == "zmin":
                    J[gp_index,gp_index+Nx*Ny] = -1.0
                    B[gp_index] = (self.phi[gp_index]-self.phi[gp_index+Nx*Ny])
                elif self.boudnary_points[gp_index] == "zmax":
                    J[gp_index,gp_index-Nx*Ny] = -1.0
                    B[gp_index] = (self.phi[gp_index]-self.phi[gp_index-Nx*Ny])
                elif self.boudnary_points[gp_index] == "Dirichlet":
                    B[gp_index] = (self.phi[gp_index]-self.lead_gate_potential[gp_index])

            if B[gp_index]!=0: # for convenience change the sign of B in later NR iteration
                B[gp_index] = -B[gp_index]           