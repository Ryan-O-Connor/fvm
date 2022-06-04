import unittest

import numpy as np


class Volume1D:

    def __init__(self, x, W_neighbor, E_neighbor):
        self.x = x
        self.W_neighbor = W_neighbor
        self.E_neighbor = E_neighbor        

    def __repr__(self):
        return "1D Volume Element @ {}\n\tWest Neighbor: {}\n\tEast Neighbor: {}".format(self.x, self.W_neighbor, self.E_neighbor)


class Mesh:

    def __init__(self, volumes, dx):
        self.volumes = volumes
        self.dx = dx

    def __repr__(self):
        n = self.num_volumes()
        rep = "Mesh: {} volumes\n".format(n)
        rep += "Volume spacing: {}\n".format(self.dx)
        for i in range(n):
            rep += self.volumes[i].__repr__() + "\n"
        return rep

    def num_volumes(self):
        return len(self.volumes)

    @classmethod
    def grid1DMesh(cls, x_min, x_max, n):
        # Splits continuous 1D span (x_min, x_max) into n domains
        dx = (x_max - x_min) / n
        volumes = []
        for i in range(n):
            x_coordinate = (i + 0.5) * dx
            if i == 0:
                W_neighbor = None
                E_neighbor = 1
            elif i == n-1:
                W_neighbor = n-2
                E_neighbor = None
            else:
                W_neighbor = i-1
                E_neighbor = i+1
            volumes.append(Volume1D(x_coordinate, W_neighbor, E_neighbor))
        return cls(volumes, dx)


class MaterialProperties:

    def __init__(self):
        self.props = {}

    def __repr__(self):
        rep = "Material properties:\n"
        for key, value in self.props.items():
            rep += "\t{}: {}\n".format(key, value)
        return rep

    def get(self, name):
        if self.props.get(name) is None:
            raise RuntimeError("Unexpected property retrieval: {}".format(name))
        return self.props[name]

    def define_property(self, name, value):
        self.props[name] = value


class BoundaryAssignment:

    def __init__(self, boundary, value):
        self.boundary = boundary
        self.value = value


class BoundaryConditions:

    def __init__(self):
        self.essential_bcs = {}
        self.natural_bcs = {}

    def __repr__(self):
        rep = "Boundary Conditions:\n"
        for node_id, assignment in self.essential_bcs.items():
            rep += "\tNode ID {}: {}deg on {} boundary\n".format(node_id, assignment.temp, assignment.boundary)
        return rep

    def num_bcs(self):
        return len(self.essential_bcs)

    def is_boundary_node(self, node_id):
        return self.essential_bcs.get(node_id) is not None or self.natural_bcs.get(node_id) is not None

    def is_essential_bc(self, node_id):
        return self.essential_bcs.get(node_id) is not None

    def is_natural_bc(self, node_id):
        return self.natural_bcs.get(node_id) is not None

    def is_west_boundary(self, node_id):
        if self.is_essential_bc(node_id):
            assignment = self.essential_bcs[node_id]
        else:
            assignment = self.natural_bcs[node_id]
        return assignment.boundary == "W"

    def is_east_boundary(self, node_id):
        if self.is_essential_bc(node_id):
            assignment = self.essential_bcs[node_id]
        else:
            assignment = self.natural_bcs[node_id]
        return assignment.boundary == "E"
    
    def get_boundary_temp(self, node_id):
        return self.essential_bcs[node_id].value
    
    def get_boundary_flux(self, node_id):
        return self.natural_bcs[node_id].value

    def set_temperature(self, node_id, boundary, temp):
        self.essential_bcs[node_id] = BoundaryAssignment(boundary, temp)

    def set_flux(self, node_id, boundary, flux):
        self.natural_bcs[node_id] = BoundaryAssignment(boundary, flux)


class UniformHeatGeneration:

    def __init__(self, q):
        self.q = q


class UniformConvection:

    def __init__(self, h, Tinf):
        self.h = h
        self.Tinf = Tinf


class Loads:

    def __init__(self):
        self.heat_generation = None
        self.convection = None

    def add_uniform_heat_generation(self, q):
        self.heat_generation = UniformHeatGeneration(q)

    def add_uniform_ambient_convection(self, h, Tinf):
        self.convection = UniformConvection(h, Tinf)


class ConductionProblem:

    def __init__(self, mesh, props, bcs, loads):
        self.mesh = mesh
        self.props = props
        self.bcs = bcs
        self.loads = loads
        self.soln = None

    def solve(self):
        # Allocate system
        num_volumes = self.mesh.num_volumes()
        A = np.zeros((num_volumes, num_volumes))
        b = np.zeros(num_volumes)
        # Setup system matrix for conduction
        self.set_conduction_terms(A, b)
        self.set_convection_terms(A, b)
        self.set_heat_generation_terms(b)
        # Solve and store
        x = np.linalg.solve(A, b)
        self.soln = x

    def print_soln(self):
        if self.soln is None:
            return
        print(self.soln)

    def set_conduction_terms(self, A, b):
        # Add terms to system matrix associated with pure conduction
        kxx = self.props.get("kxx")
        Ax = self.props.get("Ax")
        dx = self.mesh.dx
        coef = kxx*Ax/dx
        for i in range(self.mesh.num_volumes()):
            if self.bcs.is_boundary_node(i):
                if self.bcs.is_essential_bc(i):
                    A[i, i] = 3*coef
                    boundary_temp = self.bcs.get_boundary_temp(i)
                    b[i] = 2*coef*boundary_temp
                    if self.bcs.is_west_boundary(i):
                        A[i, i+1] = -coef
                    elif self.bcs.is_east_boundary(i):
                        A[i, i-1] = -coef
                    else:
                        raise RuntimeError("Unknown boundary node: {}".format(i))
                else:
                    A[i, i] = coef
                    boundary_flux = self.bcs.get_boundary_flux(i)
                    b[i] = boundary_flux*Ax
                    if self.bcs.is_west_boundary(i):
                        A[i, i+1] = -coef
                    elif self.bcs.is_east_boundary(i):
                        A[i, i-1] = -coef
                    else:
                        raise RuntimeError("Unknown boundary node: {}".format(i))
            else:
                A[i, i-1] = -coef
                A[i, i] = 2*coef
                A[i, i+1] = -coef
                
    def set_convection_terms(self, A, b):
        if self.loads.convection is None:
            return
        Px = self.props.get("Px")
        dx = self.mesh.dx
        h = self.loads.convection.h
        Tinf = self.loads.convection.Tinf
        coef = h*Px*dx
        for i in range(self.mesh.num_volumes()):
            A[i, i] += coef
            b[i] += coef*Tinf

    def set_heat_generation_terms(self, b):
        # Add terms to load vector for uniform heat generation
        if self.loads.heat_generation is None:
            return
        Ax = self.props.get("Ax")
        dx = self.mesh.dx
        q = self.loads.heat_generation.q
        term = q*Ax*dx
        for i in range(self.mesh.num_volumes()):
            b[i] += term
        


class Tests(unittest.TestCase):

    def testConduction(self):
        mesh = Mesh.grid1DMesh(0, 0.5, 5)
        props = MaterialProperties()
        props.define_property("kxx", 1000)
        props.define_property("Ax", 10*10**-3)
        bcs = BoundaryConditions()
        bcs.set_temperature(0, "W", 100)
        bcs.set_temperature(4, "E", 500)
        loads = Loads()
        problem = ConductionProblem(mesh, props, bcs, loads)
        problem.solve()
        expected_soln = np.array([140, 220, 300, 380, 460])
        self.assertTrue(np.allclose(expected_soln, problem.soln))
    

    def testHeatGeneration(self):
        mesh = Mesh.grid1DMesh(0, .02, 5)
        props = MaterialProperties()
        props.define_property("kxx", 0.5)
        props.define_property("Ax", 1)
        bcs = BoundaryConditions()
        bcs.set_temperature(0, "W", 100)
        bcs.set_temperature(4, "E", 200)
        loads = Loads()
        loads.add_uniform_heat_generation(1000*10**3)
        problem = ConductionProblem(mesh, props, bcs, loads)
        problem.solve()
        expected_soln = np.array([150, 218, 254, 258, 230])
        self.assertTrue(np.allclose(expected_soln, problem.soln))

    def testConvection(self):
        mesh = Mesh.grid1DMesh(0, 1, 5)
        props = MaterialProperties()
        props.define_property("kxx", 10)
        props.define_property("Ax", 1)
        props.define_property("Px", 1)
        bcs = BoundaryConditions()
        bcs.set_temperature(0, "W", 100)
        bcs.set_flux(4, "E", 0)
        loads = Loads()
        loads.add_uniform_ambient_convection(h=25*10, Tinf=20)
        problem = ConductionProblem(mesh, props, bcs, loads)
        problem.solve()
        expected_soln = np.array([64.22, 36.91, 26.5, 22.6, 21.3])
        self.assertTrue(np.allclose(expected_soln, problem.soln, atol=0.01))


if __name__ == '__main__':
    unittest.main(verbosity=2)   
    
    
