""" 
   Set of functions for 2D trilateration

   Three variants:
   - trilaterate_circle_intersect: analytic approach based on circle intersections
   - trilaterate_lstsq: based on reformulation into least squares problem
   - trilaterate_nonlin_leastsq: based on nonlinear least squares optimization
   
   Developed for the course PosBep (Positiebepaling) at Mechatronics, 
   The Hague University of Applied Sciences

   No Warranty

   Rufus Fraanje, GNU-GPLv3, 2022/05/03
"""

import numpy as np            # numpy for working with arrays
import sympy                  # sympy for symbolic math (calculating derivatives)
from scipy import optimize    # scipy.optimize for nonlinear least squares optimalisation
from geometry import *        # geometry.py contains number of geometry related functions


def trilaterate_circle_intersect(b1,b2,b3,r1,r2,r3,variant=1):
    """trilaterate_circle_intersect(b1,b2,b3,r1,r2,r3) returns the point (if it exists) 
       that is at distance r1 to beacon 1 (b1), r2 to beacon 2 (b2) and
       r3 to beacon 3 (b3) based on circle intersections.
       The beacons b1, b2, b3 should not be on a straight line, i.e. not collinear.
       variant=1: computes intersections of circle 1 and 2 and checks which lays on circle 3
              =2: tries other combinations of circles and returns the solution closest to the other circle
              =3: computes all pairs of solutions, selects the 3 closest variants in the pairs and returns
                  their average (computationally most complex but expected to be the most accurate solution).
    """
    if is_equal(b1,b2) or is_equal(b1,b3) or is_equal(b2,b3):
        raise ValueError('All beacons should be at different locations, at least two have same location.')

    # check whether b1,b2,b3 are on a straight line
    if is_collinear(b1,b2,b3):
        raise ValueError('b1, b2 and b3 are collinear, but should form a real triangle.')

    if variant == 1:
        # Variant 1:
        c_other, r_other   = b3, r3
        p = circle_intersect(b1,b2,r1,r2)

        if is_on_circle(c_other,r_other,p[0]):
            return p[0]
        elif is_on_circle(c_other,r_other,p[1]):
            return p[1]
        else:
            raise ValueError('No solution found.')
    elif variant == 2:
        # Variant 2:
        c_other, r_other   = b3, r3
        try:
            p = circle_intersect(b1,b2,r1,r2)
        except:
            c_other, r_other   = b2, r2
            try:
                p = circle_intersect(b1,b3,r1,r3)
            except:
                c_other, r_other   = b1, r1
                try:
                    p = circle_intersect(b2,b3,r2,r3)
                except:
                    raise ValueError('No solution found.')
                
        if np.abs(distance(p[0],c_other)-r_other) < np.abs(distance(p[1],c_other)-r_other):
            return p[0]
        else:
            return p[1]
        
    else:
        # Variant 3:
        try:
            sol12 = circle_intersect(b1,b2,r1,r2)
            sol13 = circle_intersect(b1,b3,r1,r3)
            sol23 = circle_intersect(b2,b3,r2,r3)
            indices = nearest_points(sol12,sol13,sol23)
            return (sol12[indices[0]] + sol13[indices[1]] + sol23[indices[2]])/3
        except:
            try:
                sol12 = circle_intersect(b1,b2,r1,r2)
                sol13 = circle_intersect(b1,b3,r1,r3)
                indices = nearest_points(sol12,sol13)
                return (sol12[indices[0]] + sol13[indices[1]])/2
            except:
                try:
                    sol12 = circle_intersect(b1,b2,r1,r2)
                    sol23 = circle_intersect(b2,b3,r2,r3)
                    indices = nearest_points(sol12,sol23)
                    return (sol12[indices[0]] + sol23[indices[1]])/2
                except:
                    try:
                        sol13 = circle_intersect(b1,b3,r1,r3)
                        sol23 = circle_intersect(b2,b3,r2,r3)
                        indices = nearest_points(sol13,sol23)
                        return (sol13[indices[0]] + sol23[indices[1]])/2
                    except ValueError as e:
                        raise e
                    
    
def trilaterate_lstsq(beacon_positions,distances_measured):
    """ trilaterate_lstsq(beacon_positions,distances_measured)
    Solves a least squares problem to estimate the point that is at distance distances_measured[i] of 
    circle with center at (beacon_positions[i,0], beacon_positions[i,1]) for i ranging over the number
    of beacons, which is at least 3, but can be larger. The beacons should not be collinear, i.e. they
    should not all lay on a straight line.

    The problem is made linear by introducing a slack variable q = x^2+y^2 so we can write:
    (x-x_i)^2 + (y-y_i^2) = q^2 -2x_i x - 2y_i y + x_i^2+y_i^2 = [1 -2x_i -2y_i] [ q ] + x_i^2+y_i^2
                                                                                 [ x ]
                                                                                 [ y ]
    this should be as close to distances_measured_i^2 as possible for i ranging over the number of beacons.

    Note, there are other (and better) approaches, e.g. taking q = x^2+y^2 as an additional constraint.
    """
    nb = len(beacon_positions)
    A = np.hstack((np.ones((nb,1)),-2*beacon_positions))
    b = np.zeros((nb,1))
    for i in range(nb):
        b[i] = distances_measured[i]**2-beacon_positions[i,0]**2-beacon_positions[i,1]**2
    qxy,res,rank,sv = np.linalg.lstsq(A,b,rcond=1e-10)
    return np.array([qxy[1,0], qxy[2,0]])

                    
def trilaterate_nonlin_leastsq(beacon_positions,distances_measured,pos_init=None):
    """trilaterate_nonlin_leastsq(beacon_positions,distances_measured,pos_init=None)
    Solves a nonlinear least squares problem to estimate the point that is at distance distances_measured[i] of 
    circle with center at (beacon_positions[i,0], beacon_positions[i,1]) for i ranging over the number
    of beacons, which is at least 3, but can be larger. The beacons should not be collinear, i.e. they
    should not all lay on a straight line.
    The method is iterative and needs an initial estimate of the position, pos_init. If pos_init is not specified
    pos_init is set to np.array([0., 0.]).

    The problem solved is a nonlinear least squares problem specified by:
      min_(x,y) || f(x,y) ||_2
    where the vector function f(x) contains the difference between the measured and calculated distances:
    
    f(x,y) = [ distances_measured[0] - sqrt( (x - beacon_positions[0])^2 + (y - beacon_positions[0])^2 )          ]
             [ distances_measured[1] - sqrt( (x - beacon_positions[1])^2 + (y - beacon_positions[1])^2 )          ]
             [                                 ...                                                                ]
             [ distances_measured[nb-1] - sqrt( (x - beacon_positions[nb-1])^2 + (y - beacon_positions[nb-1])^2 ) ]
    
    with nb the number of beacons and b equals the vector distances_measured.

    The method relies on the least_squares optimization function in the scipy submodule optimize.
    This function also needs the gradient of f (called the Jacobian), which is computed automatically
    using sympy, the Python symbolic math module.
    """
    
    # helper function for making sympy functions usable in scipy.optimization.
    def func_argvec_to_args(f):
        """
        Wrapper for f(X) -> f(X[0],X[1])
        From Ch6 in the book: Robert Johansson, Numerical Python: Scientific Computing and Data Science Applications
        with Numpy, SciPy and Matplotlib, Apress, available from
        https://link-springer-com.ezproxy.hhs.nl/book/10.1007/978-1-4842-4246-9
        """
        return lambda X: np.squeeze(np.array(f(X[0],X[1])))

    if pos_init is None:
        pos_init = np.array([0.,0.])

    x,y = sympy.symbols("x, y")
    f_sym = sympy.Matrix([
        d - sympy.sqrt( (x-b[0])**2 + (y-b[1])**2 )
        for (b,d) in zip(beacon_positions,distances_measured)
    ])
    jacobian_sym = sympy.Matrix.hstack(f_sym.diff(x), f_sym.diff(y))
    f_num   = sympy.lambdify((x,y),f_sym, 'numpy')
    jacobian_num = sympy.lambdify((x,y),jacobian_sym, 'numpy')
    f = func_argvec_to_args(f_num)
    jacobian = func_argvec_to_args(jacobian_num)

    res = optimize.least_squares(f,pos_init,jac=jacobian)

    pos_estimate = res.x

    return pos_estimate
        
def trilaterate_combined(beacon_positions,distances_measured):
    """trilaterate_combined(beacon_positions,distances_measured)
    Solves the trilateration problem using nonlinear least squares with
    initialization based on the least squares approximate solution. Method implements
    pe_init = trilaterate_lstsq(beacon_positions,distances_measured)
    pe = trilaterate_nonlin_leastsq(beacon_positions,distances_measured,pos_init=pe_init)
    return pe
    """
    pe_init = trilaterate_lstsq(beacon_positions,distances_measured)
    pe = trilaterate_nonlin_leastsq(beacon_positions,distances_measured,pos_init=pe_init)
    return pe
    
    
    
    
