# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 11:32:08 2024

@author: stopfer
"""

import numpy


from docplex.mp.model import Model
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.converters import InequalityToEquality, IntegerToBinary, LinearEqualityToPenalty

#from dwave.system import DWaveSampler, AutoEmbeddingComposite


class Annealing():

    def __init__(self):     
        
        self.problem = ([1,2,3], 3, [0,2])
        # create a Mixed-Integer-Program
        self.mip_docplex = self.create_MIP(self.problem)
        
        # transform the MIP to a Qubo 
        self.qubo_operator, self.qubo_qiskit = self.transform_docplex_mip_to_qubo(self.mip_docplex, 1)
        
        #do the anneal
        self.anneal_solution = self.solve_qubo_via_anneal(self.qubo_operator, number_of_reads=1000, annealing_time=10)
        print(self.anneal_solution)
        return

    
    @staticmethod
    def create_MIP(problem: (list, float, list)) -> (Model, float): 
        
        # %% initialize the problem data
        problem_part1 = problem[0]
        problem_part2 = problem[1]
        problem_part3 = problem[2]
        
        # %% create the docplex model
        mip = Model("mymodel")
        
        #define the necessary variables for the creation
        border1 = len(problem_part1)
        border2 = len(problem_part1)
        
        # %% add model variables
        variables_1 = mip.binary_var_list(keys=range(border1), name=[f"x_{i}" for i in range(border1)])        
        variables_2 = mip.binary_var_matrix(keys1=range(border2), keys2=range(border1), name="y")
        
        # %% add model objective --> minimize sum of x_i variables
        objective = mip.minimize(mip.sum([variables_1[i] for i in range(border1)]))
        
        # %% add model constraints
        constraints_1 = mip.add_constraints((mip.sum(variables_2[o, i] for i in range(border1)) == 1 for o in range(border2)), ["constraints1_%d" % i for i in range(border2)])
        constraints_2 = mip.add_constraints((mip.sum(problem_part1[o] * variables_2[o, i] for o in range(border2)) <= problem_part2 * variables_1[i] for i in range(border1)), ["constraints2%d" % i for i in range(border1)])
        constraints_3 = mip.add_quadratic_constraints(variables_2[o1, i] * variables_2[o2, i] == 0 for (o1, o2) in problem_part3 for i in range(border1))
        
        return mip
    
    @staticmethod
    def transform_docplex_mip_to_qubo(mip_docplex: Model, penalty_factor) -> (dict, QuadraticProgram):
        #transform docplex model to the qiskit-optimization framework
        mip_qiskit = from_docplex_mp(mip_docplex)
        
        #transform inequalities to equalities --> with slacks
        mip_ineq2eq = InequalityToEquality().convert(mip_qiskit)
        
        #transform integer variables to binary variables -->split up into multiple binaries
        mip_int2bin = IntegerToBinary().convert(mip_ineq2eq)
        
        qubo = LinearEqualityToPenalty(penalty=penalty_factor).convert(mip_int2bin)
        
        # squash the quadratic and linear QUBO-coefficients together into a dictionary
        quadr_coeff = qubo.objective.quadratic.to_dict(use_name=True)
        lin_coeff = qubo.objective.linear.to_dict(use_name=True)                
        for var, var_value in lin_coeff.items():
            if (var,var) in quadr_coeff.keys():
                quadr_coeff[(var,var)] += var_value
            else:
                quadr_coeff[(var,var)] = var_value
        qubo_operator = quadr_coeff 
        
        return qubo_operator, qubo
    
    
    @staticmethod
    def solve_qubo_via_anneal(qubo_dict: dict, number_of_reads: int, annealing_time: int):
        my_token = input("dwave-token please: ")
        sampler = AutoEmbeddingComposite(DWaveSampler(token=my_token))
        response = sampler.sample_qubo(Q = qubo_dict, 
                                 num_reads = number_of_reads,
                                 annealing_time = annealing_time
                                 )
        return response.lowest().first.sample

Annealing()





















