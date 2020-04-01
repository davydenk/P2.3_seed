/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2019 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Authors: Wolfgang Bangerth, 1999,
 *          Guido Kanschat, 2011
 */

#include <deal.II/base/timer.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/parsed_convergence_table.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>

using namespace dealii;


template <int dim> double coefficient(const Point<dim> &p){
  if (p.square() < 0.5 * 0.5)
    return 20;
  else
    return 1;
}

template <int dim>
class Step5{
public:
  Step5();
  void run();
private:
  //don't need make grid here, but keep it for easier history
  void make_cube_grid();
  void make_lshape_grid();
  void setup_system();
  void assemble_system();
  void solve();
  void output_results(const unsigned int cycle) const;
  void compute_error();
  
  mutable TimerOutput timer;
  Triangulation<dim> triangulation;
  FE_Q<dim>          fe;
  DoFHandler<dim>    dof_handler;
  
  Vector<double> L2_error_per_cell;
  Vector<double> H1_error_per_cell;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;
  Vector<double> solution;
  Vector<double> system_rhs;
  FunctionParser<dim> exact_solution;
  ParsedConvergenceTable error_table;
};

template <int dim>
Step5<dim>::Step5()
 : timer(std::cout, TimerOutput::summary, TimerOutput::cpu_and_wall_times),
    fe(1), 
    dof_handler(triangulation),
    exact_solution("exp(x)*exp(y)"),
    error_table({"u"}, {{VectorTools::H1_norm, VectorTools::L2_norm}})
{}

template <int dim> 
void Step5<dim>::make_cube_grid(){
  TimerOutput::Scope timer_section(timer, "Make cube grid");
  GridGenerator::hyper_cube(triangulation, -1, 1);
 // triangulation.begin_active()->face(0)->set_boundary_id(1); //we want one of faces have 1
  triangulation.refine_global(5);
    //let's see what we work with
  	  std::ofstream file_var("grid-cube3.vtu");
  	    GridOut       grid_out;
  	    grid_out.write_vtu(triangulation, file_var);
  	    std::cout << "Grid written to grid-cube3.vtu" << std::endl;
  //std::cout << "Number of active cells: " << triangulation.n_active_cells() << std::endl;
}

//ATTENTION! this is not a good function, we were lazy to write template specialization
//so it actually only works in 2d
template <int dim>
void Step5<dim>::make_lshape_grid(){
  TimerOutput::Scope timer_section(timer, "Make lshape grid");
  Triangulation<2> tr1,tr2,tr3;
  const Point<2> p1(0, 0);
  const Point<2> p2(1,2);
  const Point<2> p3(2,1);
  const Point<2> p4(1,0);
  const Point<2> p5(1,1);
  const Point<2> p6(0,1);
  GridGenerator::hyper_rectangle(tr1,p1,p5);
  GridGenerator::hyper_rectangle(tr2,p6,p2);
  GridGenerator::hyper_rectangle(tr3,p4,p3);
  //they cannot be refined!
 GridGenerator::merge_triangulations ({&tr1,&tr2,&tr3},triangulation);
 for (unsigned int step = 0; step < 5; ++step) {
      // Active cells are those that are not further refined
	  //we need to mark cells for refinement
      for (auto &cell : triangulation.active_cell_iterators()){
          // Next, we want to loop over all vertices of the cells.
          for (unsigned int v = 0; v < GeometryInfo<2>::vertices_per_cell; ++v){
        	  //std::cout<< "vertex is "<< cell->vertex(v)<<std::endl;
              const double distance_from_corner =p5.distance(cell->vertex(v));
              //choose whatever refinement condition
        //      if ( distance_from_corner  < 1e-10){
             if ( distance_from_corner  < 1.0/3){
            	  cell->set_refine_flag();
                  break;
               }//if
            }//for
        }
      //refine global calls this function too
      triangulation.execute_coarsening_and_refinement();
    }//for steps loop

 triangulation.refine_global(2); //we want a nice picture :)
  std::ofstream file_var("grid-lshape.vtu");
  GridOut       grid_out;
  grid_out.write_vtu(triangulation, file_var);
  std::cout << "Grid written to grid-lshape.vtu" << std::endl;
}

template <int dim>
void Step5<dim>::setup_system(){
  TimerOutput::Scope timer_section(timer, "setup system");
  dof_handler.distribute_dofs(fe);
  std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs()<< std::endl;
  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit(sparsity_pattern);//cause we might use pattern for other matrix
  solution.reinit(dof_handler.n_dofs());//just setting size
  system_rhs.reinit(dof_handler.n_dofs());
}

template <int dim>
void Step5<dim>::assemble_system(){
  TimerOutput::Scope timer_section(timer, "assemble system");
  QGauss<dim> quadrature_formula(fe.degree + 1);
  FEValues<dim> fe_values(fe, quadrature_formula, update_values |update_gradients |
                          update_quadrature_points | update_JxW_values);
  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  for (const auto &cell : dof_handler.active_cell_iterators()) {
      fe_values.reinit(cell);
      cell_matrix = 0;
      cell_rhs    = 0;
      for (unsigned int q_index = 0; q_index < n_q_points; ++q_index){
        const double current_coefficient =
        coefficient<dim>(fe_values.quadrature_point(q_index));
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              cell_matrix(i, j) += (current_coefficient *              // a(x_q)
              fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
              fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
              fe_values.JxW(q_index));           // dx
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            cell_rhs(i) += (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                            1 *                                 // f(x_q)
                            fe_values.JxW(q_index));            // dx
        }//over the cell

      cell->get_dof_indices(local_dof_indices);
      for (unsigned int i = 0; i < dofs_per_cell; ++i){
        for (unsigned int j = 0; j < dofs_per_cell; ++j){
          system_matrix.add(local_dof_indices[i],local_dof_indices[j],cell_matrix(i, j));
        }
      }
      for (unsigned int i = 0; i < dofs_per_cell; ++i){
        system_rhs(local_dof_indices[i]) += cell_rhs(i);
      }
    }
  std::map<types::global_dof_index, double> boundary_values;
  //dirichlet
  VectorTools::interpolate_boundary_values(dof_handler,
              0, //indicator, by default 0 on boundary
              Functions::ConstantFunction<dim>(0.0),boundary_values);
              // Functions::ZeroFunction<dim>(),boundary_values);
  
  //having different here produces nice pictures
/*  VectorTools::interpolate_boundary_values (dof_handler,
                1,//we changed that for one of cube faces
                ConstantFunction<dim>(2.),boundary_values);
*/  
  MatrixTools::apply_boundary_values(boundary_values,system_matrix,
                                     solution,system_rhs);
}

template <int dim>
void Step5<dim>::solve(){
  TimerOutput::Scope timer_section(timer, "solve");
  SolverControl solver_control(1000, 1e-12);
  SolverCG<>    solver(solver_control);
  PreconditionSSOR<> preconditioner;
  preconditioner.initialize(system_matrix, 1.2);
  solver.solve(system_matrix, solution, system_rhs, preconditioner);
  std::cout << "   " << solver_control.last_step()
  << " CG iterations needed to obtain convergence." << std::endl;  
}

template <int dim>
void Step5<dim>::output_results(const unsigned int cycle) const{
  TimerOutput::Scope timer_section(timer, "output");
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
  data_out.build_patches();
  std::ofstream output("solution-" + std::to_string(cycle) + ".vtu");
  data_out.write_vtu(output);
  std::cout << "Mean value: " << VectorTools::compute_mean_value (dof_handler,
              QGauss<dim>(fe.degree + 1), solution, 0) << std::endl;
}

template <int dim>
void Step5<dim>::compute_error(){
  TimerOutput::Scope timer_section(timer, "Compute error");
  L2_error_per_cell.reinit(triangulation.n_active_cells());
  H1_error_per_cell.reinit(triangulation.n_active_cells());
  QGauss<dim> error_quadrature(2 * fe.degree + 1);
  
  VectorTools::integrate_difference(dof_handler, solution, exact_solution,
                                    L2_error_per_cell, error_quadrature, VectorTools::L2_norm);
  
  VectorTools::integrate_difference(dof_handler, solution, exact_solution,
                                    H1_error_per_cell, error_quadrature,
                                    VectorTools::H1_norm);
  
  std::cout << "L2 norm of error: " << L2_error_per_cell.l2_norm() << std::endl;
  std::cout << "H1 norm of error: " << H1_error_per_cell.l2_norm() << std::endl;
}


template <int dim>
void Step5<dim>::run(){
  //read the grid from file instread of creating it
  GridIn<dim> grid_in;
  grid_in.attach_triangulation(triangulation);
  std::ifstream input_file("circle-grid.inp");
  Assert(dim == 2, ExcInternalError());
  grid_in.read_ucd(input_file);
  const SphericalManifold<dim> boundary;
  triangulation.set_all_manifold_ids_on_boundary(0);
  triangulation.set_manifold(0, boundary);
  //doing non-adaptive successive refinement
  for (unsigned int cycle = 0; cycle < 6; ++cycle){
    std::cout << "Cycle " << cycle << ':' << std::endl;
    if (cycle != 0)
      triangulation.refine_global(1);
    std::cout << "Number of active cells: " << triangulation.n_active_cells() << std::endl                      
    << "  Total number of cells: "   << triangulation.n_cells() << std::endl;
    setup_system();
    assemble_system();
    solve();
    output_results(cycle);
  }
  //compute_error();
  //error_table.error_from_exact(dof_handler, solution, exact_solution);

  //error_table.output_table(std::cout);
}

int main(){
  deallog.depth_console(2);

  Step5<2> laplace_problem;
  laplace_problem.run();

  return 0;
}
