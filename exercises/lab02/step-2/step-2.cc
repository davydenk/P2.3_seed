/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2015 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Wolfgang Bangerth, University of Heidelberg, 1999
 *
 * based on deal.II step-2
 */

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <fstream>

using namespace dealii;

void make_shell_grid (Triangulation<2> &tria){
  const Point<2> center (1,0);
  const double inner_radius = 0.5, outer_radius = 1.0;
  GridGenerator::hyper_shell (tria, center, inner_radius, outer_radius,5 );
  static const SphericalManifold<2> manifold_description(center);
  tria.set_all_manifold_ids(0);
  tria.set_manifold (0, manifold_description);
  for (unsigned int step=0; step<3; ++step){
	  for (auto &cell : tria.active_cell_iterators()){
        for (unsigned int v=0;v < GeometryInfo<2>::vertices_per_cell;++v){
            const double distance_from_center= center.distance (cell->vertex(v));
            if (std::fabs(distance_from_center - inner_radius) < 1e-10){
                cell->set_refine_flag ();
                break;
              }
          }
	  }//for active cells
      tria.execute_coarsening_and_refinement ();
    }//for steps
  //let's see what we work with
	  std::ofstream file_var("grid-shell2.vtu");
	    GridOut       grid_out;
	    grid_out.write_vtu(tria, file_var);
	    std::cout << "Grid written to grid-shell2.vtu" << std::endl;
}

template <int dim>
void make_cube_grid (Triangulation<dim> &tria){
  GridGenerator::hyper_cube(tria);
    tria.refine_global(2); //powers of 2
  //let's see what we work with
	  std::ofstream file_var("grid-cube2.vtu");
	    GridOut       grid_out;
	    grid_out.write_vtu(tria, file_var);
	    std::cout << "Grid written to grid-cube2.vtu" << std::endl;
}

//I hate red color, so I copied the library function here and modified it to print in green
void my_print_svg(SparsityPattern & sparsity_pattern, std::ostream &out){
  unsigned int m = sparsity_pattern.n_rows();
  unsigned int n = sparsity_pattern.n_cols();
  out
    << "<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" viewBox=\"0 0 "
    << n + 2 << " " << m + 2 << " \">\n"
       "<style type=\"text/css\" >\n"
       "     <![CDATA[\n"
       "      rect.pixel {\n"
       "          fill:   #05752f;\n"
       "      }\n"
       "    ]]>\n"
       "  </style>\n\n"
       "   <rect width=\""
    << n + 2 << "\" height=\"" << m + 2 << "\" fill=\"rgb(128, 128, 128)\"/>\n"
       "   <rect x=\"1\" y=\"1\" width=\""
    << n + 0.1 << "\" height=\"" << m + 0.1 << "\" fill=\"rgb(255, 255, 255)\"/>\n\n";
  SparsityPattern::iterator it = sparsity_pattern.begin(), end = sparsity_pattern.end();
  for (; it != end; ++it){
      out << "  <rect class=\"pixel\" x=\"" << it->column() + 1 << "\" y=\""
          << it->row() + 1 << "\" width=\".9\" height=\".9\"/>\n";
    }
  out << "</svg>" << std::endl;
}


template <int dim>
void distribute_dofs (DoFHandler<dim> &dof_handler){
  static const FE_Q<dim> finite_element(1);
  dof_handler.distribute_dofs (finite_element);
  DynamicSparsityPattern dsp(dof_handler.n_dofs(),dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern (dof_handler, dsp);
  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from (dsp);
  std::ofstream out ("sparsity_pattern1.svg");
  //sparsity_pattern.print_svg (out);
  my_print_svg(sparsity_pattern,out);
}

template <int dim>
void renumber_dofs (DoFHandler<dim> &dof_handler){
  //DoFRenumbering::Cuthill_McKee (dof_handler);
  //DoFRenumbering::hierarchical (dof_handler);
  //	DoFRenumbering::random (dof_handler);
	//DoFRenumbering::component_wise(dof_handler);
	//DoFRenumbering::boost::minimum_degree(dof_handler);
	DoFRenumbering::boost::king_ordering(dof_handler);
	DynamicSparsityPattern dynamic_sparsity_pattern(dof_handler.n_dofs(),
                                                  dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern (dof_handler, dynamic_sparsity_pattern);

  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from (dynamic_sparsity_pattern);
  std::ofstream out ("sparsity_pattern2.svg");
  my_print_svg(sparsity_pattern,out);
  //sparsity_pattern.print_svg (out);
}


int main (){
  Triangulation<3> triangulation;
  //make_shell_grid (triangulation);
  make_cube_grid<3>(triangulation);
  DoFHandler<3> dof_handler (triangulation);
  distribute_dofs (dof_handler);
  renumber_dofs (dof_handler);
}
