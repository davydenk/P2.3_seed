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

 */

#include <deal.II/grid/tria.h>
// We need the following two includes for loops over cells and/or faces:
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
// Here are some functions to generate standard grids:
#include <deal.II/grid/grid_generator.h>
// Output of grids in various graphics formats:
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/manifold_lib.h>

#include <iostream>
#include <fstream>
#include <cmath>

using namespace dealii;

//simple square or cube
template <int dim>
void cube_grid(){
  Triangulation<dim> triangulation;
  //for 3 -  volume-wireframe in paraview
  //2,3 creates a square in 3d space
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(2); //powers of 2
  std::ofstream out("grid-cube.vtu");
  GridOut       grid_out;
  grid_out.write_vtu(triangulation, out);
  std::cout << "Grid written to grid-cube.vtu" << std::endl;
}

//rectangle  - introduce points
void rectangle_grid(){
  Triangulation<3> triangulation;
  //for 3 -  volume-wireframe in paraview
  const Point<3> p1(0, 0,0);
  const Point<3> p2(1,1, 3);
  GridGenerator::hyper_rectangle(triangulation,p1,p2);
  triangulation.refine_global(3); //powers of 2
  std::ofstream out("grid-rectangle.vtu");
  GridOut       grid_out;
  grid_out.write_vtu(triangulation, out);
  std::cout << "Grid written to grid-rectangle.vtu" << std::endl;
}

void lshape_grid(){
  Triangulation<2> tr1,tr2,tr3,tr_final;
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
 GridGenerator::merge_triangulations ({&tr1,&tr2,&tr3},tr_final);
 for (unsigned int step = 0; step < 5; ++step) {
      // Active cells are those that are not further refined
	  //we need to mark cells for refinement
      for (auto &cell : tr_final.active_cell_iterators()){
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
      tr_final.execute_coarsening_and_refinement();
    }//for steps loop

 tr_final.refine_global(2); //we want a nice picture :)

  std::ofstream file_var("grid-lshape.vtu");
  GridOut       grid_out;
  grid_out.write_vtu(tr_final, file_var);
  std::cout << "Grid written to grid-lshape.vtu" << std::endl;
}

//lets make a ball - curved manifold
void ball_grid(){
  Triangulation<2> tria;
  const Point<2> center(0, 0);
 // const Point<3> center(0, 0,0);
  const double   radius = 1.0;
  GridGenerator::hyper_ball(tria, center, radius);
  tria.reset_all_manifolds();//this kills hyper_ball's manifold assigmnent
  //By default, the manifold_id is set to 0 on the boundary faces, 1 on the boundary cells
  tria.set_all_manifold_ids_on_boundary(0);
  tria.set_manifold (0, SphericalManifold<2>(Point<2>(0, 0)));

  tria.refine_global(2); //powers of 2
  std::ofstream file_var("grid-ball.vtu");
  GridOut       grid_out;
  grid_out.write_vtu(tria, file_var);
  std::cout << "Grid written to grid-ball.vtu" << std::endl;
}

void shell_grid(){
  Triangulation<3> tria;
  const Point<3> center(1, 0,0);
  const double   inner_radius = 0.5, outer_radius = 1.0;
  GridGenerator::hyper_shell( tria, center, inner_radius, outer_radius,6);
  tria.refine_global(3); //powers of 2
  std::ofstream file_var("grid-shell.vtu");
  GridOut       grid_out;
  grid_out.write_vtu(tria, file_var);
  std::cout << "Grid written to grid-shell.vtu" << std::endl;
}

void shell_grid_manual_refinement(){
  Triangulation<3> triangulation;
  const Point<3> center(1, 0,0);
  const double   inner_radius = 0.5, outer_radius = 1.0;
  //remove last paramenter for 3d in hyper shell, otherwise feed 10
  GridGenerator::hyper_shell(triangulation, center, inner_radius, outer_radius);
  for (unsigned int step = 0; step < 3; ++step) {
      // Active cells are those that are not further refined
	  //we need to mark cells for refinement
      for (auto &cell : triangulation.active_cell_iterators()){
          // Next, we want to loop over all vertices of the cells.
          for (unsigned int v = 0; v < GeometryInfo<3>::vertices_per_cell; ++v){
              const double distance_from_center =center.distance(cell->vertex(v));
              //choose whatever refinement condition
              if (std::fabs(distance_from_center - outer_radius) < 1e-10){
                  cell->set_refine_flag();
                  break;
                }//if
            }//for
        }
      //refine global calls this function too
      triangulation.execute_coarsening_and_refinement();
    }//for steps loop

  std::ofstream file_var("grid-shell-manual.vtu");
  GridOut       grid_out;
  grid_out.write_vtu(triangulation, file_var);
  std::cout << "Grid written to grid-shell-manual.vtu" << std::endl;
}

int main(){
  cube_grid<2>();
  rectangle_grid();
  ball_grid();
 lshape_grid();
 shell_grid();
  shell_grid_manual_refinement();
}
