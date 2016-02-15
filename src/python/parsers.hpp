//
// Copyright (c) 2015-2016 CNRS
//
// This file is part of Pinocchio
// Pinocchio is free software: you can redistribute it
// and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version
// 3 of the License, or (at your option) any later version.
//
// Pinocchio is distributed in the hope that it will be
// useful, but WITHOUT ANY WARRANTY; without even the implied warranty
// of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// General Lesser Public License for more details. You should have
// received a copy of the GNU Lesser General Public License along with
// Pinocchio If not, see
// <http://www.gnu.org/licenses/>.

#ifndef __se3_python_parsers_hpp__
#define __se3_python_parsers_hpp__

#include <eigenpy/exception.hpp>
#include <eigenpy/eigenpy.hpp>

#include "pinocchio/python/model.hpp"
#include "pinocchio/python/data.hpp"

#ifdef WITH_URDFDOM
  #include "pinocchio/multibody/parser/urdf.hpp"
#ifdef WITH_HPP_FCL
  #include "pinocchio/python/geometry-model.hpp"
  #include "pinocchio/python/geometry-data.hpp"
  #include "pinocchio/multibody/parser/urdf-with-geometry.hpp"
#endif
#endif

#ifdef WITH_LUA
  #include "pinocchio/multibody/parser/lua.hpp"
#endif // #ifdef WITH_LUA

namespace se3
{
  namespace python
  {
    struct ParsersPythonVisitor
    {

      template<class T1, class T2>
      struct PairToTupleConverter {
        static PyObject* convert(const std::pair<T1, T2>& pair) {
          return boost::python::incref(boost::python::make_tuple(pair.first, pair.second).ptr());
        }
      };
      
      
#ifdef WITH_URDFDOM
      struct build_model_visitor : public boost::static_visitor<ModelHandler>
      {
        const std::string& _filename;

        build_model_visitor(const std::string& filename): _filename(filename){}

        template <typename JointModel> ModelHandler operator()(const JointModel & root_joint) const
        {
          Model * model = new Model();
          *model = se3::urdf::buildModel(_filename, root_joint);
          return ModelHandler(model,true);
        }
      };

      static ModelHandler buildModelFromUrdf(const std::string & filename,
                                             bp::object & root_joint_object
                                             )
      {
        JointModelVariant root_joint = bp::extract<JointModelVariant> (root_joint_object);
        return boost::apply_visitor(build_model_visitor(filename), root_joint);
      }

      static ModelHandler buildModelFromUrdf(const std::string & filename)
      {
        Model * model = new Model();
        *model = se3::urdf::buildModel(filename);
        return ModelHandler(model,true);
      }


#ifdef WITH_HPP_FCL
      typedef std::pair<ModelHandler, GeometryModelHandler> ModelGeometryHandlerPair_t;
      struct build_model_and_geom_visitor : public boost::static_visitor<std::pair<ModelHandler, GeometryModelHandler> >
      {
        const std::string& _filenameUrdf;
        const std::string& _filenameMeshRootDir;

        build_model_and_geom_visitor(const std::string& filenameUrdf,
                                     const std::string& filenameMeshRootDir): _filenameUrdf(filenameUrdf)
                                                                            , _filenameMeshRootDir(filenameMeshRootDir)
        {}

        template <typename JointModel>
        ModelGeometryHandlerPair_t operator() (const JointModel & root_joint) const
        {


          Model * model = new Model();
          GeometryModel * geometry_model = new GeometryModel();
          std::pair < Model, GeometryModel > models = se3::urdf::buildModelAndGeom(_filenameUrdf, _filenameMeshRootDir, root_joint);
          *model = models.first;
          *geometry_model = models.second;
          return std::pair<ModelHandler, GeometryModelHandler> ( ModelHandler(model, true),
                                                                 GeometryModelHandler(geometry_model, true)
                                                               );
        }
      };

      static ModelGeometryHandlerPair_t
      buildModelAndGeomFromUrdf(const std::string & filename,
                                const std::string & mesh_dir,
                                bp::object & root_joint_object
                                )
      {
        JointModelVariant root_joint = bp::extract<JointModelVariant> (root_joint_object);
        return boost::apply_visitor(build_model_and_geom_visitor(filename, mesh_dir), root_joint);
      }

      static ModelGeometryHandlerPair_t
      buildModelAndGeomFromUrdf(const std::string & filename,
                                const std::string & mesh_dir)
      {
        Model * model = new Model();
        GeometryModel * geometry_model = new GeometryModel();
        std::pair < Model, GeometryModel > models = se3::urdf::buildModelAndGeom(filename, mesh_dir);
        *model = models.first;
        *geometry_model = models.second;
        return std::pair<ModelHandler, GeometryModelHandler> ( ModelHandler(model, true),
                                                               GeometryModelHandler(geometry_model, true)
                                                             );
      }
#endif // #ifdef WITH_HPP_FCL
#endif // #ifdef WITH_URDFDOM

#ifdef WITH_LUA
      static ModelHandler buildModelFromLua(const std::string & filename,
                                            bool ff,
                                            bool verbose
                                            )
      {
        Model * model = new Model ();
        *model = se3::lua::buildModel (filename, ff, verbose);
        return ModelHandler (model,true);
      }
#endif // #ifdef WITH_LUA

      /* --- Expose --------------------------------------------------------- */
      static void expose();
    }; // struct ParsersPythonVisitor

    inline void ParsersPythonVisitor::expose()
    {
#ifdef WITH_URDFDOM
      
      
      bp::def("buildModelFromUrdf",
              static_cast <ModelHandler (*) (const std::string &, bp::object &)> (&ParsersPythonVisitor::buildModelFromUrdf),
              bp::args("Filename (string)","Root Joint Model"),
              "Parse the urdf file given in input and return a pinocchio model starting with the given root joint model"
              "(remember to create the corresponding data structure)."
              );
      
      bp::def("buildModelFromUrdf",
              static_cast <ModelHandler (*) (const std::string &)> (&ParsersPythonVisitor::buildModelFromUrdf),
              bp::args("Filename (string)"),
              "Parse the urdf file given in input and return a pinocchio model"
              "(remember to create the corresponding data structure)."
              );
      
#ifdef WITH_HPP_FCL
      
      bp::to_python_converter<std::pair<ModelHandler, GeometryModelHandler>, PairToTupleConverter<ModelHandler, GeometryModelHandler> >();
      
      bp::def("buildModelAndGeomFromUrdf",
              static_cast <ModelGeometryHandlerPair_t (*) (const std::string &, const std::string &, bp::object &)> (&ParsersPythonVisitor::buildModelAndGeomFromUrdf),
              bp::args("filename (string)", "mesh_dir (string)",
                       "Root Joint Model"),
              "Parse the urdf file given in input and return a proper pinocchio model starting with a given root joint and geometry model "
              "(remember to create the corresponding data structures).");
      
      bp::def("buildModelAndGeomFromUrdf",
              static_cast <ModelGeometryHandlerPair_t (*) (const std::string &, const std::string &)> (&ParsersPythonVisitor::buildModelAndGeomFromUrdf),
              bp::args("filename (string)", "mesh_dir (string)"),
              "Parse the urdf file given in input and return a proper pinocchio model and geometry model "
              "(remember to create the corresponding data structures).");
      
#endif // #ifdef WITH_HPP_FCL
#endif // #ifdef WITH_URDFDOM
      
#ifdef WITH_LUA
      bp::def("buildModelFromLua",buildModelFromLua,
              bp::args("Filename (string)",
                       "Free flyer (bool, false for a fixed robot)",
                       "Verbose option "),
              "Parse the urdf file given in input and return a proper pinocchio model "
              "(remember to create the corresponding data structure).");
#endif // #ifdef WITH_LUA
    }
    
  }
} // namespace se3::python

#endif // ifndef __se3_python_parsers_hpp__