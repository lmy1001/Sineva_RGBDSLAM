#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <chrono>
#include "parameter_server"
#include "scoped_timer.h"

using namespace std;
using namespace cv;
using namespace g2o;
// g2o edge
class EdgeProjectXYZRGBDPoseOnly : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeProjectXYZRGBDPoseOnly( const Eigen::Vector3d& point ) : _point(point) {}

    virtual void computeError()
    {
        const g2o::VertexSE3Expmap* pose = static_cast<const g2o::VertexSE3Expmap*> ( _vertices[0] );
        // measurement is p, point is p'
        _error = _measurement - pose->estimate().map( _point );
    }

    virtual void linearizeOplus()
    {
        g2o::VertexSE3Expmap* pose = static_cast<g2o::VertexSE3Expmap *>(_vertices[0]);
        g2o::SE3Quat T(pose->estimate());
        Eigen::Vector3d xyz_trans = T.map(_point);
        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double z = xyz_trans[2];

        _jacobianOplusXi(0,0) = 0;
        _jacobianOplusXi(0,1) = -z;
        _jacobianOplusXi(0,2) = y;
        _jacobianOplusXi(0,3) = -1;
        _jacobianOplusXi(0,4) = 0;
        _jacobianOplusXi(0,5) = 0;

        _jacobianOplusXi(1,0) = z;
        _jacobianOplusXi(1,1) = 0;
        _jacobianOplusXi(1,2) = -x;
        _jacobianOplusXi(1,3) = 0;
        _jacobianOplusXi(1,4) = -1;
        _jacobianOplusXi(1,5) = 0;

        _jacobianOplusXi(2,0) = -y;
        _jacobianOplusXi(2,1) = x;
        _jacobianOplusXi(2,2) = 0;
        _jacobianOplusXi(2,3) = 0;
        _jacobianOplusXi(2,4) = 0;
        _jacobianOplusXi(2,5) = -1;
    }

    bool read ( istream& in ) {}
    bool write ( ostream& out ) const {}
protected:
    Eigen::Vector3d _point;
};               //23-71 copy from gaoxiang-slambook-pose_estimation_3d3d.cpp



int main(int argc, char* argv){

    vector<Point3f> pts1,pts2; //  Assume pts1,pts2 are known with matching

    // use SVD to get a initial pose of camera
    Point3f p1={0,0,0};
    Point3f p2={0,0,0};

    for (int i=0;i<pts1.size();i++){
        p1=p1+pts1[i];
    }

    for(int i=0;i<pts2.size();i++){
        p2=p2+pts2[i];
    }

    int N=pts1.size();
    p1=p1/N;
    p2=p2/N;

    Mat W, E,U, Vt;

    vector<Point3f> q1(N), q2(N);
    for (int i = 0; i<N; i++){
            q1[i] = pts1[i] - p1;
            q2[i] = pts2[i] - p2;
    }
    W = { 0 };
    for (int i = 0; i<N; i++){
            W.at<double>(1, 1) = W.at<double>(1, 1) + q1[i].x*q2[i].x;
            W.at<double>(1, 2) = W.at<double>(1, 2) + q1[i].x*q2[i].y;
            W.at<double>(1, 3) = W.at<double>(1, 3) + q1[i].x*q2[i].z;
            W.at<double>(2, 1) = W.at<double>(2, 1) + q1[i].y*q2[i].x;
            W.at<double>(2, 2) = W.at<double>(2, 2) + q1[i].y*q2[i].y;
            W.at<double>(2, 3) = W.at<double>(2, 3) + q1[i].y*q2[i].z;
            W.at<double>(3, 1) = W.at<double>(3, 1) + q1[i].z*q2[i].x;
            W.at<double>(3, 2) = W.at<double>(3, 2) + q1[i].z*q2[i].y;
            W.at<double>(3, 3) = W.at<double>(3, 3) + q1[i].z*q2[i].z;
    }

    SVD::compute(E, W, U, Vt);
    cout<<"U= "<<U<<endl;
    cout<<"V= "<<V<<endl;


    Mat Vtranspose;
    transpose(Vt, Vtranspose);
    Mat R;
    R=U*Vtranspose;
    cout<<"R= "<<R<<endl;

    Mat t;
    Mat p1_m(p1);
    Mat p2_m(p2);
    t=p1_m-R*p2_m;
    cout<<"t= "<<t<<endl;

    cout<<"R from first frame to second ="<<R.t()<<endl;
    cout<<"t from first frame to second ="<<R.t()*t<<endl;



    //use g2o to optimize
    SparseOptimizer optimizer;
    optimizer.setVerbose(false);

    g2o::BlockSolver_6_3::LinearSolverType* linearSolver=new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();
    g20::BlockSolver_6_3* block_solver=new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizerAlgorithmLevenberg* solver=new g2o::OptimizationAlgorithmLevenberg(block_solver);

    optimizer.setAlgorithm(solver);

    int vertex_index=0;

    //initial pose of camera
    //Vector3d t(0,0,0);
    //Quaterniond q;
    //q.setIdentity();

    //Eigen::Isometry3d cam;
    //cam=q;
    //cam.translation()=t;

    g2o::VertexSE3Expmap* pose=new g2o::VertexSE3Expmap();

    Eigen::Matrix3d R_mat;
    R_mat<<
            R.at<double>(0,0), R.at<double>(0,1),R.at<double>(0,2),
            R.at<double>(1,0), R.at<double>(1,1),R.at<double>(1,2),
            R.at<double>(2,0), R.at<double>(2,1),R.at<double>(2,2);

    Eigen::Vector3d t_mat;
    t_mat<<
            t.at<double>(0,0), t.at<double>(0,1), t.at<double>(0,2);
    pose->setEstimate(g2o::SE3Quat(R_mat,t_mat));

    pose->setId(vertex_index);

    optimizer.addVertex(pose);

    vertex_index++;


    //define vertexs
    for (int i=0;i<pts1.size();i++)
    {
        g2o::VertexSEBAPointXYZ* vp=new g2o::VertexSBAPointXYZ();

        vp->setId(vertex_index);
        vp->setMarginalized(true);
        vp->setEstimate(pts1[i]);

        optimizer.addVertex(vp);
        vertex_index++;
    }

    //define edges
    vector<EdgeProjectXYZRGBDPoseOnly*> edges;
    int edge_index=1;
    for(int i=0;i<pts1.size();i++)
    {
        g2o::EdgeProjectXYZRGBDPoseOnly* edge=new g2o::EdgeProjectXYZRGBDPoseOnly(Eigen::Vector3d(pts2[i].x, pts2[i].y, pts2[i].z));
        edge->setId(edge_index);
        edge->setVertex(0,dynamic_cast<g2o::VertexSE3Expmap*> pose);

        edge->setMeasurement(Eigen::Vector3d(pts1[i].x,pts1[i].y,pts1[i].z));
        edge->setInformation(Eigen::Matrix3d::Identity());

        edge->setRobustKernel(new g2o::RobustKernelHuber());
        optimizer.addEdge(edge);
        index++;
        edges.push_back(edge);

      }                  //from line 193 to line 202 are copied from gaoxiang-slambook-pose_estimation_3d3d.cpp

    //begin optimize
    cout<<"begin optimizing"<<endl;
    optimizer.setVerbose(true);
    optimizer.initilizeOptimization();
    optimizer.optimize(100);
    cout<<"end optimizing"<<endl;
    cout<<"Pose="<< Eigen::Isometry3d( pose->estimate() ).matrix()<<endl;

        return 0;
    }
