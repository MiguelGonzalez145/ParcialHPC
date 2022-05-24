#ifndef REGRESIONLINEAL_H
#define REGRESIONLINEAL_H

#include <iostream>
#include <vector>
#include <fstream>
#include <eigen3/Eigen/Dense>
class RegresionLineal
{
public:
    RegresionLineal(){}

    float fCosto(Eigen::MatrixXd X,Eigen::MatrixXd y, Eigen::MatrixXd theta);

    std::tuple<Eigen::VectorXd, std::vector<float>> GradienteDes(Eigen::MatrixXd X,Eigen::MatrixXd y,Eigen::VectorXd theta,float alpha,int iteraciones);


    };

#endif // REGRESIONLINEAL_H
