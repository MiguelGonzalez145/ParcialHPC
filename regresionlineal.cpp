#include "regresionlineal.h"
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <eigen3/Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include <string.h>

/*Se necesita entrenar el modelo, lo que implica minimizar alguna función de costo(Para este caso se ha seleccionado para funcion
 * de costo OLS) y de esta forma se puede medir la función de hipotesis. Una función de costo es la forma de penalizar al modelo por
 * cometer un error. Se implementauna función del tipó flotante, que toma como entradas "X" y "Y", y los valores de Theta inicializados
 * (los valores de Theta se fijan inicialmente en cualquier valor para que al iterar segun un alpha, consiga el menor valor para la
 * función de costo*/

float RegresionLineal::fCosto(Eigen::MatrixXd X,
                              Eigen::MatrixXd y,
                              Eigen::MatrixXd theta)
{
    /*Se almacena la diferencia elevada al cuadrado(Funcion de hipotesis que representa el error)*/
    Eigen::MatrixXd diferencia = pow((X* theta-y).array(),2);
    return (diferencia.sum()/(2*X.rows()));
}

    /*Se necesita proveer al programa una función para dar al algoritmo un valor inicial para theta, el cual va a cambiar iterativamente
     * hasta que converja al valor mínimo de la función de costo. Basicamente describe el Gradiente Descendiente: La idea es calcular
     * el gradiente para la función de costo dado por la derivada parcial. La función tendrá un alpha que representa el salto del
     * gradiente. La función tiene como entrada "X", "y", "theta", "alpha", y el número de iteraciones que necesita theta actualizada
     * cada vez para que
     * la función converja*/

std::tuple<Eigen::VectorXd, std::vector<float>> RegresionLineal::GradienteDes(Eigen::MatrixXd X,
                                                                              Eigen::MatrixXd y,
                                                                              Eigen::VectorXd theta,
                                                                              float alpha,
                                                                              int iteraciones){
    /*Se almacenan temporalmente los parametros de theta*/
    Eigen::MatrixXd tempTheta= theta;
    /*Se extrae la cantidad de parametros */
    int parametros = theta.rows();
    /*Valores de costo inicial, se actualizará cada vez con los nuevos pesos*/
    std::vector<float> costo;
    costo.push_back(fCosto(X,y,theta));

    /*Para cada iteración se calcula la función de error. Se multiplica cada feature (x) que calcula el error y se almacena en una
     * variable temporal, luego se actualiza theta y se calcula de nuevo la funcion de costo basada en el nuevo valor de theta*/

    for(int i=0;i<iteraciones;i++){
        Eigen::MatrixXd Error = X*theta-y;
            for(int j=0;j<parametros;j++){
                    Eigen::MatrixXd X_i = X.col(j);
                    Eigen::MatrixXd tempError = Error.cwiseProduct(X_i);
                    tempTheta(j,0) = theta(j,0)-((alpha/X.rows())*(tempError.sum()));
            }
            theta = tempTheta;
            costo.push_back(fCosto(X,y,theta));
    }
    /*Se empaqueta la tupla y se retorna*/
    return std::make_tuple(theta,costo);
}
