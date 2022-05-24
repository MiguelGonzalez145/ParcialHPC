#include "regresionlineal.h"
#include "Extraccion/extraer.h"
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <eigen3/Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include <string.h>

int main(int argc, char *argv[]) {

    /* Se crea un objeto del tipo Extraer
     * para incluir los 3 argumentos que necesita el objeto. */
    Extraer extraerData(argv[1], argv[2], argv[3]);
    /*Se crea un objeto del tipo LinealRegresion, sin ningún argumento de entrada*/
    RegresionLineal LR;

    /* Se requiere probar la lectura del fichero y luego se requiere observar el dataset como un objeto
     * de matriz tipo dataFrame */

    std::vector<std::vector<std::string>> dataSET = extraerData.ReadCSV();
    int filas = dataSET.size()+1;
    int columnas = dataSET[0].size();
    Eigen::MatrixXd MatrizDATAF = extraerData.CSVtoEigen(dataSET,filas,columnas);

    /*Se imprime la matriz que contiene los datos del dataSET */
    std::cout << MatrizDATAF << std:: endl;
    std::cout << "Filas: " <<filas<<std:: endl;
    std::cout << "Columnas: "<<columnas<<std::endl;
    /* se imrpime el Promedio, se debe validar */
    //std::cout<< extraerData.Promedio(MatrizDATAF) << std::endl;


    /* se crea la matrix para almacenar la normalización*/


    Eigen::MatrixXd matNormal = extraerData.Normalizador(MatrizDATAF);
    //std::cout<< matNormal <<std::endl;

    /*A continuación se divide el entrenamiento y prueba en conjunto de datos de entrada (matNormal) */

    Eigen::MatrixXd X_test, y_test, X_train, y_train;

    /*Se dividen los datos y el 80% es para entrenamiento */

    std::tuple<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd> MatrizDividida=extraerData.TrainTestSplit(matNormal,0.797);

    /*Se desempaqueta la tupla*/
    std::tie(X_train, y_train, X_test, y_test) =MatrizDividida;

    //std::cout<<matNormal.rows()<<std::endl;
    //std::cout<<X_train.rows()<<std::endl;
    //std::cout<<X_test.rows()<<std::endl;



    /*A continuación se hará el primer modulo de ML. Se hará una clase RegresionLineal. Con su correspondiente constructor
     * de argumentos de entrada y metodos para el calculo del modelo RL. Se tiene en cuenta que el RL es un metodo estadistico
     * que define la relación entre las variables independientes y la dependiente.
     * La idea principal es definir una linea recta(HiperPlano) con sus coeficientes(Pendientes) y punto de corte.
     * Se tienen diferentes metodos para resolver RL, para este caso se usará el metodo de los Minimos Cuadrados Ordinarios(OLS), por ser un metodo
     * sencillo, y computacionalmente económico.
     * Representa una solución optima para conjunto de datos no complejos. EL DataSet a utilizar es el de VinoRojo, el cual tiene 11 variables (Multivariable)
     * independientes. Para ello hemos de implementar el algoritmo del gradiente descendiente, cuyo objetivo principal es minimizar la función de costo.*/



    /*Se define un valor para entrenamiento y para prueba inicializados en unos*/
    Eigen::VectorXd VectorTrain = Eigen::VectorXd::Ones(X_train.rows());
    Eigen::VectorXd VectorTest = Eigen::VectorXd::Ones(X_test.rows());

    /*Se redimensionan las matrices para ser ubicadas en el vector de Unos: similar al Numpy reshape*/
    X_train.conservativeResize(X_train.rows(),X_train.cols()+1);
    X_train.col(X_train.cols()-1)=VectorTrain;

    X_test.conservativeResize(X_test.rows(),X_test.cols()+1);
    X_test.col(X_test.cols()-1)=VectorTest;

    /*Se define el vector theta que se pasará al algoritmo del gradientes descendiente. Básicamente es un vector de ceros del mismo
     * tamaño del entrenamiento, adicionalmente se pasará alpha y el número de iteraciones */
    Eigen::VectorXd theta = Eigen::VectorXd::Zero(X_train.cols()); /*theta:coeficientes*/
    float alpha = 0.01;
    int iteraciones = 500;
    /*A continuación se definen las variables de salida que representan los coeficientes y el vector de costo*/
    Eigen::VectorXd thetaSalida;
    std::vector<float> costo;

    /*Se desempaqueta la tupla como objeto instanciado del gradiente descendiente*/
    std::tuple<Eigen::VectorXd, std::vector<float>> objetoGradiente = LR.GradienteDes(X_train, y_train, theta, alpha, iteraciones);
    std::tie(thetaSalida, costo) = objetoGradiente;

    /*Se imprime los coeficientes para cada variable*/
    //std::cout<<thetaSalida<<std::endl;


    /* Se imprime para inspeccion ocular la funcion de costo*/

    for(auto v: costo){
        //std::cout<<v<<std::endl;
    }

    /*Se almacena la funcion de costo y las variables Theta a ficheros */

    extraerData.VectorToFile(costo, "costo.txt");
    extraerData.EigentoFile(thetaSalida, "theta.txt");

    /*Se calcula el promedio y la desviación estandar para calcular las predicciones, es decir, se debe de normalizar para calcular
     * la metrica*/

    auto muData = extraerData.Promedio(MatrizDATAF);
    auto muFeatures = muData(0,13);
    auto escalado = MatrizDATAF.rowwise()-MatrizDATAF.colwise().mean();
    auto sigmaData = extraerData.DesvStandar(escalado);
    auto sigmaFeatures = sigmaData(0,13);

    Eigen::MatrixXd y_train_hat = (X_train*thetaSalida*sigmaFeatures).array()+muFeatures;
    Eigen::MatrixXd y = MatrizDATAF.col(0).topRows(404);


    float R2_score = extraerData.R2_score(y, y_train_hat);

    std::cout<<R2_score<<std::endl;

    extraerData.EigentoFile(y_train_hat, "y_train_hatCpp.txt");

    return EXIT_SUCCESS;
}
