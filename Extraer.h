#ifndef EXTRAER_H
#define EXTRAER_H

#include <iostream>
#include <vector>
#include <fstream>
#include <eigen3/Eigen/Dense>

class Extraer
{
        /*Se presenta el constructor de los argumentos de entrada a la clase extraer */

        /*Nombre del dataset */
        std::string setDatos;
        /*Separador de columnas*/
        std::string delimitador;
        /*Si tiene cabecera o no, el dataset */
        bool header;


public:

        Extraer(std::string datos,
                std::string separador,
                bool head):
            setDatos(datos),
            delimitador(separador),
            header(head){}

    std::vector<std::vector<std::string>> ReadCSV();
     Eigen::MatrixXd CSVtoEigen(std::vector<std::vector<std::string>> setDatos,int filas, int columnas);

     /* Función paraq calcular el promedio */
     /* En c++ la herencia del tipo de dato no es directa(sobre tod si es  partir de funciones dadas por otras interfaces/clases/bibliotecas
      * :EIGEN, shrkml, etc).
      * Se declara el tipo en una expresión "decltype" con el fin de tener seguridad de qué tipo de dato retornará la función*/


     auto Promedio(Eigen::MatrixXd datos) ->decltype(datos.colwise().mean());

     /* Función para calcular la Desviación Estandar */
     /* Para implementar la desviación estadar, datos = xi - x.promedio*/

     auto DesvStandar(Eigen::MatrixXd datos) ->decltype((datos.array().square().colwise().sum()/(datos.rows()-1)).sqrt());

     Eigen::MatrixXd Normalizador(Eigen::MatrixXd datos);

     std::tuple<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd>TrainTestSplit(Eigen::MatrixXd DatosNorm, float sizetrain );

     void VectorToFile(std::vector<float> dataVector,std::string fileName);

     void EigentoFile(Eigen::MatrixXd DataMatrix,std::string fileName);

     float R2_score(Eigen::MatrixXd y, Eigen::MatrixXd y_hat);

};

#endif // EXTRAER_H
