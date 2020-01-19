//
// Created by Piotr on 14.09.2019.
//

#ifndef OeSNN_ESNN_H
#define OeSNN_RTAD_ESNN_H

#include "iostream"
#include "algorithm"
#include "vector"
#include "fstream"
#include "sstream"
#include "math.h"
#include "chrono"
#include "random"
#include "iomanip"

using namespace std;

struct neuron
{
    int ID;
    vector<vector<double>> s_weights;
    double outputValue;
    double M;
    double PSP;
    double additionTime;
}; //output neuron structure

struct GRFstruct
{
    double mu, sigma, exc;
    int rank;
}; //struct of GRF

struct inputNeuron
{
    int id;
    int FA_k;
    double firingTime;
    vector<int> order = *(new vector<int>);
}; //input neuron structure

extern vector<int> CNOsize;
extern int Wsize;
extern int NOsize;
extern int NIsize;
extern double simTr;
extern double mod;
extern int Ninit, Nsize;

extern int n;
extern int m;
extern double H;
extern int K;

extern vector<vector<neuron *>> OutputNeurons; //Pointers to output neurons (output neuron repositories)
extern vector<vector<double >> X; //input data streams
extern vector<vector<double >> Y; //predicted values of pollution level
extern vector<int> IDS;

extern vector<vector<GRFstruct>> GRFs; //input GRFs
extern vector<vector<inputNeuron>> InputNeurons; //firing order of input neurons for current X[t]

void LoadData(string fileName);
int CountInstances(string fileName);

void InitializeInputLayer(const vector<vector<double>> &Windows);
void InitializeNeuron(neuron* n_c, double x_h, int h);
void UpdateRepository(neuron* n_c, int F_k);
void InitializeNetwork(vector<vector<double>> &Windows);
double PredictValue(int F_k);


void PredictOeSNN(); //main procedure of eSNN-RTAD

void SaveResults(string filePath);
void ClearStructures();
void SaveMetrics(string, double, double, double, double);

void CalculateRMSE();

#endif //ESNN_RTAD_ESNN_H