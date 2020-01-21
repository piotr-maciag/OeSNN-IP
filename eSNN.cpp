
#include "eSNN.h"

int datasetSize;

vector<int> CNOsize;
int Wsize;
int NOsize;
int NIsize;
double simTr;
double mod;
//int K;

vector<vector<double>> Wstream;

int Nn = 2;
int m;
double H;

double Dub;
vector<double> I_min;
vector<double> I_max;
int Ninit, Nsize;


vector<neuron *> OutputNeurons; //Pointers to output neurons (output neuron repositories)
vector<vector<double >> X; //input data streams
vector<vector<double >> Y; //predicted values of pollution level
vector<int> IDS;

vector<vector<GRFstruct *>> GRFs; //input GRFs
vector<vector<inputNeuron *>> InputNeurons;


bool compFiringTime(const inputNeuron &nI1, const inputNeuron &nI2) { //comparator of firing times
    if (nI1.firingTime != nI2.firingTime) {
        return nI1.firingTime < nI2.firingTime;
    } else {
        return nI1.id < nI2.id;
    }
}


void InitializeInputLayerGRFs(const vector<vector<double>> &Windows) { //intialize input layer of OeSNN

    for (int k = 0; k < Windows.size(); k++) {
        for (int j = 0; j < InputNeurons[k].size(); j++) {
            InputNeurons[k][j]->order.clear();
        }
    }

    for (int k = 0; k < Windows.size(); k++) {
        int ord = 0;
        if (I_max[k] < Windows[k][Windows[k].size() - 1]) {
            I_max[k] = Windows[k][Windows[k].size() - 1];
        }
        if (I_min[k] > Windows[k][Windows[k].size() - 1]) {
            I_min[k] = Windows[k][Windows[k].size() - 1];
        }

        // double width = (I_max[k] - I_min[k]) / NIsize;

        for (int j = 0; j < GRFs[k].size(); j++) {
            double mu = I_min[k] + ((2.0 * j - 3.0) / 2.0) * ((I_max[k] - I_min[k]) / (double(NIsize) - 2));
            double sigma = (1.0 / 1.0) * (((I_max[k] - I_min[k]) / (double(NIsize) - 2)));

            GRFs[k][j]->mu = mu;
            GRFs[k][j]->sigma = sigma;
        }


        for (int u = Windows[k].size() - 1; u >= 0; u--) {
            vector<inputNeuron> sortInputNeurons;
            for (int j = 0; j < GRFs[k].size(); j++) {

                if (GRFs[k][j]->sigma == 0.0) {
                    GRFs[k][j]->sigma = 1.0;
                }
                double exc = (exp(-0.5 * pow(((Windows[k][u] - GRFs[k][j]->mu) / GRFs[k][j]->sigma), 2)));
                double firingTime = /*floor*/(1 - exc);

                inputNeuron newIN = {j, k, firingTime};
                sortInputNeurons.push_back(newIN);
            }

            sort(sortInputNeurons.begin(), sortInputNeurons.end(), compFiringTime);


            for (int j = 0; j < sortInputNeurons.size(); j++) {
                InputNeurons[k][sortInputNeurons[j].id]->order.push_back(ord);
                ord++;
            }

        }


    }
}

void InitializeInputLayer(const vector<vector<double>> &Windows) { //intialize input layer of OeSNN

    for (int k = 0; k < Windows.size(); k++) {
        for (int j = 0; j < InputNeurons[k].size(); j++) {
            InputNeurons[k][j]->order.clear();
        }
    }

    for (int k = 0; k < InputNeurons.size(); k++) {

        //  cout << k << endl;
        if (I_max[k] < Windows[k][Windows[k].size() - 1]) {
            I_max[k] = Windows[k][Windows[k].size() - 1];
        }
        if (I_min[k] > Windows[k][Windows[k].size() - 1]) {
            I_min[k] = Windows[k][Windows[k].size() - 1];
        }

        double width = (I_max[k] - I_min[k]) / NIsize;

        for (int j = 0; j < GRFs[k].size(); j++) {
            double mu = I_min[k] + (j + 1 - 0.5) * width;
            GRFs[k][j]->mu = mu;
        }


        for (int u = Windows[k].size() - 1; u >= 0; u--) {

            int j;
            if (Windows[k][u] != I_max[k]) {
                j = floor((Windows[k][u] - I_min[k]) / width) + 1;
            } else {
                j = NIsize;
            }
            int l;
            if (j - 1 < NIsize - j) { l = j - 1; } else { l = NIsize - j; }
            // cout << Windows[k][u] << " " << I_min[k] << " " << width << " " << floor((Windows[k][u] - I_min[k]) / width) << " j:: " << j - 1<< endl;

            GRFs[k][j - 1]->rank = 0;

            if (Windows[k][u] < GRFs[k][j - 1]->mu) {
                for (int n = 1; n <= l; n++) {
                    //cout << "-j: " << j - n - 1 << endl;
                    //cout << "-j: " << j + n - 1 << endl;
                    GRFs[k][j - n - 1]->rank = 2 * n - 1;
                    GRFs[k][j + n - 1]->rank = 2 * n;
                }
                for (int n = 1; n <= j - 1 - l; n++) //n is k in algorithms
                {
                    // cout << "-j: " << j - l - n - 1 << endl;
                    GRFs[k][j - l - n - 1]->rank = 2 * l - 1 + n;
                }
                for (int n = 1; n <= NIsize - j - l; n++) //n is k in algorithms
                {
                    //cout << "-j: " << j + l + n - 1<< endl;
                    GRFs[k][j + l + n - 1]->rank = 2 * l + n;
                }
            } else {
                for (int n = 1; n <= l; n++) {
                    // cout << "+j: " << j - n - 1 << endl;
                    //  cout << "+j: " << j + n - 1 << endl;
                    GRFs[k][j - n - 1]->rank = 2 * n;
                    GRFs[k][j + n - 1]->rank = 2 * n - 1;
                }
                for (int n = 1; n <= j - 1 - l; n++) //n is k in algorithms
                {
                    // cout << "+j: " << j - l - n - 1 << endl;
                    GRFs[k][j - l - n - 1]->rank = 2 * l + n;
                }
                for (int n = 1; n <= NIsize - j - l; n++) //n is k in algorithms
                {
                    // cout << "+j: " << j + l + n << endl;
                    GRFs[k][j + l + n - 1]->rank = 2 * l + n - 1;
                }
            }

            for (int j = 0; j < GRFs[k].size(); j++) {
                int rank = GRFs[k][j]->rank + (Wsize - u - 1) * NIsize;
                //cout << " j: " << j << " " << rank << " - ";
                InputNeurons[k][j]->order.push_back(rank);
                //<< InputNeurons[k][j].order[InputNeurons[k][j].order.size() -1];
            }
            //cout << endl;

        }
    }


}

void InitializeNeuron(neuron *n_c, double x_h, int h) { //Initalize new neron n_i

    for (int l = 0; l < InputNeurons.size(); l++) {
        vector<double> vec;
        n_c->s_weights.push_back(vec);
        for (int j = 0; j < InputNeurons[l].size(); j++) {
            n_c->s_weights[l].push_back(0);
        }
    }

    for (int l = 0; l < InputNeurons.size(); l++) {
        for (int j = 0; j < InputNeurons[l].size(); j++) {
            for (int u = 0; u < Wsize; u++) {
                n_c->s_weights[l][j] += pow(mod, InputNeurons[l][j]->order[u]);
            }
        }
    }


    n_c->outputValue = x_h;
    n_c->M = 1;
    n_c->additionTime = h;
    n_c->ID = h;

}

void InitializeNetwork(vector<vector<double>> &Windows) {

    for (int h = Wsize; h < Ninit; h++) {

        InitializeInputLayer(Windows);
        //InitializeInputLayerGRFs(Windows);

        //cout << "h: " << h << endl;



            neuron *n_c = new neuron;
        InitializeNeuron(n_c, X[0][h], h);
        UpdateRepository(n_c);


        for (int k = 0; k < Nn; k++) {
            Windows[k].erase(Windows[k].begin());
            Windows[k].push_back(X[k][h]);
        }
    }
}

double
CalculateDistance(const vector<vector<double>> &w1,
                  const vector<vector<double>> &w2) { //calculate distance between two weights vectors
    double diffSq = 0.0;

    for (int k = 0; k < w1.size(); k++) {
        for (int j = 0; j < w1.size(); j++) {
            diffSq += pow(w1[k][j] - w2[k][j], 2);
        }
    }

    return sqrt(diffSq);
}

neuron *FindMostSimilar(neuron *n_c) { //find mos similar neurons in terms of synaptic weights

    double minDist = CalculateDistance(n_c->s_weights, OutputNeurons[0]->s_weights);
    double minIdx = 0;

    if (OutputNeurons.size() > 1) {
        for (int i = 1; i < OutputNeurons.size(); i++) {
            double dist = CalculateDistance(n_c->s_weights, OutputNeurons[i]->s_weights);
            if (dist < minDist) {
                minDist = dist;
                minIdx = i;
            }
        }
    }
    return OutputNeurons[minIdx];
}

void ReplaceOldest(neuron *n_c) { //replace the oldets neuron in output repostiory
    int oldest = OutputNeurons[0]->additionTime;
    int oldestIdx = 0;

    for (int i = 1; i < OutputNeurons.size(); i++) {
        if (oldest > OutputNeurons[i]->additionTime) {
            oldest = OutputNeurons[i]->additionTime;
            oldestIdx = i;
        }
    }

    delete OutputNeurons[oldestIdx];
    OutputNeurons[oldestIdx] = n_c;

}


void UpdateRepository(neuron *n_c) { //Update neuron n_s in output repository

    neuron *n_s;

    if (OutputNeurons.size() > 0) {
        n_s = FindMostSimilar(n_c);
    }

    if (OutputNeurons.size() > 0 && CalculateDistance(n_c->s_weights, n_s->s_weights) < simTr * Dub) {
        for (int k = 0; k < n_s->s_weights.size(); k++) {
            for (int j = 0; j < n_s->s_weights[k].size(); j++) {
                n_s->s_weights[k][j] = (n_c->s_weights[k][j] + n_s->s_weights[k][j] * n_s->M) / (n_s->M + 1);
            }
        }

        n_s->outputValue = (n_c->outputValue + n_s->outputValue * n_s->M) / (n_s->M + 1);
        n_s->additionTime = (n_c->additionTime + n_s->additionTime * n_s->M) / (n_s->M + 1);
        n_s->M += 1;
        delete n_c;
    } else if (OutputNeurons.size() < NOsize) {
        OutputNeurons.push_back(n_c);
    } else {
        ReplaceOldest(n_c);
    }
}

double PredictValue() {
    for (int i = 0; i < OutputNeurons.size(); i++) {
        OutputNeurons[i]->PSP = 0;
    }

    for (int l = 0; l < InputNeurons.size(); l++) {
        for (int j = 0; j < InputNeurons[l].size(); j++) {
            for (int i = 0; i < OutputNeurons.size(); i++) {
                for (int u = 0; u < Wsize; u++) {
                    OutputNeurons[i]->PSP += OutputNeurons[i]->s_weights[l][j] *
                                             pow(mod, InputNeurons[l][j]->order[u]);
                }
            }
        }
    }


    double maxPSP = -1;
    double maxVals;
    int countMax = 0;

    for (int i = 0; i < OutputNeurons.size(); i++) {
        if (maxPSP < OutputNeurons[i]->PSP) {
            maxVals = OutputNeurons[i]->outputValue;
            countMax = 1;
            maxPSP = OutputNeurons[i]->PSP;
        } else if (maxPSP == OutputNeurons[i]->PSP) {
            maxVals += OutputNeurons[i]->outputValue;
            countMax++;
        }
    }

    return (maxVals / (double) countMax);
}

double CalculateUpperBound() {
    vector<double> v;

    for (int k = 0; k < InputNeurons.size(); k++) {
        for (int j = 0; j < InputNeurons[k].size(); j++) {

            double sum = 0;
            for (int u = 0; u < Wsize; u++) {

                sum += pow(mod, j + NIsize * u) - pow(mod, NIsize - j - 1 + NIsize * u);
                //sum += pow(mod, j) - pow(mod, NIsize - j - 1);
            }
            v.push_back(sum);
        }
    }

    double diffSq = 0.0;

    for (int j = 0; j < v.size(); j++) {
        diffSq += pow(v[j], 2);
    }

    return sqrt(diffSq);
}


void PredictOeSNN() { //main eSNN procedure

    for (int k = 0; k < Nn; k++) {
        CNOsize.push_back(0);
    }

    for (int k = 0; k < Nn; k++) {
        vector<GRFstruct *> GRFSvec;
        for (int j = 0; j < NIsize; j++) {
            GRFstruct *newGRF = new GRFstruct;
            GRFSvec.push_back(newGRF);
        }
        GRFs.push_back(GRFSvec);
    }

    for (int k = 0; k < Nn; k++) {
        vector<inputNeuron *> InputNeuronsVect;
        for (int j = 0; j < NIsize; j++) {
            inputNeuron *newInputNeuron = new inputNeuron{j, k, 0};
            InputNeuronsVect.push_back(newInputNeuron);
        }
        InputNeurons.push_back(InputNeuronsVect);
    }


    vector<vector<double>> Windows;

    for (int k = 0; k < Nn; k++) {
        vector<double> win;
        for (int h = 0; h < Wsize; h++) {
            win.push_back(X[k][h]);
        }
        Windows.push_back(win);
        Wstream.push_back(win);
    }

    for (int h = Wsize; h < Ninit; h++) {
        for (int k = 0; k < Nn; k++) {
            Wstream[k].push_back(X[k][h]);
        }
    }

    for (int k = 0; k < Wstream.size(); k++) {
        I_max.push_back(*max_element(Wstream[k].begin(), Wstream[k].end()));
        I_min.push_back(*min_element(Wstream[k].begin(), Wstream[k].end()));

        cout << "IMax " << I_max[k] << " " << "Imin " << I_min[k] << endl;
    }

    Dub = CalculateUpperBound();

    InitializeNetwork(Windows);

    vector<vector<double>> WP;

    for (int k = 0; k < Windows.size(); k++) {
        vector<double> vec;
        WP.push_back(vec);
    }

    cout << "Phase 1" << endl;

    for (int h = Ninit; h < Nsize - H; h++) {

        if (h % 100 == 0)
            cout << "#";


        InitializeInputLayer(Windows);
        //InitializeInputLayerGRFs(Windows);


        neuron *n_c = new neuron;
        InitializeNeuron(n_c, X[0][h], h);
        UpdateRepository(n_c);


        for (int k = 0; k < Nn; k++) {
            Windows[k].erase(Windows[k].begin());
            Windows[k].push_back(X[k][h]);
            WP[k] = Windows[k];
        }

        vector<double> y;
        y.push_back(h);

        for (int hpred = 1; hpred <= H; hpred++) {
            InitializeInputLayer(WP);
            //InitializeInputLayerGRFs(WP);

            double y_h_hpred = PredictValue();
            WP[0].erase(WP[0].begin());
            WP[0].push_back(y_h_hpred);

            y.push_back(y_h_hpred);

            for (int k = 1; k < Nn; k++) {

                WP[k].erase(WP[k].begin());
                WP[k].push_back(X[k][h + hpred]);
            }

        }
        Y.push_back(y);
    }
}


///////////////////////////////////////////////////////
//Print Results Procedures

void SaveResults(string filePath) {
    fstream handler;
    handler.open(filePath, iostream::out);

    for (int i = 0; i < Y[0].size() - 1; i++) {
        handler << "h+" << i << ",";
    }
    handler << "h+" << Y[0].size() - 1;
    handler << endl;

    for (int i = 0; i < Y.size(); i++) {
        for (int j = 0; j < Y[i].size() - 1; j++) {
            handler << setprecision(12) << Y[i][j] << ",";
        }
        handler << setprecision(12) << Y[i][Y[i].size() - 1];
        handler << endl;
    }
    handler.close();
}

///////////////////////////////////////////////////////
//Load Data Procedures

int CountInstances(string fileName) {
    fstream handler;
    handler.open(fileName);
    string line;

    int numInstances = 0;

    while (handler.eof() != true) {
        getline(handler, line);

        if (line != "") {
            numInstances++;
        }
    }
    handler.close();

    return numInstances;
}

void LoadData(string fileName) {
    fstream handler;

    datasetSize = CountInstances(fileName); // zlicz l. instancji w pliku


    for (int k = 0; k < Nn; k++) {
        vector<double> vec;
        X.push_back(vec);
    }

    handler.open(fileName);
    for (int i = 0; i < datasetSize; i++) {
        string line;
        getline(handler, line);
        stringstream linestream(line);
        string dataPortion;

        if (line != "") {
            IDS.push_back(i);
            for (int k = 0; k < Nn - 1; k++) {
                getline(linestream, dataPortion, ',');
                double value = stod(dataPortion);
                X[k].push_back(value);
            }

            getline(linestream, dataPortion, ' ');
            double value = stod(dataPortion);
            X[Nn - 1].push_back(value);
        }
    }
    handler.close();
}

//Clear all structures after each eSNN training and classification
void ClearStructures() {

    for (int i = 0; i < OutputNeurons.size(); i++) {
        delete OutputNeurons[i];
    }

    OutputNeurons.clear();
    X.clear();
    Y.clear();

    for (int k = 0; k < InputNeurons.size(); k++) {
        for (int j = 0; j < InputNeurons.size(); j++) {
            delete InputNeurons[k][j];
            delete GRFs[k][j];
        }
    }

    InputNeurons.clear();
    GRFs.clear();
    Wstream.clear();
    I_min.clear();
    I_max.clear();
    IDS.clear();
}

///////////////////
//Calculate metrics procedures

void CalculateRMSE() {
    vector<double> sums;

    for (int j = 1; j < Y[0].size(); j++) {
        sums.push_back(0);
    }

    for (int i = 0; i < Y.size(); i++) {
        for (int j = 1; j < Y[i].size(); j++) {
            sums[j - 1] += pow(X[0][Y[i][0] + j - 1] - Y[i][j], 2);
        }
    }

    cout << endl;

    for (int i = 0; i < sums.size(); i++) {
        cout << "hpred: " << i + 1 << " " << sqrt(sums[i] / Y.size()) << "-";
    }

    cout << endl;
}