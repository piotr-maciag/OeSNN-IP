
#include "eSNN.h"

int datasetSize;

vector<int> CNOsize;
int Wsize;
int NOsize;
int NIsize;
double simTr;
double mod;
int K;

vector<vector<double>> Wstream;

int n;
int m;
double H;

double Dub;
vector<double> I_min;
vector<double> I_max;
int Ninit, Nsize;


vector<vector<neuron *>> OutputNeurons; //Pointers to output neurons (output neuron repositories)
vector<vector<double >> X; //input data streams
vector<vector<double >> Y; //predicted values of pollution level
vector<int> IDS;

vector<vector<GRFstruct>> GRFs; //input GRFs
vector<vector<inputNeuron>> InputNeurons;


bool compFiringTime(const inputNeuron &nI1, const inputNeuron &nI2) { //comparator of firing times
    if (nI1.firingTime != nI2.firingTime) {
        return nI1.firingTime < nI2.firingTime;
    } else {
        return nI1.id < nI2.id;
    }
}

void InitializeInputLayer(const vector<vector<double>> &Windows) { //intialize input layer of OeSNN

    for (int k = 0; k < Windows.size(); k++) {
        for (int j = 0; j < InputNeurons[k].size(); j++) {
            InputNeurons[k][j].order.clear();
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

        double width = (I_max[k] - I_min[k]) / NIsize;

        for (int j = 0; j < GRFs[k].size(); j++) {
            double mu = I_min[k] + (j + 1 - 0.5) * width;
            GRFs[k][j].mu = mu;
            //  cout << j << " " << mu << ";";
        }
        //  cout << endl;
        /*
        for (int u = Windows[k].size() - 1; u >= 0; u--) {
            vector<inputNeuron> sortInputNeurons;
            for (int j = 0; j < GRFs[k].size(); j++) {

                double firingTime = abs(Windows[k][u] - GRFs[k][j].mu); //(1 - exc);


                inputNeuron newIN = {j, k, firingTime};
                sortInputNeurons.push_back(newIN);
            }

            sort(sortInputNeurons.begin(), sortInputNeurons.end(), compFiringTime);


            for (int j = 0; j < sortInputNeurons.size(); j++) {
                InputNeurons[k][sortInputNeurons[j].id].order.push_back(ord);

                ord++;
            }

        }*/

        for (int u = Windows[k].size() - 1; u >= 0; u--) {

            int j = floor((Windows[k][u] - I_min[k]) / width) + 1;
            int l;
            if (j - 1 < NIsize - j) { l = j - 1; } else { l = NIsize - j; }

            // cout << j << " " << l << endl;
            //cout << Windows[k][u] <<  endl;

            GRFs[k][j - 1].rank = 0;

            if (Windows[k][u] < GRFs[k][j].mu) {
                for (int n = 1; n <= l; n++) {
                    GRFs[k][j - n - 1].rank = 2 * n - 1;
                    GRFs[k][j + n - 1].rank = 2 * n;
                }
                for (int n = 1; n <= j - 1 - l; n++) //n is k in algorithms
                {
                    GRFs[k][j - l - n - 1].rank = 2 * l - 1 + n;
                }
                for (int n = 1; n <= NIsize - j; n++) //n is k in algorithms
                {
                    GRFs[k][j + l + n - 1].rank = 2 * l + n;
                }
            } else {
                for (int n = 1; n <= l; n++) {
                    GRFs[k][j - n - 1].rank = 2 * n;
                    GRFs[k][j + n - 1].rank = 2 * n - 1;
                }
                for (int n = 1; n <= j - 1 - l; n++) //n is k in algorithms
                {
                    GRFs[k][j - l - n - 1].rank = 2 * l + n;
                }
                for (int n = 1; n <= NIsize - j - 1; n++) //n is k in algorithms
                {
                    GRFs[k][j + l + n - 1].rank = 2 * l + n - 1;
                }
            }

            for (int jj = 0; jj < InputNeurons[k].size(); jj++) {
                InputNeurons[k][jj].order.push_back(GRFs[k][jj].rank + (Wsize - u - 1) * NIsize);

            }
            //  cout << endl;
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
                n_c->s_weights[l][j] += pow(mod, InputNeurons[l][j].order[u]);
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


        for (int k = 0; k < n; k++) {
            neuron *n_c = new neuron;
            InitializeNeuron(n_c, X[k][h], h);

            UpdateRepository(n_c, k);
        }

        for (int k = 0; k < n + m; k++) {
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

neuron *FindMostSimilar(neuron *n_c, int F_k) { //find mos similar neurons in terms of synaptic weights

    double minDist = CalculateDistance(n_c->s_weights, OutputNeurons[F_k][0]->s_weights);
    double minIdx = 0;

    if (OutputNeurons[F_k].size() > 1) {
        for (int i = 1; i < OutputNeurons[F_k].size(); i++) {
            double dist = CalculateDistance(n_c->s_weights, OutputNeurons[F_k][i]->s_weights);
            if (dist < minDist) {
                minDist = dist;
                minIdx = i;
            }
        }
    }
    return OutputNeurons[F_k][minIdx];
}

void ReplaceOldest(neuron *n_c, int F_k) { //replace the oldets neuron in output repostiory
    int oldest = OutputNeurons[F_k][0]->additionTime;
    int oldestIdx = 0;

    for (int i = 1; i < OutputNeurons[F_k].size(); i++) {
        if (oldest > OutputNeurons[F_k][i]->additionTime) {
            oldest = OutputNeurons[F_k][i]->additionTime;
            oldestIdx = i;
        }
    }

    delete OutputNeurons[F_k][oldestIdx];
    OutputNeurons[F_k][oldestIdx] = n_c;

}


void UpdateRepository(neuron *n_c, int F_k) { //Update neuron n_s in output repository

    neuron *n_s;

    if (OutputNeurons[F_k].size() > 0) {
        n_s = FindMostSimilar(n_c, F_k);
    }

    if (OutputNeurons[F_k].size() > 0 && CalculateDistance(n_c->s_weights, n_s->s_weights) < simTr * Dub) {
        for (int k = 0; k < n_s->s_weights.size(); k++) {
            for (int j = 0; j < n_s->s_weights[k].size(); j++) {
                n_s->s_weights[k][j] = (n_c->s_weights[k][j] + n_s->s_weights[k][j] * n_s->M) / (n_s->M + 1);
            }
        }

        n_s->outputValue = (n_c->outputValue + n_s->outputValue * n_s->M) / (n_s->M + 1);
        n_s->additionTime = (n_c->additionTime + n_s->additionTime * n_s->M) / (n_s->M + 1);
        n_s->M += 1;
        delete n_c;
    } else if (OutputNeurons[F_k].size() < NOsize) {
        OutputNeurons[F_k].push_back(n_c);
    } else {
        ReplaceOldest(n_c, F_k);
    }
}

double PredictValue(int F_k) {
    for (int i = 0; i < OutputNeurons[F_k].size(); i++) {
        OutputNeurons[F_k][i]->PSP = 0;
    }

    for (int l = 0; l < InputNeurons.size(); l++) {
        for (int j = 0; j < InputNeurons[l].size(); j++) {
            for (int i = 0; i < OutputNeurons[F_k].size(); i++) {
                for (int u = 0; u < Wsize; u++) {
                    OutputNeurons[F_k][i]->PSP += OutputNeurons[F_k][i]->s_weights[l][j] *
                                                  pow(mod, InputNeurons[l][j].order[u]);
                }
            }
        }
    }


    double maxPSP = -1;
    double maxVals;
    int countMax = 0;

    for (int i = 0; i < OutputNeurons[F_k].size(); i++) {
        if (maxPSP < OutputNeurons[F_k][i]->PSP) {
            maxVals = OutputNeurons[F_k][i]->outputValue;
            countMax = 1;
            maxPSP = OutputNeurons[F_k][i]->PSP;
        } else if (maxPSP == OutputNeurons[F_k][i]->PSP) {
            maxVals += OutputNeurons[F_k][i]->outputValue;
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

    for (int k = 0; k < n + m; k++) {
        CNOsize.push_back(0);
    }

    for (int k = 0; k < n + m; k++) {
        vector<GRFstruct> GRFSvec;
        for (int j = 0; j < NIsize; j++) {
            GRFstruct newGRF;
            GRFSvec.push_back(newGRF);
        }
        GRFs.push_back(GRFSvec);
    }

    for (int k = 0; k < n + m; k++) {
        vector<inputNeuron> InputNeuronsVect;
        for (int j = 0; j < NIsize; j++) {
            inputNeuron newInputNeuron = {j, k};
            InputNeuronsVect.push_back(newInputNeuron);
        }
        InputNeurons.push_back(InputNeuronsVect);
    }

    for (int k = 0; k < n; k++) {
        vector<neuron *> vec;
        OutputNeurons.push_back(vec);
    }

    vector<vector<double>> Windows;

    for (int k = 0; k < n + m; k++) {
        vector<double> win;
        for (int h = 0; h < Wsize; h++) {
            win.push_back(X[k][h]);
        }
        Windows.push_back(win);
        Wstream.push_back(win);
    }

    for (int h = Wsize; h < Ninit; h++) {
        for (int k = 0; k < n + m; k++) {
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

        for (int k = 0; k < n; k++) {
            neuron *n_c = new neuron;
            InitializeNeuron(n_c, X[k][h], h);
            UpdateRepository(n_c, k);
        }

        for (int k = 0; k < n + m; k++) {
            Windows[k].erase(Windows[k].begin());
            Windows[k].push_back(X[k][h]);
            WP[k] = Windows[k];
        }

        vector<double> y;
        y.push_back(h);

        for (int hpred = 1; hpred <= H; hpred++) {
            InitializeInputLayer(WP);

            for (int k = 0; k < n; k++) {

                double y_h_hpred = PredictValue(k);
                // WP[k].erase(WP[k].begin());
                //WP[k].push_back(y_h_hpred);

                if (k == 0) {
                    y.push_back(y_h_hpred);
                }
            }

            for (int k = n; k < n + m; k++) {

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


    for (int k = 0; k < n + m; k++) {
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
            for (int k = 0; k < n + m - 1; k++) {
                getline(linestream, dataPortion, ',');
                double value = stod(dataPortion);
                X[k].push_back(value);
            }

            getline(linestream, dataPortion, ' ');
            double value = stod(dataPortion);
            X[n + m - 1].push_back(value);
        }
    }
    handler.close();
}

//Clear all structures after each eSNN training and classification
void ClearStructures() {
    for (int k = 0; k < OutputNeurons.size(); k++) {
        for (int i = 0; i < OutputNeurons[k].size(); i++) {
            delete OutputNeurons[k][i];
        }
    }
    OutputNeurons.clear();
    X.clear();
    Y.clear();

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