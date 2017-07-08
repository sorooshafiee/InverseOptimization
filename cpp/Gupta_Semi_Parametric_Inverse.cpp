#define ILOUSESTL
using namespace std;

#include <ilcplex/ilocplex.h>
#include <string>
#include <vector>
#include "matrix.h"
#include "mex.h"
#include "math.h"

ILOSTLBEGIN
typedef vector<IloNumArray>    NumMatrix;
typedef vector<IloNumVarArray> NumVarMatrix;
typedef vector<IloBoolVarArray> BoolVarMatrix;

double * distSqrd(mxArray *X1, mxArray *X2)
{
    size_t n = mxGetM(X1);
    size_t N1 = mxGetN(X1);
    size_t N2 = mxGetN(X2);
    double *D2 = (double *) mxMalloc(N1 * N2 * sizeof(double));
    for (int i = 0; i < N1; i++){
        for (int j = 0; j < N2; j++){
            double acc = 0.0;
            for (int k = 0; k < n; k++){
                double tmp1 = mxGetPr(X1)[k+i*n];
                double tmp2 = mxGetPr(X2)[k+j*n];
                acc += pow(tmp1-tmp2,2);
            }
            D2[i+j*N1] = acc;
        }
    }
    return D2;
}

double * Kernel_Function(mxArray *X1, mxArray *X2, char *kernel_, double lengthScale)
{
    size_t n = mxGetM(X1);
    size_t N1 = mxGetN(X1);
    size_t N2 = mxGetN(X2);
    double eta = 1/pow(lengthScale,2);
    double *K = (double *) mxMalloc(N1 * N2 * sizeof(double));
    if (strcmp(kernel_,"linear") == 0){
        for (int i = 0; i < N1; i++){
            for (int j = 0; j < N2; j++){
                double acc = 0.0;
                for (int k = 0; k < n; k++){
                    double tmp1 = mxGetPr(X1)[k+i*n];
                    double tmp2 = mxGetPr(X2)[k+j*n];
                    acc += tmp1*tmp2;
                }
                K[i+j*N1] = acc;
            }
        }
    }
    else if (strncmp(kernel_,"poly",4) == 0){
        char *temp = kernel_ + 4;
        int p;
        p = stoi(temp);
        for (int i = 0; i < N1; i++){
            for (int j = 0; j < N2; j++){
                double acc = 0.0;
                for (int k = 0; k < n; k++){
                    double tmp1 = mxGetPr(X1)[k+i*n];
                    double tmp2 = mxGetPr(X2)[k+j*n];
                    acc += tmp1*tmp2;
                }
                K[i+j*N1] = pow(acc+1,p);
            }
        }
    }
    else if(strcmp(kernel_,"gauss") == 0){
        double *D2 = distSqrd(X1, X2);
        for (int i = 0; i < N1; i++){
            for (int j = 0; j < N2; j++){    
                double tmp = -eta*D2[i+j*N1];
                K[i+j*N1] = exp(tmp);
            }
        } 
    }
    else
        mexErrMsgTxt ("Unrecognised kernel function type");
    return K;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
#define param       prhs[0]
#define data        prhs[1]
#define optimal     plhs[0]
    
    if (nrhs != 2) {
        mexErrMsgTxt("Two input arguments required.");
    } else if (nlhs > 1) {
        mexErrMsgTxt("Too many output arguments.");
    }
    
    mxArray *s_in  = mxGetField(data, 0, "s");
    mxArray *x_in   = mxGetField(data, 0, "x");
    
    size_t n = mxGetM(x_in);
    size_t N = mxGetN(x_in);
    size_t m = mxGetM(s_in);    
    
    mxArray *h_in       = mxGetField(param, 0, "h");
    mxArray *H_in       = mxGetField(param, 0, "H");
    mxArray *W_in       = mxGetField(param, 0, "W");
    mxArray *pnorm_in   = mxGetField(param, 0, "pnorm");
    mxArray *kernel_in  = mxGetField(param, 0, "kernel");
    mxArray *sigma_in   = mxGetField(param, 0, "sigma");
    mxArray *kappa_in   = mxGetField(param, 0, "kappa");
    
    size_t kernel_len = mxGetN(kernel_in) + 1;
    size_t sigma_len = mxGetNumberOfElements(sigma_in);
    size_t kappa_len = mxGetNumberOfElements(kappa_in);
    size_t nh = mxGetNumberOfElements(h_in);
    double pnorm = mxGetScalar(pnorm_in); 
    char *kernel_ = (char*) mxMalloc(kernel_len);
    double status = mxGetString(kernel_in, kernel_, (mwSize)kernel_len);  
    double *sigma = (double*) mxMalloc(sigma_len * sizeof(double));   
    for (int i=0; i<sigma_len; i++)
        sigma[i] = mxGetPr(sigma_in)[i];    
    
    
    /* Initialization */
    const char *field_names[] = {"alpha","objective","diagnosis"};
    optimal = mxCreateStructMatrix(sigma_len, kappa_len, 3, field_names); 
    
    IloEnv env;
    try {
        /* Define Additional Fixed Vectors */
        NumMatrix       x(N);
        NumMatrix       s(N);
        IloNumArray     h(env,nh);
        NumMatrix       H(nh);
        NumMatrix       W(nh);
        NumMatrix       Htr(m);
        NumMatrix       Wtr(n);        
        IloNumArray     kappa(env,kappa_len);
        for (int i=0; i < N; i++) {
            x[i]        = IloNumArray(env,n);
            s[i]        = IloNumArray(env,m);
            for (int j=0; j<n; j++){
                x[i][j] = mxGetPr(x_in)[j+i*n];
            }
            for (int j=0; j<m; j++){
                s[i][j] = mxGetPr(s_in)[j+i*m];
            }
        }        
        for (int i=0; i < nh; i++) {
            H[i]        = IloNumArray(env,m);
            W[i]        = IloNumArray(env,n);
            h[i]        = mxGetPr(h_in)[i];
            for (int j=0; j<m; j++){
                H[i][j] = mxGetPr(H_in)[j*nh+i];
            }
            for (int j=0; j<n; j++){
                W[i][j] = mxGetPr(W_in)[j*nh+i];
            }
        }
        for (int i=0; i<m; i++){
            Htr[i]      = IloNumArray(env,nh);          
            for (int j=0; j<nh; j++){
                Htr[i][j] = mxGetPr(H_in)[i*nh+j];
            }
        }
        for (int i=0; i<n; i++){
            Wtr[i]      = IloNumArray(env,nh);
            for (int j=0; j<nh; j++){
                Wtr[i][j] = mxGetPr(W_in)[i*nh+j];
            }
        }          
        for (int i=0; i < kappa_len; i++)
            kappa[i] = mxGetPr(kappa_in)[i];        
        

        for (int i=0; i<sigma_len; i++){

            /* Kernel Matrix Computation */
            NumMatrix  K(N);
            double *Km = Kernel_Function(x_in,x_in,kernel_,sigma[i]);
            for (int i=0; i < N; i++) {
                K[i] = IloNumArray(env,N);
                for (int j=0; j<N; j++)
                    K[i][j] = Km[j+i*N];
            }
            for (int j=0; j<kappa_len; j++){

                IloModel model(env);
                /* Define Decision Variables */
                IloNumVarArray  r(env,N,0,IloInfinity); /* I use r instead of epsilon in Gupta's paper */
                NumVarMatrix    alpha(n);
                NumVarMatrix    mu(N);
                NumVarMatrix    zx(N);          
                NumVarMatrix    zt(N);
                for (int i=0; i < N; i++) {
                    mu[i]       = IloNumVarArray(env,nh,0,IloInfinity);
                    zt[i]       = IloNumVarArray(env,m,-IloInfinity,IloInfinity);
                    zx[i]       = IloNumVarArray(env,n,-IloInfinity,IloInfinity);
                } 
                for (int j=0; j < n; j++) 
                    alpha[j]    = IloNumVarArray(env,N,-IloInfinity,IloInfinity);
                
                /* Declare Constraints */
                for (int i=0; i<N; i++) {
                    model.add(-IloScalProd(mu[i],h) + IloScalProd(zx[i],x[i]) + IloScalProd(zt[i],s[i]) <= r[i]);
                    for (int j=0; j<m; j++)
                        model.add(zt[i][j] == -IloScalProd(Htr[j],mu[i]));
                    for (int j=0; j<n; j++){
                        model.add(IloScalProd(Wtr[j],mu[i]) == s[i][j] + IloScalProd(alpha[j],K[i]));
                        model.add(IloScalProd(Wtr[j],mu[i]) == zx[i][j]);
                    }
                }   
                if ( pnorm == 1 ) 
                    model.add(IloSum(r) <= kappa[j]);
                if ( pnorm == 2 )
                    model.add(IloScalProd(r,r) <= kappa[j]*kappa[j]);
                if ( mxIsInf(pnorm) ){
                    for (int i=0; i<N; i++)
                        model.add(r[i] <= kappa[j]);
                }
                
                /* Declare Objective */
                IloExpr objective(env);
                for (int i=0; i<N; i++)
                    for (int j=0; j<n; j++)
                        objective += IloScalProd(alpha[j],K[i]) * alpha[j][i];                
                model.add(IloMinimize(env,objective));
                objective.end();
                
                /* Solve the Problem */
                IloCplex cplex(model);
                cplex.setOut(env.getNullStream());
                cplex.solve();
                
                /* Save the Results */
                mxArray *alpha_out = mxCreateDoubleMatrix(n, N, mxREAL); 
                mxArray *objective_out = mxCreateDoubleMatrix(1, 1, mxREAL); 
                mxArray *diagnosis_out = mxCreateDoubleMatrix(1, 1, mxREAL); 
                double* out_alpha = mxGetPr(alpha_out);
                double* out_objective = mxGetPr(objective_out);
                double* out_diagnosis = mxGetPr(diagnosis_out);
                *out_diagnosis = cplex.getStatus();
                mxSetField(optimal, i + j*sigma_len, "diagnosis", diagnosis_out);
                *out_objective = cplex.getObjValue();       
                for (int i=0; i < N; i++)
                    for (int j=0; j < n; j++)
                        out_alpha[j+i*n] = cplex.getValue(alpha[j][i]);
                mxSetField(optimal, i + j*sigma_len, "alpha", alpha_out);
                mxSetField(optimal, i + j*sigma_len, "objective", objective_out);         
                
                /* Close CPLEX and Model & Free Memory! */
                cplex.end();
                model.end();
            }
        }
    }
    catch (IloException& ex) {
        mexErrMsgTxt ("Error Stoc");
    }
    catch (...) {
        mexErrMsgTxt ("Error");
    }
    env.end();
    return;
}