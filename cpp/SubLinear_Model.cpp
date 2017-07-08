#define ILOUSESTL
using namespace std;

#include <ilcplex/ilocplex.h>
#include <vector>
#include "matrix.h"
#include "mex.h"
#include <random>

ILOSTLBEGIN
typedef vector<IloNumArray>    NumMatrix;
typedef vector<IloNumVarArray> NumVarMatrix;
typedef vector<IloBoolVarArray> BoolVarMatrix;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
#define param       prhs[0]
#define theta_in    prhs[1]
#define s_in        prhs[2]
#define optimal     plhs[0]
    
    if (nrhs != 3) {
        mexErrMsgTxt("Three input arguments required.");
    } else if (nlhs > 1) {
        mexErrMsgTxt("Too many output arguments.");
    }
    
    size_t n            = mxGetNumberOfElements(theta_in);
    size_t N            = mxGetN(s_in);
    size_t m            = mxGetM(s_in);
    
    mxArray *h_in       = mxGetField(param, 0, "h");
    mxArray *H_in       = mxGetField(param, 0, "H");
    mxArray *W_in       = mxGetField(param, 0, "W");
    mxArray *delta_in   = mxGetField(param, 0, "delta");
    
    size_t nh           = mxGetM(h_in);
    
    /* Initialization */
    const char *field_names[] = {"x","objective","diagnosis"};
    optimal = mxCreateStructMatrix(N, 1, 3, field_names); 
    
    IloEnv env;
    try {
        /* Define Additional Fixed Vectors */
        NumMatrix s(N);
        IloNumArray h(env,nh);
        IloNumArray theta(env,n);
        NumMatrix H(nh);
        NumMatrix W(nh);
        NumMatrix Htr(m);
        NumMatrix Wtr(n);
        IloNum delta;
        for (int i=0; i < N; i++) {
            s[i]        = IloNumArray(env,m);
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
            theta[i]    = mxGetPr(theta_in)[i];
            for (int j=0; j<nh; j++){
                Wtr[i][j] = mxGetPr(W_in)[i*nh+j];
            }
        }
        delta = mxGetScalar(delta_in);

        for (int k = 0; k < N; k++) {
            IloModel model(env);
            
            /* Define Decision Variables */  
            IloNumVarArray  x(env,n,-IloInfinity,IloInfinity);
            IloNumVarArray  lambda(env,nh,0,IloInfinity);
            IloNumVarArray  z(env,m,-IloInfinity,IloInfinity);
            IloNumArray     random_theta(env,n);
            
            /* Declare Constraints */
            double r = ((double) rand() / (RAND_MAX));
            model.add(IloScalProd(theta,x) - IloScalProd(s[k],z) - IloScalProd(h,lambda) <= delta * r);
            for (int j=0; j<m; j++)
                model.add( z[j] == IloScalProd(Htr[j],lambda) );
            for (int j=0; j<nh; j++)
                model.add( IloScalProd(W[j],x) >= IloScalProd(H[j],s[k]) + h[j]);
            for (int j=0; j<n; j++)
                model.add( theta[j] == IloScalProd(Wtr[j],lambda) );
            for (int j=0; j<n; j++){                
                double r = ((double) rand() / (RAND_MAX));
                if (theta[j] >= 0 )
                    random_theta[j] = r;
                else
                    random_theta[j] = -r;
            }

            /* Declare Objective */
            IloExpr objective(env);
            objective = IloScalProd(random_theta,x);
            model.add(IloMinimize(env,objective));
            objective.end();
            
            /* Solve the Problem */
            IloCplex cplex(model);
            cplex.setOut(env.getNullStream());
            cplex.solve();
            
            /* Save the Results */
            mxArray *x_out           = mxCreateDoubleMatrix(n, 1, mxREAL); 
            mxArray *objective_out   = mxCreateDoubleMatrix(1, 1, mxREAL); 
            mxArray *diagnosis_out   = mxCreateDoubleMatrix(1, 1, mxREAL); 
            double* out_x            = mxGetPr(x_out);
            double* out_objective    = mxGetPr(objective_out);
            double* out_diagnosis    = mxGetPr(diagnosis_out);
            *out_diagnosis           = cplex.getStatus();
            mxSetField(optimal, k, "diagnosis", diagnosis_out);
            if (*out_diagnosis != 2){
                    continue;
            }
            *out_objective           = cplex.getObjValue();         
            for (int i=0; i < n; i++)
                out_x[i]            = cplex.getValue(x[i]);
            mxSetField(optimal, k, "x", x_out);
            mxSetField(optimal, k, "objective", objective_out);            
            
            /* Close CPLEX and Model & Free Memory! */
            cplex.end();
            model.end();
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