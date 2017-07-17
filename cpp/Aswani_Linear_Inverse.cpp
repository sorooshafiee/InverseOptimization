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
#define data        prhs[1]
#define optimal     plhs[0]

    if (nrhs != 2) {
        mexErrMsgTxt("Two input arguments required.");
    } else if (nlhs > 1) {
        mexErrMsgTxt("Too many output arguments.");
    }

    mxArray *s_in       = mxGetField(data, 0, "s");
    mxArray *x_in       = mxGetField(data, 0, "x");    
    size_t n            = mxGetM(x_in);
    size_t N            = mxGetN(x_in);
    size_t m            = mxGetM(s_in);

    mxArray *h_in       = mxGetField(param, 0, "h");
    mxArray *H_in       = mxGetField(param, 0, "H");
    mxArray *W_in       = mxGetField(param, 0, "W");
    mxArray *set_in     = mxGetField(param, 0, "set_theta");

    size_t nh           = mxGetM(h_in);

    /* Initialization */
    const char *field_names[] = {"theta","objective","diagnosis"};
    optimal = mxCreateStructMatrix(1, 1, 3, field_names); 

    IloEnv env;
    try {
        /* Define Additional Fixed Vectors */
        NumMatrix x(N);
        NumMatrix s(N);
        IloNumArray h(env,nh);
        NumMatrix H(nh);
        NumMatrix W(nh);
        NumMatrix Htr(m);
        NumMatrix Wtr(n);
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
        IloNum bigM = 100;

        IloModel model(env);

        /* Define Decision Variables */  
        IloNumVarArray  theta(env,n,-IloInfinity,IloInfinity);
        NumVarMatrix    y(N);
        NumVarMatrix    lambda(N);                   
        BoolVarMatrix   b1(N);
        BoolVarMatrix   b2(N);
        for (int i=0; i < N; i++) {             
            y[i]        = IloNumVarArray(env,n,-IloInfinity,IloInfinity);
            lambda[i]   = IloNumVarArray(env,nh,0,IloInfinity);
            b1[i]       = IloBoolVarArray(env,nh);
            b2[i]       = IloBoolVarArray(env,nh);
        } 

        /* Declare Constraints */
        for (int i=0; i < N; i++) {
            for (int j=0; j<n; j++)             
                model.add( theta[j] == IloScalProd(Wtr[j],lambda[i]) );
            for (int j=0; j<nh; j++){
                model.add( IloScalProd(W[j],y[i]) >= IloScalProd(H[j],s[i]) + h[j] );
                model.add( IloScalProd(W[j],y[i]) -  IloScalProd(H[j],s[i]) - h[j] <= bigM * b1[i][j] );
                model.add( lambda[i][j] <= bigM * b2[i][j] );
                model.add( b1[i][j] + b2[i][j] == 1 );
            }
        }

        /* Constrains corresponds to the set Theta */
        if (set_in != NULL) { 
            mxArray *center_in  = mxGetField(set_in, 0, "center");
            mxArray *radius_in  = mxGetField(set_in, 0, "radius");
            mxArray *norm_in    = mxGetField(set_in, 0, "pnorm");
            IloNumArray center(env,n);
            for (int i=0; i < n; i++)
                center[i]       = mxGetPr(center_in)[i];
            IloNum radius       = mxGetScalar(radius_in);
            double nrm          = mxGetScalar(norm_in);
            if ( nrm == 1 ) {
                IloNumVarArray  rr(env,n,0,IloInfinity);
                for (int i=0; i<n; i++){
                    model.add( theta[i] - center[i] <= rr[i] );
                    model.add( center[i] - theta[i] <= rr[i] );
                }
                model.add( IloSum(rr) <= radius );
            }
            if ( nrm == 2 ) {
                IloNumVarArray  rr(env,n,0,IloInfinity);
                for (int i=0; i<n; i++){
                    model.add( theta[i] - center[i] == rr[i] );
                }
                model.add( IloScalProd(rr,rr) <= radius * radius );
            }
            if ( mxIsInf(nrm) ) {
                for (int i=0; i<n; i++){
                    model.add( theta[i] - center[i] <= radius );
                    model.add( center[i] - theta[i] <= radius );
                }
            }
        }

        /* Declare Objective */
        IloExpr objective(env);
        for (int i=0; i<N; i++)
            objective += IloScalProd(y[i],y[i]) - 2 * IloScalProd(x[i],y[i]);
        model.add(IloMinimize(env,objective));
        objective.end();

        /* Solve the Problem */
        IloCplex cplex(model);
        cplex.setOut(env.getNullStream());
        cplex.setParam(IloCplex::TiLim, 120); 
        cplex.solve();

        /* Save the Results */
        mxArray *theta_out = mxCreateDoubleMatrix(n, 1, mxREAL); 
        mxArray *objective_out = mxCreateDoubleMatrix(1, 1, mxREAL); 
        mxArray *diagnosis_out = mxCreateDoubleMatrix(1, 1, mxREAL); 
        double* out_theta = mxGetPr(theta_out);
        double* out_objective = mxGetPr(objective_out);
        double* out_diagnosis = mxGetPr(diagnosis_out);
        *out_diagnosis = cplex.getStatus();
        mxSetField(optimal, 0, "diagnosis", diagnosis_out);
        *out_objective = cplex.getObjValue();           
        for (int i=0; i < n; i++)
        {
            out_theta[i] = cplex.getValue(theta[i]);
        }
        mxSetField(optimal, 0, "theta", theta_out);
        mxSetField(optimal, 0, "objective", objective_out);         

        /* Close CPLEX and Model & Free Memory! */
        cplex.end();
        model.end();

        /* Close CPLEX and Model & Free Memory! */
        cplex.end();
        model.end();
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